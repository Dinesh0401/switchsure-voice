"""Bounded pilot job lifecycle with durable metadata and ephemeral content.

Payloads and results deliberately remain process-local and are excluded from
representations. SQLite contains only content-free lifecycle metadata. A
process restart therefore fails incomplete or content-bearing jobs closed
instead of pretending that sensitive content survived.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import TypeVar, cast
from uuid import uuid4

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")

_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RetentionClass(StrEnum):
    EPHEMERAL = "ephemeral"


class JobQueueError(RuntimeError):
    """Base class for safe job lifecycle failures."""


class QueueFullError(JobQueueError):
    """Raised when bounded active capacity is exhausted."""


class JobNotFoundError(JobQueueError):
    """Used for both absent jobs and ownership mismatches."""


class InvalidJobTransitionError(JobQueueError):
    """Raised when a lifecycle transition would be unsafe."""


class JobResultUnavailableError(JobQueueError):
    """Raised when ephemeral result content is no longer in memory."""


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    job_id: str
    state: JobState
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    attempts: int
    max_attempts: int
    retention_class: RetentionClass
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class JobLease[PayloadT]:
    job_id: str
    attempt: int
    lease_expires_at: datetime
    payload: PayloadT = field(repr=False)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validated_identity(name: str, value: str) -> str:
    stripped = value.strip()
    if not stripped or len(stripped) > 256:
        raise ValueError(f"{name} must contain 1 to 256 characters")
    return stripped


class PilotJobQueue[PayloadT, ResultT]:
    """Thread-safe bounded queue with SQLite metadata and ephemeral content."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        capacity: int = 32,
        lease_timeout: timedelta = timedelta(minutes=2),
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if lease_timeout <= timedelta(0):
            raise ValueError("lease_timeout must be positive")
        self._capacity = capacity
        self._lease_timeout = lease_timeout
        self._clock = clock
        self._id_factory = id_factory or (lambda: f"job_{uuid4().hex}")
        self._lock = RLock()
        self._pending: deque[str] = deque()
        self._payloads: dict[str, PayloadT] = {}
        self._results: dict[str, ResultT] = {}
        self._connection = sqlite3.connect(str(database_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._create_schema()
        self._recover_process_local_state()

    def __enter__(self) -> PilotJobQueue[PayloadT, ResultT]:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pilot_jobs (
                    job_id TEXT PRIMARY KEY,
                    owner_digest TEXT NOT NULL,
                    idempotency_digest TEXT,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    retention_seconds INTEGER NOT NULL,
                    attempts INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    retention_class TEXT NOT NULL,
                    error_code TEXT,
                    worker_digest TEXT,
                    lease_expires_at TEXT,
                    UNIQUE(owner_digest, idempotency_digest)
                )
                """
            )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    def _recover_process_local_state(self) -> None:
        now = self._now()
        rows = self._connection.execute(
            """
            SELECT job_id, retention_seconds FROM pilot_jobs
            WHERE state IN (?, ?, ?)
            """,
            (JobState.QUEUED, JobState.RUNNING, JobState.SUCCEEDED),
        ).fetchall()
        with self._connection:
            self._connection.executemany(
                """
                UPDATE pilot_jobs
                SET state = ?, updated_at = ?, expires_at = ?, error_code = ?,
                    worker_digest = NULL, lease_expires_at = NULL
                WHERE job_id = ?
                """,
                (
                    (
                        JobState.FAILED,
                        now.isoformat(),
                        (now + timedelta(seconds=int(row["retention_seconds"]))).isoformat(),
                        "process_restart_ephemeral_data_unavailable",
                        str(row["job_id"]),
                    )
                    for row in rows
                ),
            )

    def submit(
        self,
        *,
        owner_id: str,
        payload: PayloadT,
        idempotency_key: str | None = None,
        retention_ttl: timedelta = timedelta(minutes=15),
        max_attempts: int = 2,
    ) -> JobSnapshot:
        owner_digest = _digest(_validated_identity("owner_id", owner_id))
        idempotency_digest = None
        if idempotency_key is not None:
            idempotency_digest = _digest(_validated_identity("idempotency_key", idempotency_key))
        retention_seconds = int(retention_ttl.total_seconds())
        if retention_seconds < 1:
            raise ValueError("retention_ttl must be at least one second")
        if not 1 <= max_attempts <= 10:
            raise ValueError("max_attempts must be between 1 and 10")

        with self._lock:
            now = self._now()
            self._purge_expired_locked(now)
            if idempotency_digest is not None:
                existing = self._connection.execute(
                    """
                    SELECT * FROM pilot_jobs
                    WHERE owner_digest = ? AND idempotency_digest = ?
                    """,
                    (owner_digest, idempotency_digest),
                ).fetchone()
                if existing is not None:
                    return self._snapshot(existing)
            if self._active_count_locked() >= self._capacity:
                raise QueueFullError("job queue capacity is exhausted")

            job_id = self._id_factory()
            if not re.fullmatch(r"job_[0-9a-f]{32}", job_id):
                raise ValueError(
                    "id_factory must return an opaque job_<32 lowercase hex> identifier"
                )
            expires_at = now + retention_ttl
            try:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT INTO pilot_jobs (
                            job_id, owner_digest, idempotency_digest, state,
                            created_at, updated_at, expires_at, retention_seconds,
                            attempts, max_attempts, retention_class
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                        """,
                        (
                            job_id,
                            owner_digest,
                            idempotency_digest,
                            JobState.QUEUED,
                            now.isoformat(),
                            now.isoformat(),
                            expires_at.isoformat(),
                            retention_seconds,
                            max_attempts,
                            RetentionClass.EPHEMERAL,
                        ),
                    )
            except sqlite3.IntegrityError as error:
                raise JobQueueError("job metadata could not be created") from error
            self._payloads[job_id] = payload
            self._pending.append(job_id)
            return self.snapshot(owner_id=owner_id, job_id=job_id)

    def lease_next(self, *, worker_id: str) -> JobLease[PayloadT] | None:
        worker_digest = _digest(_validated_identity("worker_id", worker_id))
        with self._lock:
            now = self._now()
            self._reap_stalled_locked(now)
            while self._pending:
                job_id = self._pending.popleft()
                row = self._row(job_id)
                if row is None or row["state"] != JobState.QUEUED:
                    continue
                payload = self._payloads.get(job_id)
                if payload is None:
                    self._fail_locked(
                        job_id,
                        error_code="ephemeral_payload_unavailable",
                        now=now,
                    )
                    continue
                attempt = int(row["attempts"]) + 1
                lease_expires_at = now + self._lease_timeout
                with self._connection:
                    self._connection.execute(
                        """
                        UPDATE pilot_jobs
                        SET state = ?, attempts = ?, updated_at = ?,
                            worker_digest = ?, lease_expires_at = ?, error_code = NULL
                        WHERE job_id = ?
                        """,
                        (
                            JobState.RUNNING,
                            attempt,
                            now.isoformat(),
                            worker_digest,
                            lease_expires_at.isoformat(),
                            job_id,
                        ),
                    )
                return JobLease(
                    job_id=job_id,
                    attempt=attempt,
                    lease_expires_at=lease_expires_at,
                    payload=payload,
                )
            return None

    def complete(self, *, job_id: str, worker_id: str, result: ResultT) -> JobSnapshot:
        worker_digest = _digest(_validated_identity("worker_id", worker_id))
        with self._lock:
            row = self._require_worker_lease(job_id, worker_digest)
            now = self._now()
            expires_at = now + timedelta(seconds=int(row["retention_seconds"]))
            with self._connection:
                self._connection.execute(
                    """
                    UPDATE pilot_jobs
                    SET state = ?, updated_at = ?, expires_at = ?, error_code = NULL,
                        worker_digest = NULL, lease_expires_at = NULL
                    WHERE job_id = ?
                    """,
                    (JobState.SUCCEEDED, now.isoformat(), expires_at.isoformat(), job_id),
                )
            self._payloads.pop(job_id, None)
            self._results[job_id] = result
            return self._snapshot(self._required_row(job_id))

    def fail(self, *, job_id: str, worker_id: str, error_code: str) -> JobSnapshot:
        worker_digest = _digest(_validated_identity("worker_id", worker_id))
        with self._lock:
            self._require_worker_lease(job_id, worker_digest)
            return self._fail_locked(job_id, error_code=error_code, now=self._now())

    def retry(self, *, owner_id: str, job_id: str, payload: PayloadT) -> JobSnapshot:
        with self._lock:
            row = self._owned_row(owner_id, job_id)
            if row["state"] != JobState.FAILED:
                raise InvalidJobTransitionError("only failed jobs can be retried")
            if int(row["attempts"]) >= int(row["max_attempts"]):
                raise InvalidJobTransitionError("job retry limit is exhausted")
            if self._active_count_locked() >= self._capacity:
                raise QueueFullError("job queue capacity is exhausted")
            now = self._now()
            with self._connection:
                self._connection.execute(
                    """
                    UPDATE pilot_jobs
                    SET state = ?, updated_at = ?, error_code = NULL
                    WHERE job_id = ?
                    """,
                    (JobState.QUEUED, now.isoformat(), job_id),
                )
            self._payloads[job_id] = payload
            self._results.pop(job_id, None)
            self._pending.append(job_id)
            return self._snapshot(self._required_row(job_id))

    def cancel(self, *, owner_id: str, job_id: str) -> JobSnapshot:
        with self._lock:
            row = self._owned_row(owner_id, job_id)
            if row["state"] not in {JobState.QUEUED, JobState.RUNNING}:
                raise InvalidJobTransitionError("only queued or running jobs can be cancelled")
            now = self._now()
            expires_at = now + timedelta(seconds=int(row["retention_seconds"]))
            with self._connection:
                self._connection.execute(
                    """
                    UPDATE pilot_jobs
                    SET state = ?, updated_at = ?, expires_at = ?, error_code = ?,
                        worker_digest = NULL, lease_expires_at = NULL
                    WHERE job_id = ?
                    """,
                    (
                        JobState.CANCELLED,
                        now.isoformat(),
                        expires_at.isoformat(),
                        "cancelled_by_owner",
                        job_id,
                    ),
                )
            self._payloads.pop(job_id, None)
            self._results.pop(job_id, None)
            return self._snapshot(self._required_row(job_id))

    def snapshot(self, *, owner_id: str, job_id: str) -> JobSnapshot:
        with self._lock:
            return self._snapshot(self._owned_row(owner_id, job_id))

    def result(self, *, owner_id: str, job_id: str) -> ResultT:
        with self._lock:
            row = self._owned_row(owner_id, job_id)
            if row["state"] != JobState.SUCCEEDED:
                raise InvalidJobTransitionError("job has not succeeded")
            try:
                return self._results[job_id]
            except KeyError as error:
                raise JobResultUnavailableError("ephemeral job result is unavailable") from error

    def reap_stalled(self) -> int:
        with self._lock:
            return self._reap_stalled_locked(self._now())

    def purge_expired(self) -> int:
        with self._lock:
            return self._purge_expired_locked(self._now())

    def healthcheck(self) -> bool:
        """Probe metadata storage without exposing job or content fields."""

        with self._lock:
            try:
                row = self._connection.execute("SELECT 1 AS healthy").fetchone()
            except sqlite3.Error:
                return False
            return row is not None and int(row["healthy"]) == 1

    def _active_count_locked(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM pilot_jobs WHERE state IN (?, ?)",
            (JobState.QUEUED, JobState.RUNNING),
        ).fetchone()
        assert row is not None
        return int(row["count"])

    def _reap_stalled_locked(self, now: datetime) -> int:
        rows = self._connection.execute(
            """
            SELECT job_id FROM pilot_jobs
            WHERE state = ? AND lease_expires_at <= ?
            """,
            (JobState.RUNNING, now.isoformat()),
        ).fetchall()
        for row in rows:
            self._fail_locked(
                str(row["job_id"]),
                error_code="worker_lease_timeout",
                now=now,
            )
        return len(rows)

    def _purge_expired_locked(self, now: datetime) -> int:
        rows = self._connection.execute(
            """
            SELECT job_id FROM pilot_jobs
            WHERE state NOT IN (?, ?) AND expires_at <= ?
            """,
            (JobState.QUEUED, JobState.RUNNING, now.isoformat()),
        ).fetchall()
        job_ids = [str(row["job_id"]) for row in rows]
        if job_ids:
            with self._connection:
                self._connection.executemany(
                    "DELETE FROM pilot_jobs WHERE job_id = ?",
                    ((job_id,) for job_id in job_ids),
                )
        for job_id in job_ids:
            self._payloads.pop(job_id, None)
            self._results.pop(job_id, None)
        return len(job_ids)

    def _fail_locked(self, job_id: str, *, error_code: str, now: datetime) -> JobSnapshot:
        if not _SAFE_ERROR_CODE.fullmatch(error_code):
            raise ValueError("error_code must be a safe lowercase identifier")
        row = self._required_row(job_id)
        expires_at = now + timedelta(seconds=int(row["retention_seconds"]))
        with self._connection:
            self._connection.execute(
                """
                UPDATE pilot_jobs
                SET state = ?, updated_at = ?, expires_at = ?, error_code = ?,
                    worker_digest = NULL, lease_expires_at = NULL
                WHERE job_id = ?
                """,
                (JobState.FAILED, now.isoformat(), expires_at.isoformat(), error_code, job_id),
            )
        self._payloads.pop(job_id, None)
        self._results.pop(job_id, None)
        return self._snapshot(self._required_row(job_id))

    def _require_worker_lease(self, job_id: str, worker_digest: str) -> sqlite3.Row:
        row = self._required_row(job_id)
        if row["state"] != JobState.RUNNING or row["worker_digest"] != worker_digest:
            raise InvalidJobTransitionError("job is not leased by this worker")
        return row

    def _owned_row(self, owner_id: str, job_id: str) -> sqlite3.Row:
        owner_digest = _digest(_validated_identity("owner_id", owner_id))
        row = cast(
            sqlite3.Row | None,
            self._connection.execute(
                "SELECT * FROM pilot_jobs WHERE job_id = ? AND owner_digest = ?",
                (job_id, owner_digest),
            ).fetchone(),
        )
        if row is None:
            raise JobNotFoundError("job was not found")
        return row

    def _required_row(self, job_id: str) -> sqlite3.Row:
        row = self._row(job_id)
        if row is None:
            raise JobNotFoundError("job was not found")
        return row

    def _row(self, job_id: str) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            self._connection.execute(
                "SELECT * FROM pilot_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone(),
        )

    @staticmethod
    def _snapshot(row: sqlite3.Row) -> JobSnapshot:
        return JobSnapshot(
            job_id=str(row["job_id"]),
            state=JobState(row["state"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            retention_class=RetentionClass(row["retention_class"]),
            error_code=row["error_code"],
        )
