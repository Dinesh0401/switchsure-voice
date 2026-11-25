from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from switchsure_core import (
    InvalidJobTransitionError,
    JobNotFoundError,
    JobResultUnavailableError,
    JobState,
    PilotJobQueue,
    QueueFullError,
)

JOB_1 = "job_00000000000000000000000000000001"
JOB_2 = "job_00000000000000000000000000000002"


@dataclass
class MutableClock:
    current: datetime = datetime(2026, 11, 25, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta


def identifiers(*values: str) -> Callable[[], str]:
    iterator = iter(values)
    return lambda: next(iterator)


def test_queue_lifecycle_keeps_payload_and_result_out_of_metadata_repr(tmp_path: Path) -> None:
    queue: PilotJobQueue[dict[str, str], dict[str, str]] = PilotJobQueue(
        tmp_path / "jobs.sqlite3",
        id_factory=identifiers(JOB_1),
    )

    submitted = queue.submit(owner_id="owner-a", payload={"transcript": "private"})
    lease = queue.lease_next(worker_id="worker-a")

    assert submitted.state is JobState.QUEUED
    assert lease is not None
    assert lease.payload == {"transcript": "private"}
    assert "private" not in repr(lease)

    completed = queue.complete(
        job_id=lease.job_id,
        worker_id="worker-a",
        result={"entity": "secret"},
    )

    assert completed.state is JobState.SUCCEEDED
    assert queue.result(owner_id="owner-a", job_id=JOB_1) == {"entity": "secret"}
    assert "secret" not in repr(completed)


def test_capacity_idempotency_and_owner_isolation_are_explicit(tmp_path: Path) -> None:
    queue: PilotJobQueue[str, str] = PilotJobQueue(
        tmp_path / "jobs.sqlite3",
        capacity=1,
        id_factory=identifiers(JOB_1, JOB_2),
    )
    first = queue.submit(
        owner_id="owner-a",
        payload="private-audio-ref",
        idempotency_key="attempt-1",
    )

    duplicate = queue.submit(
        owner_id="owner-a",
        payload="different-content-is-not-stored",
        idempotency_key="attempt-1",
    )
    assert duplicate.job_id == first.job_id

    with pytest.raises(QueueFullError):
        queue.submit(owner_id="owner-b", payload="second")
    with pytest.raises(JobNotFoundError):
        queue.snapshot(owner_id="owner-b", job_id=first.job_id)


def test_failure_requires_fresh_payload_for_bounded_retry(tmp_path: Path) -> None:
    queue: PilotJobQueue[str, str] = PilotJobQueue(
        tmp_path / "jobs.sqlite3",
        id_factory=identifiers(JOB_1),
    )
    queue.submit(owner_id="owner", payload="first")
    lease = queue.lease_next(worker_id="worker")
    assert lease is not None
    failed = queue.fail(job_id=lease.job_id, worker_id="worker", error_code="model_timeout")

    assert failed.state is JobState.FAILED
    retried = queue.retry(owner_id="owner", job_id=JOB_1, payload="fresh")
    assert retried.state is JobState.QUEUED
    second_lease = queue.lease_next(worker_id="worker")
    assert second_lease is not None
    assert second_lease.payload == "fresh"
    queue.fail(job_id=JOB_1, worker_id="worker", error_code="model_timeout")

    with pytest.raises(InvalidJobTransitionError, match="retry limit"):
        queue.retry(owner_id="owner", job_id=JOB_1, payload="third")


def test_stalled_lease_fails_closed_and_terminal_metadata_expires(tmp_path: Path) -> None:
    clock = MutableClock()
    queue: PilotJobQueue[str, str] = PilotJobQueue(
        tmp_path / "jobs.sqlite3",
        lease_timeout=timedelta(seconds=5),
        clock=clock,
        id_factory=identifiers(JOB_1),
    )
    queue.submit(owner_id="owner", payload="private", retention_ttl=timedelta(seconds=10))
    assert queue.lease_next(worker_id="worker") is not None

    clock.advance(timedelta(seconds=6))
    assert queue.reap_stalled() == 1
    failed = queue.snapshot(owner_id="owner", job_id=JOB_1)
    assert failed.error_code == "worker_lease_timeout"

    clock.advance(timedelta(seconds=11))
    assert queue.purge_expired() == 1
    with pytest.raises(JobNotFoundError):
        queue.snapshot(owner_id="owner", job_id=JOB_1)


def test_restart_marks_process_local_content_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    first: PilotJobQueue[str, str] = PilotJobQueue(path, id_factory=identifiers(JOB_1))
    first.submit(owner_id="owner", payload="private")
    first.close()

    second: PilotJobQueue[str, str] = PilotJobQueue(path)
    recovered = second.snapshot(owner_id="owner", job_id=JOB_1)

    assert recovered.state is JobState.FAILED
    assert recovered.error_code == "process_restart_ephemeral_data_unavailable"
    with pytest.raises(InvalidJobTransitionError):
        second.result(owner_id="owner", job_id=JOB_1)


def test_completed_result_is_not_falsely_recovered_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    first: PilotJobQueue[str, str] = PilotJobQueue(path, id_factory=identifiers(JOB_1))
    first.submit(owner_id="owner", payload="private")
    lease = first.lease_next(worker_id="worker")
    assert lease is not None
    first.complete(job_id=JOB_1, worker_id="worker", result="sensitive result")
    first.close()

    second: PilotJobQueue[str, str] = PilotJobQueue(path)
    recovered = second.snapshot(owner_id="owner", job_id=JOB_1)

    assert recovered.state is JobState.FAILED
    assert recovered.error_code == "process_restart_ephemeral_data_unavailable"
    with pytest.raises((InvalidJobTransitionError, JobResultUnavailableError)):
        second.result(owner_id="owner", job_id=JOB_1)
