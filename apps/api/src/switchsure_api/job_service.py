"""Bounded local pilot worker for model-neutral evidence jobs."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import suppress
from datetime import timedelta
from threading import BoundedSemaphore
from uuid import uuid4

from switchsure_core import (
    InvalidJobTransitionError,
    JobResultUnavailableError,
    JobSnapshot,
    PilotJobQueue,
    analyze_transcript,
)

from .schemas import AnalyzeEvidenceRequest, TranscriptAssessmentOut

EvidenceProcessor = Callable[[AnalyzeEvidenceRequest], TranscriptAssessmentOut]


def analyze_evidence_request(payload: AnalyzeEvidenceRequest) -> TranscriptAssessmentOut:
    transcript_id = payload.transcript_id or str(uuid4())
    evidence, supplied_entities = payload.to_domain(transcript_id)
    result = analyze_transcript(evidence, supplied_entities=supplied_entities)
    return TranscriptAssessmentOut.from_domain(result)


class EvidenceJobService:
    """Runs a bounded number of content-bearing tasks over durable safe metadata."""

    def __init__(
        self,
        queue: PilotJobQueue[AnalyzeEvidenceRequest, TranscriptAssessmentOut],
        *,
        processor: EvidenceProcessor = analyze_evidence_request,
        worker_concurrency: int = 1,
        processing_timeout: timedelta = timedelta(seconds=30),
    ) -> None:
        if worker_concurrency < 1:
            raise ValueError("worker_concurrency must be positive")
        if processing_timeout <= timedelta(0):
            raise ValueError("processing_timeout must be positive")
        self._queue = queue
        self._processor = processor
        self._processing_timeout = processing_timeout
        self._worker_slots = BoundedSemaphore(worker_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=worker_concurrency,
            thread_name_prefix="switchsure-evidence",
        )

    @property
    def ready(self) -> bool:
        return self._queue.healthcheck()

    def submit(
        self,
        *,
        capability: str,
        payload: AnalyzeEvidenceRequest,
        idempotency_key: str | None,
    ) -> JobSnapshot:
        return self._queue.submit(
            owner_id=capability,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    def snapshot(self, *, capability: str, job_id: str) -> JobSnapshot:
        return self._queue.snapshot(owner_id=capability, job_id=job_id)

    def result(
        self,
        *,
        capability: str,
        job_id: str,
    ) -> TranscriptAssessmentOut | None:
        snapshot = self.snapshot(capability=capability, job_id=job_id)
        if snapshot.state.value != "succeeded":
            return None
        return self._queue.result(owner_id=capability, job_id=job_id)

    def cancel(self, *, capability: str, job_id: str) -> JobSnapshot:
        return self._queue.cancel(owner_id=capability, job_id=job_id)

    def retry(
        self,
        *,
        capability: str,
        job_id: str,
        payload: AnalyzeEvidenceRequest,
    ) -> JobSnapshot:
        return self._queue.retry(owner_id=capability, job_id=job_id, payload=payload)

    def process_pending(self) -> int:
        """Drain available work while holding one bounded worker slot."""

        if not self._worker_slots.acquire(blocking=False):
            return 0
        processed = 0
        worker_id = f"local-{uuid4().hex}"
        try:
            while lease := self._queue.lease_next(worker_id=worker_id):
                processed += 1
                future = self._executor.submit(self._processor, lease.payload)
                try:
                    result = future.result(timeout=self._processing_timeout.total_seconds())
                except TimeoutError:
                    future.cancel()
                    self._safe_fail(lease.job_id, worker_id, "model_timeout")
                except ValueError:
                    self._safe_fail(lease.job_id, worker_id, "invalid_evidence")
                except Exception:
                    self._safe_fail(lease.job_id, worker_id, "model_unavailable")
                else:
                    # Owner cancellation wins; content-bearing output is discarded.
                    with suppress(InvalidJobTransitionError):
                        self._queue.complete(
                            job_id=lease.job_id,
                            worker_id=worker_id,
                            result=result,
                        )
        finally:
            self._worker_slots.release()
        return processed

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._queue.close()

    def _safe_fail(self, job_id: str, worker_id: str, error_code: str) -> None:
        # Cancellation may have invalidated the lease while work was in flight.
        with suppress(InvalidJobTransitionError):
            self._queue.fail(job_id=job_id, worker_id=worker_id, error_code=error_code)


__all__ = [
    "EvidenceJobService",
    "JobResultUnavailableError",
    "analyze_evidence_request",
]
