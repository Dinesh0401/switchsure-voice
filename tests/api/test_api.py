from __future__ import annotations

import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from switchsure_api import create_app
from switchsure_api.job_service import EvidenceJobService, analyze_evidence_request
from switchsure_api.schemas import AnalyzeEvidenceRequest, TranscriptAssessmentOut
from switchsure_core import PilotJobQueue

CAPABILITY_A = "owner-a-capability-token-000000000000"
CAPABILITY_B = "owner-b-capability-token-000000000000"


def evidence_payload() -> dict[str, Any]:
    return {
        "model_version": "fixture-v1",
        "words": [
            {
                "surface": "hello",
                "start_ms": 0,
                "end_ms": 100,
                "decoder_confidence": 0.9,
                "languages": [{"code": "en", "probability": 1.0}],
            }
        ],
    }


def make_job_service(
    tmp_path: Path,
    *,
    capacity: int = 32,
    processor: Callable[[AnalyzeEvidenceRequest], TranscriptAssessmentOut] = (
        analyze_evidence_request
    ),
    processing_timeout: timedelta = timedelta(seconds=30),
) -> EvidenceJobService:
    queue: PilotJobQueue[AnalyzeEvidenceRequest, TranscriptAssessmentOut] = PilotJobQueue(
        tmp_path / "jobs.sqlite3",
        capacity=capacity,
    )
    return EvidenceJobService(
        queue,
        processor=processor,
        processing_timeout=processing_timeout,
    )


def client() -> TestClient:
    return TestClient(create_app())


def test_health_has_request_id() -> None:
    response = client().get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "switchsure-api"}
    assert response.headers["X-Request-ID"]


def test_demo_analyzes_uncertain_code_switched_dosage() -> None:
    response = client().post(
        "/v1/demo/analyze-evidence",
        json={
            "model_version": "fixture-v1",
            "words": [
                {
                    "surface": "மருந்து",
                    "start_ms": 0,
                    "end_ms": 350,
                    "decoder_confidence": 0.92,
                    "boundary_probability": 0.25,
                    "languages": [
                        {"code": "ta", "probability": 0.94},
                        {"code": "en", "probability": 0.06},
                    ],
                },
                {
                    "surface": "500",
                    "start_ms": 360,
                    "end_ms": 580,
                    "decoder_confidence": 0.22,
                    "acoustic_quality": 0.6,
                    "boundary_probability": 0.52,
                    "languages": [
                        {"code": "neutral", "probability": 0.55},
                        {"code": "en", "probability": 0.45},
                    ],
                },
                {
                    "surface": "mg",
                    "start_ms": 590,
                    "end_ms": 760,
                    "decoder_confidence": 0.35,
                    "acoustic_quality": 0.65,
                    "boundary_probability": 0.49,
                    "languages": [
                        {"code": "en", "probability": 0.85},
                        {"code": "ta", "probability": 0.15},
                    ],
                },
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] in {"review", "defer"}
    assert body["calibration_status"] == "uncalibrated_baseline"
    assert body["entities"][0]["entity_type"] == "dosage"
    assert body["entities"][0]["surface"] == "500 mg"


def test_demo_rejects_non_normalized_language_probabilities() -> None:
    response = client().post(
        "/v1/demo/analyze-evidence",
        json={
            "words": [
                {
                    "surface": "hello",
                    "start_ms": 0,
                    "end_ms": 100,
                    "decoder_confidence": 0.9,
                    "languages": [{"code": "en", "probability": 0.8}],
                }
            ]
        },
    )

    assert response.status_code == 422


def test_job_submit_and_poll_returns_ephemeral_result(tmp_path: Path) -> None:
    service = make_job_service(tmp_path)
    try:
        with TestClient(create_app(job_service=service)) as api:
            submitted = api.post(
                "/v1/jobs/analyze-evidence",
                headers={"X-Job-Capability": CAPABILITY_A},
                json=evidence_payload(),
            )

            assert submitted.status_code == 202
            submitted_body = submitted.json()
            assert submitted_body["state"] == "queued"
            assert submitted_body["retention_class"] == "ephemeral"
            assert submitted_body["result"] is None

            completed = api.get(
                f"/v1/jobs/{submitted_body['job_id']}",
                headers={"X-Job-Capability": CAPABILITY_A},
            )

        assert completed.status_code == 200
        completed_body = completed.json()
        assert completed_body["state"] == "succeeded"
        assert completed_body["attempts"] == 1
        assert completed_body["result"]["text"] == "hello"
        assert completed_body["result"]["model_version"] == "fixture-v1"
    finally:
        service.shutdown()


def test_job_ownership_mismatch_is_indistinguishable_from_missing(tmp_path: Path) -> None:
    service = make_job_service(tmp_path)
    try:
        with TestClient(create_app(job_service=service, auto_process_jobs=False)) as api:
            submitted = api.post(
                "/v1/jobs/analyze-evidence",
                headers={"X-Job-Capability": CAPABILITY_A},
                json=evidence_payload(),
            )
            job_id = submitted.json()["job_id"]

            hidden = api.get(
                f"/v1/jobs/{job_id}",
                headers={"X-Job-Capability": CAPABILITY_B},
            )

            assert hidden.status_code == 404
            assert hidden.json()["code"] == "job_not_found"
            assert hidden.json()["message"] == "The job was not found."
            assert hidden.json()["request_id"] == hidden.headers["X-Request-ID"]

            owner_view = api.get(
                f"/v1/jobs/{job_id}",
                headers={"X-Job-Capability": CAPABILITY_A},
            )
            assert owner_view.status_code == 200
            assert owner_view.json()["state"] == "queued"
    finally:
        service.shutdown()


def test_queue_full_returns_typed_retryable_503(tmp_path: Path) -> None:
    service = make_job_service(tmp_path, capacity=1)
    try:
        with TestClient(create_app(job_service=service, auto_process_jobs=False)) as api:
            first = api.post(
                "/v1/jobs/analyze-evidence",
                headers={"X-Job-Capability": CAPABILITY_A},
                json=evidence_payload(),
            )
            assert first.status_code == 202

            rejected = api.post(
                "/v1/jobs/analyze-evidence",
                headers={"X-Job-Capability": CAPABILITY_B},
                json=evidence_payload(),
            )

            assert rejected.status_code == 503
            assert rejected.headers["Retry-After"] == "2"
            assert rejected.json() == {
                "code": "queue_full",
                "message": "The service is busy. Please retry shortly.",
                "request_id": rejected.headers["X-Request-ID"],
            }
    finally:
        service.shutdown()


def test_queued_job_can_be_cancelled_but_not_retried(tmp_path: Path) -> None:
    service = make_job_service(tmp_path)
    try:
        with TestClient(create_app(job_service=service, auto_process_jobs=False)) as api:
            submitted = api.post(
                "/v1/jobs/analyze-evidence",
                headers={"X-Job-Capability": CAPABILITY_A},
                json=evidence_payload(),
            )
            job_id = submitted.json()["job_id"]

            cancelled = api.post(
                f"/v1/jobs/{job_id}/cancel",
                headers={"X-Job-Capability": CAPABILITY_A},
            )
            assert cancelled.status_code == 200
            assert cancelled.json()["state"] == "cancelled"
            assert cancelled.json()["error_code"] == "cancelled_by_owner"

            retry = api.post(
                f"/v1/jobs/{job_id}/retry",
                headers={"X-Job-Capability": CAPABILITY_A},
                json=evidence_payload(),
            )
            assert retry.status_code == 409
            assert retry.json()["code"] == "invalid_job_state"
    finally:
        service.shutdown()


def test_failed_job_accepts_one_fresh_payload_retry(tmp_path: Path) -> None:
    attempts = 0

    def fail_once(payload: AnalyzeEvidenceRequest) -> TranscriptAssessmentOut:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("private validation detail")
        return analyze_evidence_request(payload)

    service = make_job_service(tmp_path, processor=fail_once)
    try:
        with TestClient(create_app(job_service=service)) as api:
            submitted = api.post(
                "/v1/jobs/analyze-evidence",
                headers={"X-Job-Capability": CAPABILITY_A},
                json=evidence_payload(),
            )
            job_id = submitted.json()["job_id"]
            failed = api.get(
                f"/v1/jobs/{job_id}",
                headers={"X-Job-Capability": CAPABILITY_A},
            )
            assert failed.json()["state"] == "failed"
            assert failed.json()["error_code"] == "invalid_evidence"
            assert "private" not in failed.text

            retried = api.post(
                f"/v1/jobs/{job_id}/retry",
                headers={"X-Job-Capability": CAPABILITY_A},
                json=evidence_payload(),
            )
            assert retried.status_code == 202
            assert retried.json()["state"] == "queued"

            completed = api.get(
                f"/v1/jobs/{job_id}",
                headers={"X-Job-Capability": CAPABILITY_A},
            )
            assert completed.json()["state"] == "succeeded"
            assert completed.json()["attempts"] == 2
            assert completed.json()["result"]["text"] == "hello"
    finally:
        service.shutdown()


def test_processing_timeout_fails_closed_with_safe_code(tmp_path: Path) -> None:
    def slow_processor(payload: AnalyzeEvidenceRequest) -> TranscriptAssessmentOut:
        time.sleep(0.05)
        return analyze_evidence_request(payload)

    service = make_job_service(
        tmp_path,
        processor=slow_processor,
        processing_timeout=timedelta(milliseconds=1),
    )
    try:
        with TestClient(create_app(job_service=service)) as api:
            submitted = api.post(
                "/v1/jobs/analyze-evidence",
                headers={"X-Job-Capability": CAPABILITY_A},
                json=evidence_payload(),
            )
            failed = api.get(
                f"/v1/jobs/{submitted.json()['job_id']}",
                headers={"X-Job-Capability": CAPABILITY_A},
            )

            assert failed.status_code == 200
            assert failed.json()["state"] == "failed"
            assert failed.json()["error_code"] == "model_timeout"
            assert failed.json()["result"] is None
    finally:
        service.shutdown()
