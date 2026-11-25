"""SwitchSure HTTP application factory."""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from switchsure_core import (
    InvalidJobTransitionError,
    JobNotFoundError,
    JobResultUnavailableError,
    JobSnapshot,
    PilotJobQueue,
    QueueFullError,
    analyze_transcript,
)

from .job_service import EvidenceJobService
from .schemas import (
    AnalyzeEvidenceRequest,
    ApiErrorOut,
    HealthOut,
    JobStatusOut,
    TranscriptAssessmentOut,
)

LOGGER = logging.getLogger("switchsure.api")
REQUEST_ID_HEADER = "X-Request-ID"
CAPABILITY_HEADER = "X-Job-Capability"
IDEMPOTENCY_HEADER = "Idempotency-Key"
SAFE_MESSAGES = {
    "invalid_request": "Please check the submitted fields and try again.",
    "invalid_evidence": "The supplied evidence could not be analyzed.",
    "job_not_found": "The job was not found.",
    "invalid_job_state": "The job cannot perform that action in its current state.",
    "queue_unavailable": "The job service is not ready.",
    "queue_full": "The service is busy. Please retry shortly.",
    "result_unavailable": "The temporary result is no longer available. Please retry the job.",
    "internal_error": "The request could not be completed safely.",
}


def _default_job_service() -> EvidenceJobService:
    path = os.environ.get("SWITCHSURE_JOB_DB_PATH", ":memory:")
    queue: PilotJobQueue[AnalyzeEvidenceRequest, TranscriptAssessmentOut] = PilotJobQueue(path)
    return EvidenceJobService(queue)


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", uuid4()))


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = ApiErrorOut(
        code=code,
        message=SAFE_MESSAGES[code],
        request_id=_request_id(request),
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


def _job_status(
    snapshot: JobSnapshot,
    result: TranscriptAssessmentOut | None = None,
) -> JobStatusOut:
    return JobStatusOut(
        job_id=snapshot.job_id,
        state=snapshot.state.value,
        created_at=snapshot.created_at,
        updated_at=snapshot.updated_at,
        expires_at=snapshot.expires_at,
        attempts=snapshot.attempts,
        max_attempts=snapshot.max_attempts,
        retention_class=snapshot.retention_class.value,
        error_code=snapshot.error_code,
        result=result,
    )


def create_app(
    *,
    job_service: EvidenceJobService | None = None,
    auto_process_jobs: bool = True,
) -> FastAPI:
    service = job_service or _default_job_service()
    app = FastAPI(
        title="SwitchSure Voice API",
        summary="Risk-aware multilingual transcript evidence API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.job_service = service

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _error: RequestValidationError) -> JSONResponse:
        return _error_response(
            request,
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="invalid_request",
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        code = "internal_error"
        detail = cast(object, error.detail)
        if isinstance(detail, dict):
            candidate = detail.get("code")
            if isinstance(candidate, str) and candidate in SAFE_MESSAGES:
                code = candidate
        return _error_response(
            request,
            status_code=error.status_code,
            code=code,
            headers=error.headers,
        )

    @app.middleware("http")
    async def attach_request_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        LOGGER.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "method": request.method,
                "route": request.url.path,
                "status_code": response.status_code,
            },
        )
        return response

    @app.get("/health/live", response_model=HealthOut, tags=["health"])
    async def live() -> HealthOut:
        return HealthOut(status="ok", service="switchsure-api")

    @app.get("/health/ready", response_model=HealthOut, tags=["health"])
    async def ready() -> HealthOut:
        if not service.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "queue_unavailable", "message": "Job metadata is unavailable."},
            )
        return HealthOut(status="ready", service="switchsure-api")

    @app.post(
        "/v1/demo/analyze-evidence",
        response_model=TranscriptAssessmentOut,
        status_code=status.HTTP_200_OK,
        tags=["demo"],
        summary="Analyze model-neutral word evidence without running an ASR model",
    )
    async def analyze_evidence(payload: AnalyzeEvidenceRequest) -> TranscriptAssessmentOut:
        transcript_id = payload.transcript_id or str(uuid4())
        try:
            evidence, supplied_entities = payload.to_domain(transcript_id)
            result = analyze_transcript(evidence, supplied_entities=supplied_entities)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"code": "invalid_evidence", "message": str(error)},
            ) from error
        return TranscriptAssessmentOut.from_domain(result)

    @app.post(
        "/v1/jobs/analyze-evidence",
        response_model=JobStatusOut,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["jobs"],
    )
    async def submit_job(
        request: Request,
        payload: AnalyzeEvidenceRequest,
        background_tasks: BackgroundTasks,
        capability: str = Header(min_length=32, max_length=128, alias=CAPABILITY_HEADER),
        idempotency_key: str | None = Header(
            default=None,
            min_length=8,
            max_length=128,
            alias=IDEMPOTENCY_HEADER,
        ),
    ) -> JobStatusOut | JSONResponse:
        try:
            snapshot = service.submit(
                capability=capability,
                payload=payload,
                idempotency_key=idempotency_key,
            )
        except QueueFullError:
            return _error_response(
                request,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="queue_full",
                headers={"Retry-After": "2"},
            )
        if auto_process_jobs:
            background_tasks.add_task(service.process_pending)
        return _job_status(snapshot)

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=JobStatusOut,
        tags=["jobs"],
    )
    async def get_job(
        request: Request,
        job_id: str,
        capability: str = Header(min_length=32, max_length=128, alias=CAPABILITY_HEADER),
    ) -> JobStatusOut | JSONResponse:
        try:
            snapshot = service.snapshot(capability=capability, job_id=job_id)
            result = service.result(capability=capability, job_id=job_id)
        except JobNotFoundError:
            return _error_response(
                request,
                status_code=status.HTTP_404_NOT_FOUND,
                code="job_not_found",
            )
        except JobResultUnavailableError:
            return _error_response(
                request,
                status_code=status.HTTP_410_GONE,
                code="result_unavailable",
            )
        return _job_status(snapshot, result)

    @app.post(
        "/v1/jobs/{job_id}/cancel",
        response_model=JobStatusOut,
        tags=["jobs"],
    )
    async def cancel_job(
        request: Request,
        job_id: str,
        capability: str = Header(min_length=32, max_length=128, alias=CAPABILITY_HEADER),
    ) -> JobStatusOut | JSONResponse:
        try:
            return _job_status(service.cancel(capability=capability, job_id=job_id))
        except JobNotFoundError:
            return _error_response(
                request,
                status_code=status.HTTP_404_NOT_FOUND,
                code="job_not_found",
            )
        except InvalidJobTransitionError:
            return _error_response(
                request,
                status_code=status.HTTP_409_CONFLICT,
                code="invalid_job_state",
            )

    @app.post(
        "/v1/jobs/{job_id}/retry",
        response_model=JobStatusOut,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["jobs"],
    )
    async def retry_job(
        request: Request,
        job_id: str,
        payload: AnalyzeEvidenceRequest,
        background_tasks: BackgroundTasks,
        capability: str = Header(min_length=32, max_length=128, alias=CAPABILITY_HEADER),
    ) -> JobStatusOut | JSONResponse:
        try:
            snapshot = service.retry(
                capability=capability,
                job_id=job_id,
                payload=payload,
            )
        except JobNotFoundError:
            return _error_response(
                request,
                status_code=status.HTTP_404_NOT_FOUND,
                code="job_not_found",
            )
        except InvalidJobTransitionError:
            return _error_response(
                request,
                status_code=status.HTTP_409_CONFLICT,
                code="invalid_job_state",
            )
        except QueueFullError:
            return _error_response(
                request,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="queue_full",
                headers={"Retry-After": "2"},
            )
        if auto_process_jobs:
            background_tasks.add_task(service.process_pending)
        return _job_status(snapshot)

    return app


app = create_app()


def run() -> Any:
    """Local development entry point."""

    import uvicorn

    return uvicorn.run("switchsure_api.main:app", host="127.0.0.1", port=8000, reload=True)
