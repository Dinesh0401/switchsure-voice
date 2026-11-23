"""SwitchSure HTTP application factory."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from switchsure_core import analyze_transcript

from .schemas import AnalyzeEvidenceRequest, HealthOut, TranscriptAssessmentOut

LOGGER = logging.getLogger("switchsure.api")
REQUEST_ID_HEADER = "X-Request-ID"


def create_app() -> FastAPI:
    app = FastAPI(
        title="SwitchSure Voice API",
        summary="Risk-aware multilingual transcript evidence API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
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
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "invalid_evidence", "message": str(error)},
            ) from error
        return TranscriptAssessmentOut.from_domain(result)

    return app


app = create_app()


def run() -> Any:
    """Local development entry point."""

    import uvicorn

    return uvicorn.run("switchsure_api.main:app", host="127.0.0.1", port=8000, reload=True)
