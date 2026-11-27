from pathlib import Path

import pytest
from switchsure_api.job_service import EvidenceJobService
from switchsure_api.schemas import AnalyzeEvidenceRequest, TranscriptAssessmentOut
from switchsure_api.telemetry import (
    BoundedMetricRegistry,
    ContentFreeEvent,
    DecisionAction,
    PipelineStage,
    SafeErrorClass,
    TelemetryEventKind,
    UnsafeTelemetryFieldError,
)
from switchsure_core import PilotJobQueue


def event(kind: TelemetryEventKind, **fields: object) -> ContentFreeEvent:
    return ContentFreeEvent.from_fields(kind, fields)


@pytest.mark.parametrize(
    "field",
    ["transcript", "raw_audio", "entity_value", "file_name", "access_token"],
)
def test_protected_content_fields_are_rejected(field: str) -> None:
    with pytest.raises(UnsafeTelemetryFieldError, match="protected telemetry fields"):
        event(
            TelemetryEventKind.STAGE_LATENCY,
            stage="asr",
            latency_ms=10.0,
            **{field: "a-real-secret"},
        )


def test_non_allowlisted_field_is_rejected_even_when_not_content_named() -> None:
    with pytest.raises(ValueError, match="not allowlisted"):
        event(TelemetryEventKind.ACTION, action="review", customer_id="arbitrary-label")


def test_arbitrary_string_cannot_become_a_metric_label() -> None:
    with pytest.raises(ValueError, match="unsupported PipelineStage"):
        event(
            TelemetryEventKind.STAGE_LATENCY,
            stage="secret spoken words",
            latency_ms=10,
        )


def test_direct_construction_cannot_bypass_label_validation() -> None:
    with pytest.raises(TypeError, match="action must be"):
        ContentFreeEvent(
            kind=TelemetryEventKind.ACTION,
            action="secret spoken words",  # type: ignore[arg-type]
        )


def test_registry_aggregates_latency_action_error_and_queue_metrics() -> None:
    registry = BoundedMetricRegistry()
    registry.record(event(TelemetryEventKind.STAGE_LATENCY, stage="asr", latency_ms=10.0))
    registry.record(event(TelemetryEventKind.STAGE_LATENCY, stage="asr", latency_ms=30.0))
    registry.record(event(TelemetryEventKind.ACTION, action="review"))
    registry.record(event(TelemetryEventKind.ERROR, stage="asr", error_class="model_timeout"))
    registry.record(event(TelemetryEventKind.QUEUE, queue_depth=2, queue_capacity=4))
    registry.record(event(TelemetryEventKind.QUEUE, queue_depth=4, queue_capacity=4))

    snapshot = registry.snapshot()

    assert snapshot.stage_latency[PipelineStage.ASR].count == 2
    assert snapshot.stage_latency[PipelineStage.ASR].mean_ms == 20.0
    assert snapshot.stage_latency[PipelineStage.ASR].maximum_ms == 30.0
    assert snapshot.actions[DecisionAction.REVIEW] == 1
    assert snapshot.errors[(PipelineStage.ASR, SafeErrorClass.MODEL_TIMEOUT)] == 1
    assert snapshot.queue.current_depth == 4
    assert snapshot.queue.high_watermark == 4
    assert snapshot.queue.full_observations == 1
    assert snapshot.event_count == 6


def test_registry_remains_fixed_cardinality_under_load() -> None:
    registry = BoundedMetricRegistry()

    for index in range(25_000):
        registry.record(
            event(
                TelemetryEventKind.STAGE_LATENCY,
                stage="end_to_end",
                latency_ms=float(index % 1_000),
            )
        )

    snapshot = registry.snapshot()
    assert snapshot.event_count == 25_000
    assert snapshot.stage_latency[PipelineStage.END_TO_END].count == 25_000
    assert snapshot.populated_series == 1
    assert snapshot.populated_series <= snapshot.series_capacity
    assert not hasattr(registry, "events")


@pytest.mark.parametrize(
    ("kind", "fields", "message"),
    [
        (TelemetryEventKind.STAGE_LATENCY, {"stage": "asr"}, "required"),
        (
            TelemetryEventKind.STAGE_LATENCY,
            {"stage": "asr", "latency_ms": float("inf")},
            "finite",
        ),
        (
            TelemetryEventKind.QUEUE,
            {"queue_depth": 5, "queue_capacity": 4},
            "cannot exceed",
        ),
    ],
)
def test_invalid_metric_values_fail_closed(
    kind: TelemetryEventKind,
    fields: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ContentFreeEvent.from_fields(kind, fields)


def test_job_service_records_only_bounded_content_free_aggregates(tmp_path: Path) -> None:
    queue: PilotJobQueue[AnalyzeEvidenceRequest, TranscriptAssessmentOut] = PilotJobQueue(
        tmp_path / "jobs.sqlite3"
    )
    service = EvidenceJobService(queue)
    payload = AnalyzeEvidenceRequest.model_validate(
        {
            "words": [
                {
                    "surface": "private words",
                    "start_ms": 0,
                    "end_ms": 100,
                    "decoder_confidence": 0.9,
                }
            ]
        }
    )
    try:
        service.submit(
            capability="capability-000000000000000000000000",
            payload=payload,
            idempotency_key=None,
        )
        assert service.process_pending() == 1

        snapshot = service.telemetry
        assert snapshot.event_count >= 4
        assert snapshot.actions
        assert snapshot.stage_latency[PipelineStage.END_TO_END].count == 1
        assert snapshot.queue.observations >= 2
        assert "private words" not in repr(snapshot)
        assert snapshot.populated_series <= snapshot.series_capacity
    finally:
        service.shutdown()
