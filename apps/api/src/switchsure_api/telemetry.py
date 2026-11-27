"""Content-free, fixed-cardinality telemetry for the controlled pilot.

The registry deliberately stores aggregates rather than an event history.  Its
dimensions are enums, so request content cannot become a metric label and a
high request volume cannot create an unbounded number of time series.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from threading import RLock
from types import MappingProxyType
from typing import ClassVar


class UnsafeTelemetryFieldError(ValueError):
    """Raised when a field name could contain protected request content."""


class TelemetryEventKind(StrEnum):
    STAGE_LATENCY = "stage_latency"
    ACTION = "action"
    ERROR = "error"
    QUEUE = "queue"


class PipelineStage(StrEnum):
    GATEWAY = "gateway"
    QUEUE = "queue"
    ASR = "asr"
    LANGUAGE_ID = "language_id"
    ENTITY_GUARD = "entity_guard"
    CALIBRATION = "calibration"
    POLICY = "policy"
    END_TO_END = "end_to_end"


class DecisionAction(StrEnum):
    ACCEPT = "accept"
    REVIEW = "review"
    DEFER = "defer"


class SafeErrorClass(StrEnum):
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    QUEUE_FULL = "queue_full"
    QUEUE_UNAVAILABLE = "queue_unavailable"
    MODEL_TIMEOUT = "model_timeout"
    MODEL_UNAVAILABLE = "model_unavailable"
    LOW_EVIDENCE = "low_evidence"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


_PROTECTED_FIELD_TERMS = frozenset(
    {
        "audio",
        "entity",
        "file",
        "filename",
        "token",
        "transcript",
    }
)
_MAX_LATENCY_MS = 86_400_000.0
_MAX_QUEUE_SIZE = 100_000


@dataclass(frozen=True, slots=True)
class ContentFreeEvent:
    """Validated event whose schema cannot represent request content."""

    kind: TelemetryEventKind
    stage: PipelineStage | None = None
    latency_ms: float | None = None
    action: DecisionAction | None = None
    error_class: SafeErrorClass | None = None
    queue_depth: int | None = None
    queue_capacity: int | None = None

    _FIELDS_BY_KIND: ClassVar[Mapping[TelemetryEventKind, frozenset[str]]] = MappingProxyType(
        {
            TelemetryEventKind.STAGE_LATENCY: frozenset({"stage", "latency_ms"}),
            TelemetryEventKind.ACTION: frozenset({"action"}),
            TelemetryEventKind.ERROR: frozenset({"stage", "error_class"}),
            TelemetryEventKind.QUEUE: frozenset({"queue_depth", "queue_capacity"}),
        }
    )

    def __post_init__(self) -> None:
        """Keep direct construction as strict as the mapping factory."""

        if not isinstance(self.kind, TelemetryEventKind):
            raise TypeError("kind must be a TelemetryEventKind")
        values = {
            "stage": self.stage,
            "latency_ms": self.latency_ms,
            "action": self.action,
            "error_class": self.error_class,
            "queue_depth": self.queue_depth,
            "queue_capacity": self.queue_capacity,
        }
        present = {name for name, value in values.items() if value is not None}
        if present != self._FIELDS_BY_KIND[self.kind]:
            raise ValueError("event fields do not match the metric kind")

        if self.stage is not None and not isinstance(self.stage, PipelineStage):
            raise TypeError("stage must be a PipelineStage")
        if self.action is not None and not isinstance(self.action, DecisionAction):
            raise TypeError("action must be a DecisionAction")
        if self.error_class is not None and not isinstance(self.error_class, SafeErrorClass):
            raise TypeError("error_class must be a SafeErrorClass")
        if self.latency_ms is not None:
            _bounded_float("latency_ms", self.latency_ms)
        if self.queue_depth is not None:
            _bounded_int("queue_depth", self.queue_depth, minimum=0)
        if self.queue_capacity is not None:
            _bounded_int("queue_capacity", self.queue_capacity, minimum=1)
        if (
            self.queue_depth is not None
            and self.queue_capacity is not None
            and self.queue_depth > self.queue_capacity
        ):
            raise ValueError("queue_depth cannot exceed queue_capacity")

    @classmethod
    def from_fields(
        cls,
        kind: TelemetryEventKind,
        fields: Mapping[str, object],
    ) -> ContentFreeEvent:
        """Build an event through a strict per-kind field allowlist."""

        if any(not isinstance(key, str) for key in fields):
            raise TypeError("telemetry field names must be strings")
        protected = sorted(
            key for key in fields if any(term in key.casefold() for term in _PROTECTED_FIELD_TERMS)
        )
        if protected:
            raise UnsafeTelemetryFieldError(
                f"protected telemetry fields are forbidden: {', '.join(protected)}"
            )
        allowed = cls._FIELDS_BY_KIND[kind]
        unexpected = sorted(set(fields) - allowed)
        if unexpected:
            raise ValueError(f"telemetry fields are not allowlisted: {', '.join(unexpected)}")
        missing = sorted(allowed - set(fields))
        if missing:
            raise ValueError(f"required telemetry fields are missing: {', '.join(missing)}")

        if kind is TelemetryEventKind.STAGE_LATENCY:
            return cls(
                kind=kind,
                stage=_enum_value(PipelineStage, fields["stage"]),
                latency_ms=_bounded_float("latency_ms", fields["latency_ms"]),
            )
        if kind is TelemetryEventKind.ACTION:
            return cls(kind=kind, action=_enum_value(DecisionAction, fields["action"]))
        if kind is TelemetryEventKind.ERROR:
            return cls(
                kind=kind,
                stage=_enum_value(PipelineStage, fields["stage"]),
                error_class=_enum_value(SafeErrorClass, fields["error_class"]),
            )

        queue_depth = _bounded_int("queue_depth", fields["queue_depth"], minimum=0)
        queue_capacity = _bounded_int("queue_capacity", fields["queue_capacity"], minimum=1)
        if queue_depth > queue_capacity:
            raise ValueError("queue_depth cannot exceed queue_capacity")
        return cls(
            kind=kind,
            queue_depth=queue_depth,
            queue_capacity=queue_capacity,
        )


@dataclass(frozen=True, slots=True)
class LatencyMetric:
    count: int
    total_ms: float
    maximum_ms: float

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


@dataclass(frozen=True, slots=True)
class QueueMetric:
    observations: int
    current_depth: int
    capacity: int
    high_watermark: int
    full_observations: int


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    stage_latency: Mapping[PipelineStage, LatencyMetric]
    actions: Mapping[DecisionAction, int]
    errors: Mapping[tuple[PipelineStage, SafeErrorClass], int]
    queue: QueueMetric
    event_count: int
    populated_series: int
    series_capacity: int


@dataclass(slots=True)
class _LatencyAccumulator:
    count: int = 0
    total_ms: float = 0.0
    maximum_ms: float = 0.0

    def observe(self, latency_ms: float) -> None:
        self.count += 1
        self.total_ms += latency_ms
        self.maximum_ms = max(self.maximum_ms, latency_ms)


class BoundedMetricRegistry:
    """Thread-safe registry with a fixed upper bound on metric series."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._latencies = {stage: _LatencyAccumulator() for stage in PipelineStage}
        self._actions = {action: 0 for action in DecisionAction}
        self._errors = {
            (stage, error_class): 0 for stage in PipelineStage for error_class in SafeErrorClass
        }
        self._queue_observations = 0
        self._queue_depth = 0
        self._queue_capacity = 0
        self._queue_high_watermark = 0
        self._queue_full_observations = 0
        self._event_count = 0

    @property
    def series_capacity(self) -> int:
        """Maximum series count; independent of request volume and values."""

        return len(self._latencies) + len(self._actions) + len(self._errors) + 1

    def record(self, event: ContentFreeEvent) -> None:
        with self._lock:
            if event.kind is TelemetryEventKind.STAGE_LATENCY:
                assert event.stage is not None and event.latency_ms is not None
                self._latencies[event.stage].observe(event.latency_ms)
            elif event.kind is TelemetryEventKind.ACTION:
                assert event.action is not None
                self._actions[event.action] += 1
            elif event.kind is TelemetryEventKind.ERROR:
                assert event.stage is not None and event.error_class is not None
                self._errors[(event.stage, event.error_class)] += 1
            else:
                assert event.queue_depth is not None and event.queue_capacity is not None
                self._queue_observations += 1
                self._queue_depth = event.queue_depth
                self._queue_capacity = event.queue_capacity
                self._queue_high_watermark = max(self._queue_high_watermark, event.queue_depth)
                if event.queue_depth == event.queue_capacity:
                    self._queue_full_observations += 1
            self._event_count += 1

    def snapshot(self) -> TelemetrySnapshot:
        """Copy aggregate metrics without exposing or retaining individual events."""

        with self._lock:
            latency = {
                stage: LatencyMetric(item.count, item.total_ms, item.maximum_ms)
                for stage, item in self._latencies.items()
                if item.count
            }
            actions = {action: count for action, count in self._actions.items() if count}
            errors = {key: count for key, count in self._errors.items() if count}
            queue = QueueMetric(
                observations=self._queue_observations,
                current_depth=self._queue_depth,
                capacity=self._queue_capacity,
                high_watermark=self._queue_high_watermark,
                full_observations=self._queue_full_observations,
            )
            populated_series = len(latency) + len(actions) + len(errors)
            if queue.observations:
                populated_series += 1
            return TelemetrySnapshot(
                stage_latency=MappingProxyType(latency),
                actions=MappingProxyType(actions),
                errors=MappingProxyType(errors),
                queue=queue,
                event_count=self._event_count,
                populated_series=populated_series,
                series_capacity=self.series_capacity,
            )


def _enum_value[EnumT: StrEnum](enum_type: type[EnumT], value: object) -> EnumT:
    if not isinstance(value, str):
        raise TypeError(f"{enum_type.__name__} value must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"unsupported {enum_type.__name__} value") from error


def _bounded_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not isfinite(numeric) or not 0.0 <= numeric <= _MAX_LATENCY_MS:
        raise ValueError(f"{name} must be finite and between 0 and {_MAX_LATENCY_MS:g}")
    return numeric


def _bounded_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= _MAX_QUEUE_SIZE:
        raise ValueError(f"{name} must be between {minimum} and {_MAX_QUEUE_SIZE}")
    return value


__all__ = [
    "BoundedMetricRegistry",
    "ContentFreeEvent",
    "DecisionAction",
    "LatencyMetric",
    "PipelineStage",
    "QueueMetric",
    "SafeErrorClass",
    "TelemetryEventKind",
    "TelemetrySnapshot",
    "UnsafeTelemetryFieldError",
]
