"""Leakage-safe boundary-aware uncertainty models for Phase 2 research.

The implementation is intentionally dependency-light: it provides deterministic
regularized logistic heads and temperature scaling without committing the domain
contract to a tensor framework. Decoder values remain evidence features; only a
model calibrated on the dedicated calibration split emits an error probability.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .dataset_manifest import DataSplit


class BoundaryFeatureMode(StrEnum):
    """Declared ablation used by the word-error head."""

    BASELINE = "baseline"
    BOUNDARY_AWARE = "boundary_aware"


@dataclass(frozen=True, slots=True)
class BoundaryObservation:
    """Privacy-safe token evidence and optional research labels.

    No transcript surface is stored. Continuous decoder values are explicitly
    named ``evidence`` because they are not probabilities of correctness.
    """

    example_id: str
    record_id: str
    speaker_id: str
    session_id: str
    split: DataSplit
    token_index: int
    language: str
    acoustic_band: str
    token_length: int
    is_entity: bool
    decoder_error_evidence: float
    lid_entropy: float
    language_transition_evidence: float
    timestamp_gap_ms: float
    hypothesis_disagreement: float
    target_is_boundary: bool | None = None
    target_is_error: bool | None = None

    def __post_init__(self) -> None:
        for name in (
            "example_id",
            "record_id",
            "speaker_id",
            "session_id",
            "language",
            "acoustic_band",
        ):
            value = getattr(self, name)
            if not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty and have no surrounding whitespace")
        if not isinstance(self.split, DataSplit):
            raise ValueError("split must be a DataSplit")
        if self.token_index < 0:
            raise ValueError("token_index cannot be negative")
        if self.token_length < 1:
            raise ValueError("token_length must be at least one")
        bounded = {
            "decoder_error_evidence": self.decoder_error_evidence,
            "lid_entropy": self.lid_entropy,
            "language_transition_evidence": self.language_transition_evidence,
            "hypothesis_disagreement": self.hypothesis_disagreement,
        }
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in bounded.values()):
            raise ValueError("uncertainty evidence values must be finite and between 0 and 1")
        if not math.isfinite(self.timestamp_gap_ms) or self.timestamp_gap_ms < 0.0:
            raise ValueError("timestamp_gap_ms must be finite and non-negative")

    @property
    def control_stratum(self) -> tuple[str, bool, str, int]:
        """Exact matching stratum: language, entity, acoustics, and length band."""

        if self.token_length <= 3:
            length_band = 0
        elif self.token_length <= 7:
            length_band = 1
        else:
            length_band = 2
        return (self.language, self.is_entity, self.acoustic_band, length_band)


@dataclass(frozen=True, slots=True)
class MatchedBoundaryPair:
    """One gold boundary and one exact-stratum non-boundary control."""

    boundary: BoundaryObservation
    control: BoundaryObservation

    def __post_init__(self) -> None:
        if self.boundary.target_is_boundary is not True:
            raise ValueError("boundary member must have a positive gold boundary label")
        if self.control.target_is_boundary is not False:
            raise ValueError("control member must have a negative gold boundary label")
        if self.boundary.split is not self.control.split:
            raise ValueError("matched observations cannot cross dataset splits")
        if self.boundary.control_stratum != self.control.control_stratum:
            raise ValueError("matched observations must share the declared control stratum")
        if self.boundary.example_id == self.control.example_id:
            raise ValueError("an observation cannot control itself")


@dataclass(frozen=True, slots=True)
class ControlMatchResult:
    """Deterministic matched-control output with explicit unmatched boundaries."""

    pairs: tuple[MatchedBoundaryPair, ...]
    unmatched_boundary_ids: tuple[str, ...]


def match_boundary_controls(
    observations: Sequence[BoundaryObservation],
) -> ControlMatchResult:
    """Greedily match gold boundaries to unused exact-stratum controls.

    Matching is deterministic and restricted to one declared split. Same-speaker
    controls are preferred to reduce speaker confounding, then the closest token
    length and stable example ID are used. This function is for analysis only and
    consumes gold boundary labels; it is never an inference feature.
    """

    if not observations:
        raise ValueError("observations cannot be empty")
    splits = {observation.split for observation in observations}
    if len(splits) != 1:
        raise ValueError("matched-control analysis must not cross dataset splits")
    _require_unique_examples(observations)
    if any(observation.target_is_boundary is None for observation in observations):
        raise ValueError("matched-control analysis requires gold boundary labels")

    boundaries = sorted(
        (observation for observation in observations if observation.target_is_boundary),
        key=lambda observation: observation.example_id,
    )
    available = {
        observation.example_id: observation
        for observation in observations
        if observation.target_is_boundary is False
    }
    pairs: list[MatchedBoundaryPair] = []
    unmatched: list[str] = []
    for boundary in boundaries:
        candidates = [
            control
            for control in available.values()
            if control.control_stratum == boundary.control_stratum
        ]
        if not candidates:
            unmatched.append(boundary.example_id)
            continue
        control = min(
            candidates,
            key=lambda candidate: (
                candidate.speaker_id != boundary.speaker_id,
                abs(candidate.token_length - boundary.token_length),
                candidate.example_id,
            ),
        )
        pairs.append(MatchedBoundaryPair(boundary=boundary, control=control))
        del available[control.example_id]
    return ControlMatchResult(tuple(pairs), tuple(unmatched))


@dataclass(frozen=True, slots=True)
class BoundaryUncertaintyOutput:
    """Per-token evidence suitable for metrics and privacy-safe artifacts."""

    example_id: str
    record_id: str
    speaker_id: str
    split: DataSplit
    feature_mode: BoundaryFeatureMode
    boundary_probability: float
    boundary_entropy: float
    raw_error_logit: float
    calibrated_error_probability: float | None
    target_is_boundary: bool | None
    target_is_error: bool | None


@dataclass(frozen=True, slots=True)
class _Standardizer:
    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(cls, rows: Sequence[Sequence[float]]) -> _Standardizer:
        if not rows or not rows[0]:
            raise ValueError("feature rows must be non-empty")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("feature rows must have a consistent width")
        means = tuple(sum(row[index] for row in rows) / len(rows) for index in range(width))
        variances = tuple(
            sum((row[index] - means[index]) ** 2 for row in rows) / len(rows)
            for index in range(width)
        )
        scales = tuple(max(math.sqrt(variance), 1e-8) for variance in variances)
        return cls(means, scales)

    def transform(self, row: Sequence[float]) -> tuple[float, ...]:
        if len(row) != len(self.means):
            raise ValueError("feature width does not match the fitted model")
        return tuple(
            (value - mean) / scale
            for value, mean, scale in zip(row, self.means, self.scales, strict=True)
        )


@dataclass(frozen=True, slots=True)
class _LogisticHead:
    standardizer: _Standardizer
    weights: tuple[float, ...]
    bias: float

    @classmethod
    def fit(
        cls,
        rows: Sequence[Sequence[float]],
        labels: Sequence[bool],
        *,
        learning_rate: float,
        iterations: int,
        l2: float,
    ) -> _LogisticHead:
        if len(rows) != len(labels) or not rows:
            raise ValueError("features and labels must have equal non-zero length")
        if len(set(labels)) != 2:
            raise ValueError("binary training requires both positive and negative labels")
        standardizer = _Standardizer.fit(rows)
        transformed = [standardizer.transform(row) for row in rows]
        width = len(transformed[0])
        weights = [0.0] * width
        bias = 0.0
        positives = sum(labels)
        positive_weight = len(labels) / (2.0 * positives)
        negative_weight = len(labels) / (2.0 * (len(labels) - positives))

        for step in range(iterations):
            gradient = [0.0] * width
            bias_gradient = 0.0
            for row, label in zip(transformed, labels, strict=True):
                prediction = _sigmoid(sum(w * x for w, x in zip(weights, row, strict=True)) + bias)
                sample_weight = positive_weight if label else negative_weight
                residual = (prediction - float(label)) * sample_weight
                for index, value in enumerate(row):
                    gradient[index] += residual * value
                bias_gradient += residual
            rate = learning_rate / math.sqrt(1.0 + step / 100.0)
            count = len(transformed)
            for index in range(width):
                gradient[index] = gradient[index] / count + l2 * weights[index]
                weights[index] -= rate * gradient[index]
            bias -= rate * bias_gradient / count
        return cls(standardizer, tuple(weights), bias)

    def logit(self, row: Sequence[float]) -> float:
        values = self.standardizer.transform(row)
        return sum(w * x for w, x in zip(self.weights, values, strict=True)) + self.bias


@dataclass(frozen=True, slots=True)
class _TemperatureCalibrator:
    temperature: float

    @classmethod
    def fit(cls, logits: Sequence[float], labels: Sequence[bool]) -> _TemperatureCalibrator:
        if len(logits) != len(labels) or not logits:
            raise ValueError("calibration logits and labels must have equal non-zero length")
        if len(set(labels)) != 2:
            raise ValueError("calibration requires both positive and negative labels")
        log_temperature = 0.0
        for _ in range(600):
            temperature = math.exp(log_temperature)
            gradient = sum(
                (_sigmoid(logit / temperature) - float(label)) * (-logit / temperature)
                for logit, label in zip(logits, labels, strict=True)
            ) / len(logits)
            log_temperature -= 0.03 * gradient
            log_temperature = min(max(log_temperature, math.log(0.05)), math.log(20.0))
        return cls(math.exp(log_temperature))

    def probability(self, logit: float) -> float:
        return _sigmoid(logit / self.temperature)


class BoundaryAwareUncertaintyModel:
    """Two-head boundary and word-error model with held-out calibration.

    The boundary head is trained first. Its probability and binary entropy are
    supplied to the error head only in ``BOUNDARY_AWARE`` mode. Temperature is
    then fit strictly on the calibration split. Validation and test examples are
    prediction-only and cannot mutate the model.
    """

    def __init__(
        self,
        *,
        feature_mode: BoundaryFeatureMode,
        boundary_head: _LogisticHead,
        error_head: _LogisticHead,
        training_speakers: frozenset[str],
        training_sessions: frozenset[str],
    ) -> None:
        self.feature_mode = feature_mode
        self._boundary_head = boundary_head
        self._error_head = error_head
        self._training_speakers = training_speakers
        self._training_sessions = training_sessions
        self._calibrator: _TemperatureCalibrator | None = None

    @classmethod
    def fit(
        cls,
        observations: Sequence[BoundaryObservation],
        *,
        feature_mode: BoundaryFeatureMode = BoundaryFeatureMode.BOUNDARY_AWARE,
        learning_rate: float = 0.08,
        iterations: int = 500,
        l2: float = 0.01,
    ) -> BoundaryAwareUncertaintyModel:
        """Fit both heads using only explicitly labelled training examples."""

        if not isinstance(feature_mode, BoundaryFeatureMode):
            raise ValueError("feature_mode must be a BoundaryFeatureMode")
        _validate_optimizer(learning_rate, iterations, l2)
        _validate_fit_examples(observations, DataSplit.TRAIN)
        boundary_rows = [_boundary_features(observation) for observation in observations]
        boundary_labels = [bool(observation.target_is_boundary) for observation in observations]
        boundary_head = _LogisticHead.fit(
            boundary_rows,
            boundary_labels,
            learning_rate=learning_rate,
            iterations=iterations,
            l2=l2,
        )
        error_rows = [
            _error_features(observation, feature_mode, boundary_head)
            for observation in observations
        ]
        error_labels = [bool(observation.target_is_error) for observation in observations]
        error_head = _LogisticHead.fit(
            error_rows,
            error_labels,
            learning_rate=learning_rate,
            iterations=iterations,
            l2=l2,
        )
        return cls(
            feature_mode=feature_mode,
            boundary_head=boundary_head,
            error_head=error_head,
            training_speakers=frozenset(observation.speaker_id for observation in observations),
            training_sessions=frozenset(observation.session_id for observation in observations),
        )

    @property
    def is_calibrated(self) -> bool:
        return self._calibrator is not None

    @property
    def calibration_temperature(self) -> float | None:
        return self._calibrator.temperature if self._calibrator is not None else None

    def calibrate(self, observations: Sequence[BoundaryObservation]) -> None:
        """Fit temperature once on speaker/session-disjoint calibration data."""

        if self._calibrator is not None:
            raise RuntimeError("model has already been calibrated")
        _validate_fit_examples(observations, DataSplit.CALIBRATION)
        speakers = {observation.speaker_id for observation in observations}
        sessions = {observation.session_id for observation in observations}
        if overlap := sorted(speakers & self._training_speakers):
            raise ValueError(f"calibration speakers overlap training speakers: {overlap}")
        if overlap := sorted(sessions & self._training_sessions):
            raise ValueError(f"calibration sessions overlap training sessions: {overlap}")
        logits = [self._raw_error_logit(observation) for observation in observations]
        labels = [bool(observation.target_is_error) for observation in observations]
        self._calibrator = _TemperatureCalibrator.fit(logits, labels)

    def predict(self, observation: BoundaryObservation) -> BoundaryUncertaintyOutput:
        """Emit boundary evidence, raw error score, and held-out-calibrated risk."""

        boundary_probability = _sigmoid(self._boundary_head.logit(_boundary_features(observation)))
        entropy = _binary_entropy(boundary_probability)
        raw_error_logit = self._raw_error_logit(observation)
        calibrated = (
            None if self._calibrator is None else self._calibrator.probability(raw_error_logit)
        )
        return BoundaryUncertaintyOutput(
            example_id=observation.example_id,
            record_id=observation.record_id,
            speaker_id=observation.speaker_id,
            split=observation.split,
            feature_mode=self.feature_mode,
            boundary_probability=boundary_probability,
            boundary_entropy=entropy,
            raw_error_logit=raw_error_logit,
            calibrated_error_probability=calibrated,
            target_is_boundary=observation.target_is_boundary,
            target_is_error=observation.target_is_error,
        )

    def predict_many(
        self, observations: Iterable[BoundaryObservation]
    ) -> tuple[BoundaryUncertaintyOutput, ...]:
        return tuple(self.predict(observation) for observation in observations)

    def _raw_error_logit(self, observation: BoundaryObservation) -> float:
        return self._error_head.logit(
            _error_features(observation, self.feature_mode, self._boundary_head)
        )


def _boundary_features(observation: BoundaryObservation) -> tuple[float, ...]:
    return (
        observation.lid_entropy,
        observation.language_transition_evidence,
        min(observation.timestamp_gap_ms / 500.0, 4.0),
        observation.hypothesis_disagreement,
    )


def _error_features(
    observation: BoundaryObservation,
    feature_mode: BoundaryFeatureMode,
    boundary_head: _LogisticHead,
) -> tuple[float, ...]:
    base = (
        observation.decoder_error_evidence,
        observation.lid_entropy,
        observation.hypothesis_disagreement,
        min(observation.token_length / 20.0, 2.0),
        float(observation.is_entity),
    )
    if feature_mode is BoundaryFeatureMode.BASELINE:
        return base
    boundary_probability = _sigmoid(boundary_head.logit(_boundary_features(observation)))
    return (*base, boundary_probability, _binary_entropy(boundary_probability))


def _validate_fit_examples(
    observations: Sequence[BoundaryObservation], expected_split: DataSplit
) -> None:
    if not observations:
        raise ValueError("observations cannot be empty")
    _require_unique_examples(observations)
    wrong = sorted(
        observation.example_id
        for observation in observations
        if observation.split is not expected_split
    )
    if wrong:
        raise ValueError(
            f"fit requires only the {expected_split.value} split; invalid examples: {wrong}"
        )
    if any(observation.target_is_boundary is None for observation in observations):
        raise ValueError("fitting requires gold boundary labels")
    if any(observation.target_is_error is None for observation in observations):
        raise ValueError("fitting requires gold word-error labels")


def _require_unique_examples(observations: Sequence[BoundaryObservation]) -> None:
    ids = [observation.example_id for observation in observations]
    duplicates = sorted(value for value in set(ids) if ids.count(value) > 1)
    if duplicates:
        raise ValueError(f"example IDs must be unique; duplicates: {duplicates}")


def _validate_optimizer(learning_rate: float, iterations: int, l2: float) -> None:
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if not math.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be finite and non-negative")


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(
        probability * math.log2(probability) + (1.0 - probability) * math.log2(1.0 - probability)
    )
