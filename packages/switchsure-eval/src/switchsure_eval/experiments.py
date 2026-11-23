"""Reproducible experiment contracts and mechanical Phase-1 release decisions.

The types in this module deliberately contain no model or data-frame dependency.  They
make evaluation provenance, non-success outcomes, privacy suppression, and statistical
comparison explicit before a report is rendered.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from random import Random


class PredictionOutcome(StrEnum):
    """Every terminal prediction state, including states without a transcript."""

    SUCCESS = "success"
    MISSING = "missing"
    TIMEOUT = "timeout"
    INVALID_AUDIO = "invalid_audio"
    DECODER_FAILURE = "decoder_failure"
    ABSTENTION = "abstention"


class ErrorCategory(StrEnum):
    """Predeclared mutually countable error categories for baseline analysis."""

    ORDINARY_SUBSTITUTION = "ordinary_substitution"
    ORDINARY_DELETION = "ordinary_deletion"
    ORDINARY_INSERTION = "ordinary_insertion"
    SWITCH_REGION_ERROR = "switch_region_error"
    LANGUAGE_ID_ERROR = "language_id_error"
    SWITCH_BOUNDARY_ERROR = "switch_boundary_error"
    ENTITY_DETECTION_ERROR = "entity_detection_error"
    ENTITY_SPAN_ERROR = "entity_span_error"
    ENTITY_VALUE_ERROR = "entity_value_error"
    HIGH_SEVERITY_ACCEPTED_ERROR = "high_severity_accepted_error"
    CONTEXT_BIAS_FALSE_POSITIVE = "context_bias_false_positive"


@dataclass(frozen=True, slots=True)
class ErrorCount:
    category: ErrorCategory
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("error counts cannot be negative")


@dataclass(frozen=True, slots=True)
class ErrorTaxonomyCounts:
    """A complete, duplicate-free set of declared error counts."""

    counts: tuple[ErrorCount, ...]

    def __post_init__(self) -> None:
        categories = [item.category for item in self.counts]
        if len(categories) != len(set(categories)):
            raise ValueError("each error category may appear only once")

    def count(self, category: ErrorCategory) -> int:
        return next((item.count for item in self.counts if item.category == category), 0)

    @property
    def total(self) -> int:
        return sum(item.count for item in self.counts)


@dataclass(frozen=True, slots=True)
class OutcomeCounts:
    success: int = 0
    missing: int = 0
    timeout: int = 0
    invalid_audio: int = 0
    decoder_failure: int = 0
    abstention: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.as_mapping().values()):
            raise ValueError("outcome counts cannot be negative")

    @classmethod
    def from_outcomes(cls, outcomes: Iterable[PredictionOutcome]) -> OutcomeCounts:
        counts = Counter(outcomes)
        return cls(**{outcome.value: counts[outcome] for outcome in PredictionOutcome})

    def as_mapping(self) -> Mapping[str, int]:
        return {
            "success": self.success,
            "missing": self.missing,
            "timeout": self.timeout,
            "invalid_audio": self.invalid_audio,
            "decoder_failure": self.decoder_failure,
            "abstention": self.abstention,
        }

    @property
    def total(self) -> int:
        return sum(self.as_mapping().values())

    @property
    def non_success(self) -> int:
        return self.total - self.success


@dataclass(frozen=True, slots=True)
class DatasetPartition:
    """A partition manifest using stable, non-identifying speaker/session keys."""

    name: str
    digest: str
    speaker_keys: tuple[str, ...]
    session_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.digest.strip():
            raise ValueError("partition name and digest are required")
        if not self.speaker_keys:
            raise ValueError("each partition must declare speaker keys")
        if len(self.speaker_keys) != len(set(self.speaker_keys)):
            raise ValueError(f"partition {self.name!r} contains duplicate speaker keys")
        if len(self.session_keys) != len(set(self.session_keys)):
            raise ValueError(f"partition {self.name!r} contains duplicate session keys")


@dataclass(frozen=True, slots=True)
class SplitManifest:
    train: DatasetPartition
    calibration: DatasetPartition
    validation: DatasetPartition
    test: DatasetPartition

    def __post_init__(self) -> None:
        partitions = self.partitions
        if tuple(partition.name for partition in partitions) != (
            "train",
            "calibration",
            "validation",
            "test",
        ):
            raise ValueError("partition names must be train, calibration, validation, and test")
        self.assert_disjoint()

    @property
    def partitions(self) -> tuple[DatasetPartition, ...]:
        return (self.train, self.calibration, self.validation, self.test)

    def assert_disjoint(self) -> None:
        for index, left in enumerate(self.partitions):
            for right in self.partitions[index + 1 :]:
                shared_speakers = set(left.speaker_keys) & set(right.speaker_keys)
                if shared_speakers:
                    raise ValueError(
                        f"speaker leakage between {left.name} and {right.name}: "
                        f"{len(shared_speakers)} shared key(s)"
                    )
                shared_sessions = set(left.session_keys) & set(right.session_keys)
                if shared_sessions:
                    raise ValueError(
                        f"session leakage between {left.name} and {right.name}: "
                        f"{len(shared_sessions)} shared key(s)"
                    )


@dataclass(frozen=True, slots=True)
class ArtifactRevision:
    name: str
    revision: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.revision.strip():
            raise ValueError("artifact name and pinned revision are required")


@dataclass(frozen=True, slots=True)
class BaselineExperimentSpec:
    """Immutable preregistration data required to reproduce a baseline run."""

    experiment_id: str
    git_commit: str
    environment_lock_hash: str
    dataset_manifest_hash: str
    normalization_version: str
    splits: SplitManifest
    model: ArtifactRevision
    calibrator: ArtifactRevision
    policy: ArtifactRevision
    primary_comparison: str
    primary_metric: str
    seeds: tuple[int, ...]
    reference_hardware: str
    tuning_partitions: tuple[str, ...] = ("train", "calibration", "validation")
    subgroup_min_speakers: int = 20

    def __post_init__(self) -> None:
        required = (
            self.experiment_id,
            self.git_commit,
            self.environment_lock_hash,
            self.dataset_manifest_hash,
            self.normalization_version,
            self.primary_comparison,
            self.primary_metric,
            self.reference_hardware,
        )
        if any(not value.strip() for value in required):
            raise ValueError("all experiment provenance fields are required")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("at least one unique seed is required")
        if self.subgroup_min_speakers < 2:
            raise ValueError("subgroup suppression must require at least two speakers")
        allowed = {partition.name for partition in self.splits.partitions}
        if any(partition not in allowed for partition in self.tuning_partitions):
            raise ValueError("tuning partitions must refer to declared splits")
        if "test" in self.tuning_partitions:
            raise ValueError("the locked test partition cannot be used for tuning")


@dataclass(frozen=True, slots=True)
class SubgroupObservation:
    speaker_key: str
    slice_name: str
    slice_value: str
    numerator: float
    denominator: float
    outcome: PredictionOutcome

    def __post_init__(self) -> None:
        if not self.speaker_key or not self.slice_name or not self.slice_value:
            raise ValueError("speaker and subgroup keys are required")
        if not isfinite(self.numerator) or not isfinite(self.denominator):
            raise ValueError("metric components must be finite")
        if self.numerator < 0.0 or self.denominator < 0.0:
            raise ValueError("metric components cannot be negative")
        if self.numerator > self.denominator:
            raise ValueError("numerator cannot exceed denominator")


@dataclass(frozen=True, slots=True)
class SubgroupSummary:
    slice_name: str
    slice_value: str
    example_count: int
    speaker_count: int
    numerator: float
    denominator: float
    metric: float | None
    outcome_counts: OutcomeCounts
    suppressed: bool
    suppression_reason: str | None


def summarize_subgroups(
    observations: Iterable[SubgroupObservation], *, min_speakers: int
) -> tuple[SubgroupSummary, ...]:
    """Aggregate declared slices and suppress values with too few unique speakers."""

    if min_speakers < 2:
        raise ValueError("min_speakers must be at least two")
    grouped: dict[tuple[str, str], list[SubgroupObservation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.slice_name, observation.slice_value)].append(observation)

    summaries: list[SubgroupSummary] = []
    for (slice_name, slice_value), items in sorted(grouped.items()):
        speaker_count = len({item.speaker_key for item in items})
        numerator = sum(item.numerator for item in items)
        denominator = sum(item.denominator for item in items)
        too_small = speaker_count < min_speakers
        empty_denominator = denominator == 0.0
        if too_small:
            reason = f"fewer than {min_speakers} unique speakers"
        elif empty_denominator:
            reason = "metric denominator is zero"
        else:
            reason = None
        suppressed = reason is not None
        summaries.append(
            SubgroupSummary(
                slice_name=slice_name,
                slice_value=slice_value,
                example_count=len(items),
                speaker_count=speaker_count,
                numerator=numerator,
                denominator=denominator,
                metric=None if suppressed else numerator / denominator,
                outcome_counts=OutcomeCounts.from_outcomes(item.outcome for item in items),
                suppressed=suppressed,
                suppression_reason=reason,
            )
        )
    return tuple(summaries)


@dataclass(frozen=True, slots=True)
class SpeakerMetricPair:
    speaker_key: str
    baseline_numerator: float
    baseline_denominator: float
    candidate_numerator: float
    candidate_denominator: float

    def __post_init__(self) -> None:
        values = (
            self.baseline_numerator,
            self.baseline_denominator,
            self.candidate_numerator,
            self.candidate_denominator,
        )
        if not self.speaker_key or any(not isfinite(value) for value in values):
            raise ValueError("speaker key and finite metric components are required")
        if any(value < 0.0 for value in values):
            raise ValueError("metric components cannot be negative")
        if self.baseline_denominator == 0.0 or self.candidate_denominator == 0.0:
            raise ValueError("paired metric denominators must be positive")


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int
    seed: int
    speaker_count: int


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def paired_speaker_bootstrap(
    pairs: Sequence[SpeakerMetricPair],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 17,
) -> BootstrapInterval:
    """Estimate candidate-minus-baseline difference by paired speaker resampling."""

    if len(pairs) < 2:
        raise ValueError("at least two paired speakers are required")
    if len({pair.speaker_key for pair in pairs}) != len(pairs):
        raise ValueError("speaker keys must be unique; aggregate utterances before bootstrapping")
    if resamples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    def difference(sample: Sequence[SpeakerMetricPair]) -> float:
        baseline = sum(pair.baseline_numerator for pair in sample) / sum(
            pair.baseline_denominator for pair in sample
        )
        candidate = sum(pair.candidate_numerator for pair in sample) / sum(
            pair.candidate_denominator for pair in sample
        )
        return candidate - baseline

    random = Random(seed)
    sample_size = len(pairs)
    differences = [
        difference(tuple(pairs[random.randrange(sample_size)] for _ in range(sample_size)))
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence_level) / 2.0
    return BootstrapInterval(
        estimate=difference(pairs),
        lower=_quantile(differences, tail),
        upper=_quantile(differences, 1.0 - tail),
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
        speaker_count=sample_size,
    )


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    value: float
    lower: float | None = None
    upper: float | None = None
    denominator: int | None = None

    def __post_init__(self) -> None:
        values = tuple(value for value in (self.value, self.lower, self.upper) if value is not None)
        if any(not isfinite(value) for value in values):
            raise ValueError("metric estimates must be finite")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("confidence interval lower bound cannot exceed upper bound")
        if self.denominator is not None and self.denominator < 0:
            raise ValueError("metric denominator cannot be negative")


class GateOperator(StrEnum):
    AT_MOST = "at_most"
    AT_LEAST = "at_least"
    CI_UPPER_BELOW = "ci_upper_below"
    CI_LOWER_ABOVE = "ci_lower_above"


@dataclass(frozen=True, slots=True)
class GateClause:
    metric: str
    operator: GateOperator
    threshold: float


@dataclass(frozen=True, slots=True)
class ReleaseGate:
    gate_id: str
    description: str
    any_of: tuple[GateClause, ...]

    def __post_init__(self) -> None:
        if not self.gate_id or not self.description or not self.any_of:
            raise ValueError("release gate id, description, and clauses are required")


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUABLE = "not_evaluable"


@dataclass(frozen=True, slots=True)
class ClauseResult:
    clause: GateClause
    status: GateStatus
    observed: float | None


@dataclass(frozen=True, slots=True)
class ReleaseGateResult:
    gate_id: str
    status: GateStatus
    clause_results: tuple[ClauseResult, ...]


@dataclass(frozen=True, slots=True)
class ReleaseDecision:
    gate_results: tuple[ReleaseGateResult, ...]

    @property
    def ready(self) -> bool:
        return bool(self.gate_results) and all(
            result.status == GateStatus.PASS for result in self.gate_results
        )


def _evaluate_clause(clause: GateClause, estimate: MetricEstimate | None) -> ClauseResult:
    if estimate is None:
        return ClauseResult(clause, GateStatus.NOT_EVALUABLE, None)
    if clause.operator == GateOperator.AT_MOST:
        observed = estimate.value
        passed = observed <= clause.threshold
    elif clause.operator == GateOperator.AT_LEAST:
        observed = estimate.value
        passed = observed >= clause.threshold
    elif clause.operator == GateOperator.CI_UPPER_BELOW:
        if estimate.upper is None:
            return ClauseResult(clause, GateStatus.NOT_EVALUABLE, None)
        observed = estimate.upper
        passed = observed < clause.threshold
    else:
        if estimate.lower is None:
            return ClauseResult(clause, GateStatus.NOT_EVALUABLE, None)
        observed = estimate.lower
        passed = observed > clause.threshold
    return ClauseResult(clause, GateStatus.PASS if passed else GateStatus.FAIL, observed)


def evaluate_release_gates(
    gates: Sequence[ReleaseGate], metrics: Mapping[str, MetricEstimate]
) -> ReleaseDecision:
    """Apply gates mechanically; absent evidence can never produce release readiness."""

    if not gates:
        raise ValueError("at least one release gate is required")
    if len({gate.gate_id for gate in gates}) != len(gates):
        raise ValueError("release gate ids must be unique")
    results: list[ReleaseGateResult] = []
    for gate in gates:
        clause_results = tuple(
            _evaluate_clause(clause, metrics.get(clause.metric)) for clause in gate.any_of
        )
        statuses = {result.status for result in clause_results}
        if GateStatus.PASS in statuses:
            status = GateStatus.PASS
        elif GateStatus.NOT_EVALUABLE in statuses:
            status = GateStatus.NOT_EVALUABLE
        else:
            status = GateStatus.FAIL
        results.append(ReleaseGateResult(gate.gate_id, status, clause_results))
    return ReleaseDecision(tuple(results))


PHASE1_RELEASE_GATES: tuple[ReleaseGate, ...] = (
    ReleaseGate(
        "general-wer-regression",
        "General-speech WER regression is no more than one absolute point.",
        (GateClause("general_wer_absolute_difference", GateOperator.AT_MOST, 0.01),),
    ),
    ReleaseGate(
        "critical-entity-errors",
        "Critical-entity error rate falls by at least 20% relative.",
        (GateClause("critical_entity_error_relative_reduction", GateOperator.AT_LEAST, 0.20),),
    ),
    ReleaseGate(
        "switch-or-severity",
        "Switch-region WER improves by 15% or high-severity accepted errors improve.",
        (
            GateClause("switch_region_wer_relative_reduction", GateOperator.AT_LEAST, 0.15),
            GateClause(
                "high_severity_accepted_error_difference",
                GateOperator.CI_UPPER_BELOW,
                0.0,
            ),
        ),
    ),
    ReleaseGate(
        "overall-ece",
        "Overall word-error ECE is at most 0.05.",
        (GateClause("word_error_ece", GateOperator.AT_MOST, 0.05),),
    ),
    ReleaseGate(
        "major-slice-ece",
        "Worst declared major-slice word-error ECE is at most 0.08.",
        (GateClause("major_slice_max_word_error_ece", GateOperator.AT_MOST, 0.08),),
    ),
    ReleaseGate(
        "aurc-improvement",
        "AURC is below raw decoder confidence with a CI excluding zero.",
        (GateClause("aurc_difference", GateOperator.CI_UPPER_BELOW, 0.0),),
    ),
    ReleaseGate(
        "subgroup-severity-ratio",
        "Worst major subgroup severity rate is no more than 1.5 times overall.",
        (GateClause("major_subgroup_max_severity_ratio", GateOperator.AT_MOST, 1.5),),
    ),
    ReleaseGate(
        "latency-target",
        "Declared p95 short-utterance latency target is met.",
        (GateClause("p95_latency_target_met", GateOperator.AT_LEAST, 1.0),),
    ),
    ReleaseGate(
        "critical-accessibility-violations",
        "Automated and manual launch checks have zero critical accessibility violations.",
        (GateClause("critical_accessibility_violations", GateOperator.AT_MOST, 0.0),),
    ),
)
