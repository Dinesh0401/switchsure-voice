from dataclasses import FrozenInstanceError

import pytest
from switchsure_eval.experiments import (
    PHASE1_RELEASE_GATES,
    ArtifactRevision,
    BaselineExperimentSpec,
    DatasetPartition,
    ErrorCategory,
    ErrorCount,
    ErrorTaxonomyCounts,
    GateClause,
    GateOperator,
    GateStatus,
    MetricEstimate,
    OutcomeCounts,
    PredictionOutcome,
    ReleaseGate,
    SpeakerMetricPair,
    SplitManifest,
    SubgroupObservation,
    evaluate_release_gates,
    paired_speaker_bootstrap,
    summarize_subgroups,
)


def partition(name: str, speaker: str, session: str = "") -> DatasetPartition:
    return DatasetPartition(
        name=name,
        digest=f"sha256:{name}",
        speaker_keys=(speaker,),
        session_keys=(session,) if session else (),
    )


def split_manifest() -> SplitManifest:
    return SplitManifest(
        train=partition("train", "speaker-a", "session-a"),
        calibration=partition("calibration", "speaker-b", "session-b"),
        validation=partition("validation", "speaker-c", "session-c"),
        test=partition("test", "speaker-d", "session-d"),
    )


def test_experiment_spec_is_immutable_and_preregistered() -> None:
    spec = BaselineExperimentSpec(
        experiment_id="phase1-whisper-zero-shot",
        git_commit="abc123",
        environment_lock_hash="sha256:lock",
        dataset_manifest_hash="sha256:data",
        normalization_version="ta-en-v1",
        splits=split_manifest(),
        model=ArtifactRevision("whisper", "revision-1"),
        calibrator=ArtifactRevision("none", "not-applicable-v1"),
        policy=ArtifactRevision("unguarded", "v1"),
        primary_comparison="SwitchSure vs Whisper zero-shot",
        primary_metric="critical_entity_error_rate",
        seeds=(17, 29, 43),
        reference_hardware="NVIDIA L4, CUDA pinned in environment lock",
    )

    assert spec.primary_metric == "critical_entity_error_rate"
    with pytest.raises(FrozenInstanceError):
        spec.primary_metric = "wer"  # type: ignore[misc]


def test_experiment_spec_rejects_locked_test_tuning() -> None:
    with pytest.raises(ValueError, match="locked test"):
        BaselineExperimentSpec(
            "experiment",
            "commit",
            "lock",
            "manifest",
            "normalization-v1",
            split_manifest(),
            ArtifactRevision("model", "rev"),
            ArtifactRevision("calibrator", "rev"),
            ArtifactRevision("policy", "rev"),
            "comparison",
            "metric",
            (17,),
            "hardware",
            tuning_partitions=("train", "test"),
        )


@pytest.mark.parametrize("leakage", ["speaker", "session"])
def test_split_manifest_rejects_speaker_and_session_leakage(leakage: str) -> None:
    train = partition("train", "shared" if leakage == "speaker" else "speaker-a", "shared")
    calibration = partition(
        "calibration", "shared" if leakage == "speaker" else "speaker-b", "shared"
    )
    with pytest.raises(ValueError, match=f"{leakage} leakage"):
        SplitManifest(
            train=train,
            calibration=calibration,
            validation=partition("validation", "speaker-c", "session-c"),
            test=partition("test", "speaker-d", "session-d"),
        )


def test_error_taxonomy_is_declared_and_duplicate_free() -> None:
    counts = ErrorTaxonomyCounts(
        (
            ErrorCount(ErrorCategory.ENTITY_VALUE_ERROR, 3),
            ErrorCount(ErrorCategory.HIGH_SEVERITY_ACCEPTED_ERROR, 1),
        )
    )

    assert counts.total == 4
    assert counts.count(ErrorCategory.ENTITY_VALUE_ERROR) == 3
    assert counts.count(ErrorCategory.ORDINARY_INSERTION) == 0
    with pytest.raises(ValueError, match="only once"):
        ErrorTaxonomyCounts(
            (
                ErrorCount(ErrorCategory.ENTITY_VALUE_ERROR, 1),
                ErrorCount(ErrorCategory.ENTITY_VALUE_ERROR, 2),
            )
        )
    with pytest.raises(ValueError, match="negative"):
        ErrorCount(ErrorCategory.ORDINARY_DELETION, -1)


def test_every_non_success_outcome_remains_visible() -> None:
    outcomes = tuple(PredictionOutcome)
    counts = OutcomeCounts.from_outcomes(outcomes)

    assert counts.total == len(PredictionOutcome)
    assert counts.success == 1
    assert counts.non_success == len(PredictionOutcome) - 1
    assert counts.timeout == 1
    assert counts.invalid_audio == 1
    assert counts.decoder_failure == 1
    assert counts.abstention == 1


def test_subgroup_summary_reports_denominator_failures_and_privacy_suppression() -> None:
    observations = (
        SubgroupObservation("s1", "device", "mobile", 1, 4, PredictionOutcome.SUCCESS),
        SubgroupObservation("s2", "device", "mobile", 1, 6, PredictionOutcome.TIMEOUT),
        SubgroupObservation("s1", "noise", "high", 0, 0, PredictionOutcome.ABSTENTION),
    )

    summaries = summarize_subgroups(observations, min_speakers=2)
    mobile = next(summary for summary in summaries if summary.slice_name == "device")
    high_noise = next(summary for summary in summaries if summary.slice_name == "noise")

    assert mobile.metric == pytest.approx(0.2)
    assert mobile.denominator == 10
    assert mobile.outcome_counts.timeout == 1
    assert not mobile.suppressed
    assert high_noise.metric is None
    assert high_noise.suppressed
    assert high_noise.suppression_reason == "fewer than 2 unique speakers"


def test_subgroup_zero_denominator_is_explicitly_undefined() -> None:
    observations = tuple(
        SubgroupObservation(f"s{index}", "entity", "medicine", 0, 0, PredictionOutcome.MISSING)
        for index in range(2)
    )

    (summary,) = summarize_subgroups(observations, min_speakers=2)

    assert summary.metric is None
    assert summary.suppression_reason == "metric denominator is zero"
    assert summary.outcome_counts.missing == 2


def test_paired_speaker_bootstrap_is_seeded_and_clustered() -> None:
    pairs = tuple(
        SpeakerMetricPair(f"s{index}", errors, 10, errors - 1, 10)
        for index, errors in enumerate((5, 6, 4, 7, 3), start=1)
    )

    first = paired_speaker_bootstrap(pairs, resamples=500, seed=31)
    second = paired_speaker_bootstrap(pairs, resamples=500, seed=31)

    assert first == second
    assert first.estimate == pytest.approx(-0.1)
    assert first.upper < 0
    assert first.speaker_count == 5
    assert first.resamples == 500


def test_paired_bootstrap_rejects_unaggregated_duplicate_speakers() -> None:
    pair = SpeakerMetricPair("same", 2, 10, 1, 10)
    with pytest.raises(ValueError, match="aggregate utterances"):
        paired_speaker_bootstrap((pair, pair), resamples=100)


def test_release_gates_pass_fail_and_missing_evidence_mechanically() -> None:
    gates = (
        ReleaseGate(
            "calibration",
            "Calibration must be low.",
            (GateClause("ece", GateOperator.AT_MOST, 0.05),),
        ),
        ReleaseGate(
            "supported-improvement",
            "The confidence interval must exclude zero.",
            (GateClause("difference", GateOperator.CI_UPPER_BELOW, 0.0),),
        ),
        ReleaseGate(
            "alternative",
            "Either metric may establish the benefit.",
            (
                GateClause("missing", GateOperator.AT_LEAST, 0.2),
                GateClause("relative_reduction", GateOperator.AT_LEAST, 0.2),
            ),
        ),
    )
    decision = evaluate_release_gates(
        gates,
        {
            "ece": MetricEstimate(0.08, denominator=100),
            "difference": MetricEstimate(-0.02),
            "relative_reduction": MetricEstimate(0.25),
        },
    )

    assert [result.status for result in decision.gate_results] == [
        GateStatus.FAIL,
        GateStatus.NOT_EVALUABLE,
        GateStatus.PASS,
    ]
    assert not decision.ready


def test_phase1_release_requires_all_declared_evidence() -> None:
    passing = {
        "general_wer_absolute_difference": MetricEstimate(0.01),
        "critical_entity_error_relative_reduction": MetricEstimate(0.20),
        "switch_region_wer_relative_reduction": MetricEstimate(0.15),
        "word_error_ece": MetricEstimate(0.05),
        "major_slice_max_word_error_ece": MetricEstimate(0.08),
        "aurc_difference": MetricEstimate(-0.02, lower=-0.03, upper=-0.01),
        "major_subgroup_max_severity_ratio": MetricEstimate(1.5),
        "p95_latency_target_met": MetricEstimate(1.0),
        "critical_accessibility_violations": MetricEstimate(0.0),
    }

    decision = evaluate_release_gates(PHASE1_RELEASE_GATES, passing)

    assert decision.ready
    assert all(result.status == GateStatus.PASS for result in decision.gate_results)
    incomplete = evaluate_release_gates(PHASE1_RELEASE_GATES, {})
    assert not incomplete.ready
    assert all(result.status == GateStatus.NOT_EVALUABLE for result in incomplete.gate_results)


def test_validation_rejects_invalid_statistical_inputs() -> None:
    with pytest.raises(ValueError, match="at least two paired speakers"):
        paired_speaker_bootstrap((SpeakerMetricPair("s1", 1, 2, 0, 2),), resamples=100)
    with pytest.raises(ValueError, match="at least 100"):
        paired_speaker_bootstrap(
            (SpeakerMetricPair("s1", 1, 2, 0, 2), SpeakerMetricPair("s2", 1, 2, 0, 2)),
            resamples=99,
        )
    with pytest.raises(ValueError, match="between zero and one"):
        paired_speaker_bootstrap(
            (SpeakerMetricPair("s1", 1, 2, 0, 2), SpeakerMetricPair("s2", 1, 2, 0, 2)),
            resamples=100,
            confidence_level=1.0,
        )
    with pytest.raises(ValueError, match="at least one release gate"):
        evaluate_release_gates((), {})
