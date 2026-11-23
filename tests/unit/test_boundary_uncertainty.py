import math
from dataclasses import replace

import pytest
from switchsure_ml.boundary_uncertainty import (
    BoundaryAwareUncertaintyModel,
    BoundaryFeatureMode,
    BoundaryObservation,
    MatchedBoundaryPair,
    match_boundary_controls,
)
from switchsure_ml.dataset_manifest import DataSplit


def _observation(
    index: int,
    *,
    split: DataSplit = DataSplit.TRAIN,
    boundary: bool = False,
    error: bool = False,
    speaker: str | None = None,
    session: str | None = None,
    language: str = "ta",
    entity: bool = False,
    acoustic_band: str = "clean",
    token_length: int = 5,
) -> BoundaryObservation:
    high = boundary or error
    return BoundaryObservation(
        example_id=f"{split.value}-{index}",
        record_id=f"record-{split.value}-{index // 2}",
        speaker_id=speaker or f"speaker-{split.value}-{index // 2}",
        session_id=session or f"session-{split.value}-{index // 2}",
        split=split,
        token_index=index,
        language=language,
        acoustic_band=acoustic_band,
        token_length=token_length,
        is_entity=entity,
        decoder_error_evidence=0.85 if error else 0.08,
        lid_entropy=0.90 if boundary else 0.10,
        language_transition_evidence=0.92 if boundary else 0.05,
        timestamp_gap_ms=180.0 if boundary else 15.0,
        hypothesis_disagreement=0.80 if high else 0.05,
        target_is_boundary=boundary,
        target_is_error=error,
    )


def _training_examples() -> tuple[BoundaryObservation, ...]:
    return tuple(
        _observation(
            index,
            boundary=index % 2 == 0,
            error=index % 3 == 0 or index % 4 == 0,
        )
        for index in range(24)
    )


def _calibration_examples() -> tuple[BoundaryObservation, ...]:
    return tuple(
        _observation(
            index,
            split=DataSplit.CALIBRATION,
            boundary=index % 2 == 0,
            error=index % 3 == 0,
        )
        for index in range(12)
    )


def test_observation_rejects_invalid_evidence_without_storing_surface_text() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        replace(_observation(1), decoder_error_evidence=1.01)

    observation = _observation(1)

    assert not hasattr(observation, "surface")
    assert observation.control_stratum == ("ta", False, "clean", 1)


def test_boundary_model_learns_separable_boundary_and_error_evidence() -> None:
    model = BoundaryAwareUncertaintyModel.fit(_training_examples())
    boundary_risk = model.predict(
        replace(_observation(100, boundary=True, error=True), target_is_boundary=None)
    )
    plain_risk = model.predict(
        replace(_observation(101, boundary=False, error=False), target_is_boundary=None)
    )

    assert boundary_risk.boundary_probability > plain_risk.boundary_probability
    assert boundary_risk.raw_error_logit > plain_risk.raw_error_logit
    assert boundary_risk.calibrated_error_probability is None
    assert 0.0 <= boundary_risk.boundary_entropy <= 1.0


def test_boundary_aware_and_baseline_modes_are_explicit_ablations() -> None:
    examples = _training_examples()
    baseline = BoundaryAwareUncertaintyModel.fit(
        examples, feature_mode=BoundaryFeatureMode.BASELINE
    )
    aware = BoundaryAwareUncertaintyModel.fit(
        examples, feature_mode=BoundaryFeatureMode.BOUNDARY_AWARE
    )
    candidate = replace(
        _observation(40, boundary=True, error=False),
        decoder_error_evidence=0.25,
        target_is_boundary=None,
        target_is_error=None,
    )

    baseline_output = baseline.predict(candidate)
    aware_output = aware.predict(candidate)

    assert baseline_output.feature_mode is BoundaryFeatureMode.BASELINE
    assert aware_output.feature_mode is BoundaryFeatureMode.BOUNDARY_AWARE
    assert baseline_output.raw_error_logit != pytest.approx(aware_output.raw_error_logit)


def test_calibration_is_fit_only_on_disjoint_calibration_split() -> None:
    model = BoundaryAwareUncertaintyModel.fit(_training_examples())

    model.calibrate(_calibration_examples())
    output = model.predict(
        replace(
            _observation(99, split=DataSplit.TEST, boundary=True, error=True),
            target_is_boundary=None,
            target_is_error=None,
        )
    )

    assert model.is_calibrated
    assert model.calibration_temperature is not None
    assert math.isfinite(model.calibration_temperature)
    assert output.calibrated_error_probability is not None
    assert 0.0 <= output.calibrated_error_probability <= 1.0


def test_calibration_rejects_training_or_validation_examples() -> None:
    model = BoundaryAwareUncertaintyModel.fit(_training_examples())

    with pytest.raises(ValueError, match="calibration split"):
        model.calibrate(_training_examples())

    validation = tuple(
        replace(item, split=DataSplit.VALIDATION, example_id=f"val-{item.example_id}")
        for item in _calibration_examples()
    )
    with pytest.raises(ValueError, match="calibration split"):
        model.calibrate(validation)


def test_calibration_rejects_speaker_and_session_leakage() -> None:
    training = _training_examples()
    speaker_leak = tuple(
        replace(
            item,
            speaker_id=training[0].speaker_id if index == 0 else item.speaker_id,
        )
        for index, item in enumerate(_calibration_examples())
    )
    model = BoundaryAwareUncertaintyModel.fit(training)

    with pytest.raises(ValueError, match="speakers overlap"):
        model.calibrate(speaker_leak)

    session_leak = tuple(
        replace(
            item,
            session_id=training[0].session_id if index == 0 else item.session_id,
        )
        for index, item in enumerate(_calibration_examples())
    )
    model = BoundaryAwareUncertaintyModel.fit(training)

    with pytest.raises(ValueError, match="sessions overlap"):
        model.calibrate(session_leak)


def test_calibration_is_single_use_to_keep_artifact_provenance_stable() -> None:
    model = BoundaryAwareUncertaintyModel.fit(_training_examples())
    model.calibrate(_calibration_examples())

    with pytest.raises(RuntimeError, match="already been calibrated"):
        model.calibrate(_calibration_examples())


def test_training_rejects_non_train_split_missing_labels_and_duplicate_ids() -> None:
    examples = list(_training_examples())
    examples[0] = replace(examples[0], split=DataSplit.VALIDATION)
    with pytest.raises(ValueError, match="only the train split"):
        BoundaryAwareUncertaintyModel.fit(examples)

    missing_label = list(_training_examples())
    missing_label[0] = replace(missing_label[0], target_is_error=None)
    with pytest.raises(ValueError, match="gold word-error labels"):
        BoundaryAwareUncertaintyModel.fit(missing_label)

    duplicate = [*_training_examples(), _training_examples()[0]]
    with pytest.raises(ValueError, match="example IDs must be unique"):
        BoundaryAwareUncertaintyModel.fit(duplicate)


def test_training_requires_both_binary_classes() -> None:
    all_negative_boundaries = tuple(
        replace(item, target_is_boundary=False) for item in _training_examples()
    )
    with pytest.raises(ValueError, match="both positive and negative"):
        BoundaryAwareUncertaintyModel.fit(all_negative_boundaries)

    all_correct = tuple(replace(item, target_is_error=False) for item in _training_examples())
    with pytest.raises(ValueError, match="both positive and negative"):
        BoundaryAwareUncertaintyModel.fit(all_correct)


def test_training_rejects_an_undeclared_feature_mode() -> None:
    with pytest.raises(ValueError, match="feature_mode"):
        BoundaryAwareUncertaintyModel.fit(_training_examples(), feature_mode="boundary")  # type: ignore[arg-type]


def test_matched_controls_are_exact_deterministic_and_not_reused() -> None:
    boundary_one = _observation(10, split=DataSplit.TEST, boundary=True, token_length=5)
    boundary_two = _observation(11, split=DataSplit.TEST, boundary=True, token_length=6)
    preferred_same_speaker = _observation(
        20,
        split=DataSplit.TEST,
        boundary=False,
        token_length=7,
        speaker=boundary_one.speaker_id,
    )
    closer_other_speaker = _observation(
        21,
        split=DataSplit.TEST,
        boundary=False,
        token_length=5,
    )

    result = match_boundary_controls(
        (boundary_two, closer_other_speaker, boundary_one, preferred_same_speaker)
    )

    assert tuple(pair.boundary.example_id for pair in result.pairs) == (
        boundary_one.example_id,
        boundary_two.example_id,
    )
    assert result.pairs[0].control.example_id == preferred_same_speaker.example_id
    assert len({pair.control.example_id for pair in result.pairs}) == 2
    assert result.unmatched_boundary_ids == ()


def test_matched_controls_report_unmatched_and_reject_cross_split_matching() -> None:
    boundary = _observation(
        1,
        split=DataSplit.TEST,
        boundary=True,
        language="en",
        entity=True,
    )
    wrong_stratum = _observation(2, split=DataSplit.TEST, boundary=False, language="ta")

    result = match_boundary_controls((boundary, wrong_stratum))

    assert result.pairs == ()
    assert result.unmatched_boundary_ids == (boundary.example_id,)
    with pytest.raises(ValueError, match="must not cross"):
        match_boundary_controls((boundary, replace(wrong_stratum, split=DataSplit.VALIDATION)))


def test_matched_pair_contract_rejects_non_exact_control() -> None:
    boundary = _observation(1, split=DataSplit.TEST, boundary=True, language="en")
    control = _observation(2, split=DataSplit.TEST, boundary=False, language="ta")

    with pytest.raises(ValueError, match="control stratum"):
        MatchedBoundaryPair(boundary, control)


def test_predict_many_preserves_order_and_metric_targets() -> None:
    model = BoundaryAwareUncertaintyModel.fit(_training_examples())
    examples = (
        _observation(50, split=DataSplit.VALIDATION, boundary=True, error=False),
        _observation(51, split=DataSplit.VALIDATION, boundary=False, error=True),
    )

    outputs = model.predict_many(examples)

    assert tuple(output.example_id for output in outputs) == tuple(
        example.example_id for example in examples
    )
    assert outputs[0].target_is_boundary is True
    assert outputs[1].target_is_error is True
