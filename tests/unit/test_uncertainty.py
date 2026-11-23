import math

import pytest
from switchsure_ml import (
    conformal_quantile,
    normalized_entropy,
    pairwise_sequence_disagreement,
    prediction_set,
    predictive_mutual_information,
)


def test_normalized_entropy_has_expected_extremes() -> None:
    assert normalized_entropy([1.0, 0.0]) == 0.0
    assert normalized_entropy([0.5, 0.5]) == pytest.approx(1.0)


def test_mutual_information_detects_model_disagreement() -> None:
    identical = predictive_mutual_information([[0.8, 0.2], [0.8, 0.2]])
    disagreement = predictive_mutual_information([[0.95, 0.05], [0.05, 0.95]])

    assert identical == pytest.approx(0.0, abs=1e-12)
    assert disagreement > 0.4
    assert math.isfinite(disagreement)


def test_pairwise_sequence_disagreement_uses_edit_distance() -> None:
    value = pairwise_sequence_disagreement(
        [("எனக்கு", "medicine"), ("எனக்கு", "மருந்து"), ("எனக்கு", "medicine")]
    )

    assert value == pytest.approx(1 / 3)


def test_split_conformal_quantile_and_set() -> None:
    quantile = conformal_quantile([0.05, 0.10, 0.20, 0.25, 0.40], alpha=0.2)
    labels = prediction_set({"500 mg": 0.72, "50 mg": 0.23, "5 mg": 0.05}, quantile)

    assert quantile == 0.40
    assert labels == ("500 mg",)
