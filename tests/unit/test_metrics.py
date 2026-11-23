import pytest
from switchsure_eval import (
    brier_score,
    expected_calibration_error,
    risk_coverage_curve,
    word_error_counts,
    word_error_rate,
)


def test_word_error_metrics_preserve_operation_counts() -> None:
    reference = ["take", "500", "mg"]
    hypothesis = ["take", "50", "mg", "today"]

    counts = word_error_counts(reference, hypothesis)

    assert counts.substitutions == 1
    assert counts.insertions == 1
    assert counts.deletions == 0
    assert counts.correct == 2
    assert word_error_rate(reference, hypothesis) == pytest.approx(2 / 3)


def test_calibration_metrics() -> None:
    probabilities = [0.9, 0.8, 0.2, 0.1]
    outcomes = [1, 1, 0, 0]

    assert brier_score(probabilities, outcomes) == pytest.approx(0.025)
    assert expected_calibration_error(probabilities, outcomes, bins=5) == pytest.approx(0.15)


def test_risk_coverage_accepts_low_risk_first() -> None:
    curve = risk_coverage_curve([0.8, 0.1, 0.3], [1.0, 0.0, 0.0])

    first_coverage, first_risk = curve[0]
    final_coverage, final_risk = curve[-1]

    assert first_coverage == pytest.approx(1 / 3)
    assert first_risk == 0.0
    assert final_coverage == 1.0
    assert final_risk == pytest.approx(1 / 3)
