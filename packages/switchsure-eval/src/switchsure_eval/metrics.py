"""Metrics with explicit edge-case behavior and no hidden normalization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    correct: int

    @property
    def reference_length(self) -> int:
        return self.substitutions + self.deletions + self.correct

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def word_error_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Return a deterministic optimal Levenshtein alignment's operation counts."""

    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    operations = [[(0, 0, 0, 0)] * columns for _ in range(rows)]

    for row in range(1, rows):
        costs[row][0] = row
        operations[row][0] = (0, row, 0, 0)
    for column in range(1, columns):
        costs[0][column] = column
        operations[0][column] = (0, 0, column, 0)

    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                costs[row][column] = costs[row - 1][column - 1]
                s, d, i, c = operations[row - 1][column - 1]
                operations[row][column] = (s, d, i, c + 1)
                continue

            candidates = (
                (costs[row - 1][column - 1] + 1, 0),  # substitution
                (costs[row - 1][column] + 1, 1),  # deletion
                (costs[row][column - 1] + 1, 2),  # insertion
            )
            cost, operation = min(candidates)
            costs[row][column] = cost
            if operation == 0:
                s, d, i, c = operations[row - 1][column - 1]
                operations[row][column] = (s + 1, d, i, c)
            elif operation == 1:
                s, d, i, c = operations[row - 1][column]
                operations[row][column] = (s, d + 1, i, c)
            else:
                s, d, i, c = operations[row][column - 1]
                operations[row][column] = (s, d, i + 1, c)

    return EditCounts(*operations[-1][-1])


def word_error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    counts = word_error_counts(reference, hypothesis)
    if counts.reference_length == 0:
        raise ValueError("WER is undefined for an empty reference")
    return counts.errors / counts.reference_length


def brier_score(probability_correct: Sequence[float], outcomes: Sequence[int]) -> float:
    if len(probability_correct) != len(outcomes) or not outcomes:
        raise ValueError("probabilities and non-empty outcomes must have equal length")
    if any(value < 0.0 or value > 1.0 or not isfinite(value) for value in probability_correct):
        raise ValueError("probabilities must be finite and between 0 and 1")
    if any(outcome not in (0, 1) for outcome in outcomes):
        raise ValueError("outcomes must be binary")
    return sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(probability_correct, outcomes, strict=True)
    ) / len(outcomes)


def expected_calibration_error(
    probability_correct: Sequence[float], outcomes: Sequence[int], *, bins: int = 10
) -> float:
    if len(probability_correct) != len(outcomes) or not outcomes:
        raise ValueError("probabilities and non-empty outcomes must have equal length")
    if bins < 2:
        raise ValueError("at least two bins are required")
    if any(value < 0.0 or value > 1.0 for value in probability_correct):
        raise ValueError("probabilities must be between 0 and 1")
    if any(outcome not in (0, 1) for outcome in outcomes):
        raise ValueError("outcomes must be binary")

    total = len(outcomes)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indexes = [
            index
            for index, probability in enumerate(probability_correct)
            if lower <= probability < upper or (bin_index == bins - 1 and probability == 1.0)
        ]
        if not indexes:
            continue
        confidence = sum(probability_correct[index] for index in indexes) / len(indexes)
        accuracy = sum(outcomes[index] for index in indexes) / len(indexes)
        error += len(indexes) / total * abs(accuracy - confidence)
    return error


def risk_coverage_curve(
    error_probabilities: Sequence[float], losses: Sequence[float]
) -> tuple[tuple[float, float], ...]:
    """Return empirical (coverage, selective risk), accepting lowest risk first."""

    if len(error_probabilities) != len(losses) or not losses:
        raise ValueError("risk estimates and non-empty losses must have equal length")
    if any(value < 0.0 or value > 1.0 for value in error_probabilities):
        raise ValueError("error probabilities must be between 0 and 1")
    if any(loss < 0.0 for loss in losses):
        raise ValueError("losses cannot be negative")

    ranked = sorted(zip(error_probabilities, losses, strict=True), key=lambda item: item[0])
    cumulative_loss = 0.0
    curve: list[tuple[float, float]] = []
    for accepted, (_, loss) in enumerate(ranked, start=1):
        cumulative_loss += loss
        curve.append((accepted / len(ranked), cumulative_loss / accepted))
    return tuple(curve)
