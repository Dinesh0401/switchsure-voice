"""Dependency-light uncertainty primitives with explicit mathematical semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _validate_distribution(probabilities: Sequence[float]) -> None:
    if not probabilities:
        raise ValueError("probability distribution cannot be empty")
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("probabilities must be between 0 and 1")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-7):
        raise ValueError("probabilities must sum to 1")


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Return Shannon entropy divided by log of support size."""

    _validate_distribution(probabilities)
    if len(probabilities) == 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
    return entropy / math.log(len(probabilities))


def predictive_mutual_information(draws: Sequence[Sequence[float]]) -> float:
    """Estimate epistemic uncertainty as H[E p] - E[H p]."""

    if not draws:
        raise ValueError("at least one predictive draw is required")
    width = len(draws[0])
    if width == 0 or any(len(draw) != width for draw in draws):
        raise ValueError("predictive draws must have equal non-zero width")
    for draw in draws:
        _validate_distribution(draw)

    mean = [sum(draw[index] for draw in draws) / len(draws) for index in range(width)]

    def entropy(values: Sequence[float]) -> float:
        return -sum(value * math.log(value) for value in values if value > 0.0)

    mutual_information = entropy(mean) - sum(entropy(draw) for draw in draws) / len(draws)
    return max(0.0, mutual_information)


def _levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_token in enumerate(left, start=1):
        current = [row]
        for column, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def pairwise_sequence_disagreement(candidates: Sequence[Sequence[str]]) -> float:
    """Average pairwise edit distance normalized by the longer candidate."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    if len(candidates) == 1:
        return 0.0
    values: list[float] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            denominator = max(len(left), len(right), 1)
            values.append(_levenshtein(left, right) / denominator)
    return sum(values) / len(values)


def conformal_quantile(nonconformity_scores: Sequence[float], alpha: float) -> float:
    """Finite-sample split-conformal quantile with the ceiling correction."""

    if not nonconformity_scores:
        raise ValueError("calibration scores cannot be empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if any(score < 0.0 or score > 1.0 for score in nonconformity_scores):
        raise ValueError("nonconformity scores must be between 0 and 1")

    ordered = sorted(nonconformity_scores)
    rank = math.ceil((len(ordered) + 1) * (1.0 - alpha))
    rank = min(max(rank, 1), len(ordered))
    return ordered[rank - 1]


def prediction_set(probabilities: Mapping[str, float], quantile: float) -> tuple[str, ...]:
    """Return labels whose 1 - probability does not exceed the conformal quantile."""

    if not probabilities:
        raise ValueError("probabilities cannot be empty")
    _validate_distribution(tuple(probabilities.values()))
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    return tuple(
        label for label, probability in probabilities.items() if 1.0 - probability <= quantile
    )
