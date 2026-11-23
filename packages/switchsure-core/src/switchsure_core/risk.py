"""Baseline risk scoring and cost-sensitive selective prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .domain import (
    AssessedEntity,
    AssessedWord,
    Criticality,
    DecisionAction,
    EntityEvidence,
    WordEvidence,
)


def _entropy(probabilities: tuple[float, ...]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    value = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    return value / math.log(len(probabilities))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _binary_entropy(probability: float) -> float:
    if probability in (0.0, 1.0):
        return 0.0
    return -(probability * math.log2(probability) + (1 - probability) * math.log2(1 - probability))


@dataclass(frozen=True, slots=True)
class HeuristicRiskModel:
    """Transparent vertical-slice baseline; it is intentionally not called calibrated."""

    version: str = "heuristic-v1"

    def assess_word(self, word: WordEvidence, criticality: Criticality | None) -> AssessedWord:
        language_entropy = _entropy(tuple(item.probability for item in word.languages))
        boundary_uncertainty = _binary_entropy(word.boundary_probability)
        alternative_ambiguity = 0.0
        if word.alternatives:
            alternative_ambiguity = 1.0 - max(item.probability for item in word.alternatives)
        criticality_feature = 0.0 if criticality is None else float(criticality) / 4.0

        logit = (
            -3.0
            + 3.6 * (1.0 - word.decoder_confidence)
            + 0.8 * language_entropy
            + 0.7 * boundary_uncertainty
            + 0.7 * alternative_ambiguity
            + 1.2 * (1.0 - word.acoustic_quality)
            + 0.25 * criticality_feature
        )
        return AssessedWord(
            evidence=word,
            recognition_risk=_sigmoid(logit),
            boundary_uncertainty=boundary_uncertainty,
        )

    def assess_entity(
        self, entity: EntityEvidence, assessed_words: tuple[AssessedWord, ...]
    ) -> AssessedEntity:
        span = assessed_words[entity.start_word : entity.end_word + 1]
        if not span:
            raise ValueError("entity span does not overlap transcript words")
        word_risk = max(word.recognition_risk for word in span)
        detector_risk = 1.0 - entity.detector_confidence
        alternative_risk = 0.0
        if entity.alternatives:
            alternative_risk = 1.0 - max(item.probability for item in entity.alternatives)
        combined = 1.0 - (1.0 - word_risk) * (1.0 - detector_risk) * (1.0 - alternative_risk)
        return AssessedEntity(evidence=entity, preservation_risk=min(1.0, combined))


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: DecisionAction
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CostSensitivePolicy:
    """Choose the lowest expected-cost action from baseline risk estimates."""

    version: str = "cost-policy-v1"
    review_cost: float = 0.18
    defer_cost: float = 0.55
    residual_review_error_fraction: float = 0.12

    def decide(
        self,
        words: tuple[AssessedWord, ...],
        entities: tuple[AssessedEntity, ...],
    ) -> PolicyDecision:
        candidates: list[tuple[float, float, str]] = [
            (word.recognition_risk, 1.0, "word-risk") for word in words
        ]
        candidates.extend(
            (
                entity.preservation_risk,
                float(entity.evidence.criticality),
                f"critical-entity:{entity.evidence.entity_type}",
            )
            for entity in entities
        )
        risk, severity, reason = max(candidates, key=lambda item: item[0] * item[1])

        costs = {
            DecisionAction.ACCEPT: risk * severity,
            DecisionAction.REVIEW: self.review_cost
            + risk * severity * self.residual_review_error_fraction,
            DecisionAction.DEFER: self.defer_cost,
        }
        action = min(costs, key=costs.__getitem__)
        reason_codes = () if action is DecisionAction.ACCEPT else (reason,)
        return PolicyDecision(action=action, reason_codes=reason_codes)
