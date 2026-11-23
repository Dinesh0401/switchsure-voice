"""Application-independent transcript analysis orchestration."""

from __future__ import annotations

from collections.abc import Iterable

from .domain import (
    CalibrationStatus,
    Criticality,
    EntityEvidence,
    RiskBearingTranscript,
    TranscriptEvidence,
)
from .entities import detect_structured_entities
from .risk import CostSensitivePolicy, HeuristicRiskModel


def analyze_transcript(
    transcript: TranscriptEvidence,
    *,
    supplied_entities: Iterable[EntityEvidence] = (),
    risk_model: HeuristicRiskModel | None = None,
    policy: CostSensitivePolicy | None = None,
) -> RiskBearingTranscript:
    model = risk_model or HeuristicRiskModel()
    decision_policy = policy or CostSensitivePolicy()
    entities = tuple(supplied_entities) or detect_structured_entities(transcript.words)

    criticality_by_word: dict[int, Criticality] = {}
    for entity in entities:
        if entity.end_word >= len(transcript.words):
            raise ValueError("entity span exceeds transcript words")
        for index in range(entity.start_word, entity.end_word + 1):
            existing = criticality_by_word.get(index)
            if existing is None or entity.criticality > existing:
                criticality_by_word[index] = entity.criticality

    assessed_words = tuple(
        model.assess_word(word, criticality_by_word.get(word.index)) for word in transcript.words
    )
    assessed_entities = tuple(model.assess_entity(entity, assessed_words) for entity in entities)
    decision = decision_policy.decide(assessed_words, assessed_entities)

    return RiskBearingTranscript(
        transcript_id=transcript.transcript_id,
        text=transcript.text,
        words=assessed_words,
        entities=assessed_entities,
        action=decision.action,
        reason_codes=decision.reason_codes,
        model_version=transcript.model_version,
        risk_model_version=model.version,
        policy_version=decision_policy.version,
        calibration_status=CalibrationStatus.UNCALIBRATED_BASELINE,
    )
