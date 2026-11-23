"""SwitchSure's framework-independent domain."""

from .domain import (
    Alternative,
    AssessedEntity,
    AssessedWord,
    CalibrationStatus,
    Criticality,
    DecisionAction,
    EntityEvidence,
    EntityType,
    LanguageProbability,
    RiskBearingTranscript,
    TranscriptEvidence,
    WordEvidence,
)
from .entities import detect_structured_entities
from .orchestrator import analyze_transcript
from .risk import CostSensitivePolicy, HeuristicRiskModel

__all__ = [
    "Alternative",
    "AssessedEntity",
    "AssessedWord",
    "CalibrationStatus",
    "CostSensitivePolicy",
    "Criticality",
    "DecisionAction",
    "EntityEvidence",
    "EntityType",
    "HeuristicRiskModel",
    "LanguageProbability",
    "RiskBearingTranscript",
    "TranscriptEvidence",
    "WordEvidence",
    "analyze_transcript",
    "detect_structured_entities",
]
