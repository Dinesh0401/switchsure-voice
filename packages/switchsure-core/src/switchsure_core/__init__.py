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
from .jobs import (
    InvalidJobTransitionError,
    JobLease,
    JobNotFoundError,
    JobQueueError,
    JobResultUnavailableError,
    JobSnapshot,
    JobState,
    PilotJobQueue,
    QueueFullError,
    RetentionClass,
)
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
    "InvalidJobTransitionError",
    "JobLease",
    "JobNotFoundError",
    "JobQueueError",
    "JobResultUnavailableError",
    "JobSnapshot",
    "JobState",
    "LanguageProbability",
    "PilotJobQueue",
    "QueueFullError",
    "RetentionClass",
    "RiskBearingTranscript",
    "TranscriptEvidence",
    "WordEvidence",
    "analyze_transcript",
    "detect_structured_entities",
]
