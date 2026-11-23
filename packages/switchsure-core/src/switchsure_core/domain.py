"""Immutable domain objects shared by product and research code."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


class DecisionAction(StrEnum):
    ACCEPT = "accept"
    REVIEW = "review"
    DEFER = "defer"


class CalibrationStatus(StrEnum):
    UNCALIBRATED_BASELINE = "uncalibrated_baseline"
    CALIBRATED = "calibrated"


class Criticality(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 4


class EntityType(StrEnum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    MEDICATION = "medication"
    DOSAGE = "dosage"
    MEDICAL_CONDITION = "medical_condition"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    IDENTIFIER = "identifier"
    DATE = "date"
    TIME = "time"
    QUANTITY = "quantity"
    MONEY = "money"


@dataclass(frozen=True, slots=True)
class LanguageProbability:
    code: str
    probability: float

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("language code cannot be empty")
        _validate_probability("language probability", self.probability)


@dataclass(frozen=True, slots=True)
class Alternative:
    surface: str
    probability: float

    def __post_init__(self) -> None:
        if not self.surface.strip():
            raise ValueError("alternative surface cannot be empty")
        _validate_probability("alternative probability", self.probability)


@dataclass(frozen=True, slots=True)
class WordEvidence:
    index: int
    surface: str
    start_ms: int
    end_ms: int
    decoder_confidence: float
    acoustic_quality: float = 1.0
    boundary_probability: float = 0.0
    languages: tuple[LanguageProbability, ...] = field(default_factory=tuple)
    alternatives: tuple[Alternative, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("word index cannot be negative")
        if not self.surface.strip():
            raise ValueError("word surface cannot be empty")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("word timestamps are invalid")
        _validate_probability("decoder confidence", self.decoder_confidence)
        _validate_probability("acoustic quality", self.acoustic_quality)
        _validate_probability("boundary probability", self.boundary_probability)
        if self.languages:
            total = sum(item.probability for item in self.languages)
            if abs(total - 1.0) > 1e-4:
                raise ValueError("language probabilities must sum to 1")


@dataclass(frozen=True, slots=True)
class EntityEvidence:
    start_word: int
    end_word: int
    entity_type: EntityType
    surface: str
    criticality: Criticality
    detector_confidence: float
    canonical_value: str | None = None
    alternatives: tuple[Alternative, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.start_word < 0 or self.end_word < self.start_word:
            raise ValueError("entity word span is invalid")
        if not self.surface.strip():
            raise ValueError("entity surface cannot be empty")
        _validate_probability("entity detector confidence", self.detector_confidence)


@dataclass(frozen=True, slots=True)
class TranscriptEvidence:
    transcript_id: str
    words: tuple[WordEvidence, ...]
    model_version: str

    def __post_init__(self) -> None:
        if not self.transcript_id.strip():
            raise ValueError("transcript id cannot be empty")
        if not self.words:
            raise ValueError("transcript must contain at least one word")
        expected = list(range(len(self.words)))
        actual = [word.index for word in self.words]
        if actual != expected:
            raise ValueError("word indexes must be contiguous and start at zero")

    @property
    def text(self) -> str:
        return " ".join(word.surface for word in self.words)


@dataclass(frozen=True, slots=True)
class AssessedWord:
    evidence: WordEvidence
    recognition_risk: float
    boundary_uncertainty: float

    def __post_init__(self) -> None:
        _validate_probability("recognition risk", self.recognition_risk)
        _validate_probability("boundary uncertainty", self.boundary_uncertainty)


@dataclass(frozen=True, slots=True)
class AssessedEntity:
    evidence: EntityEvidence
    preservation_risk: float

    def __post_init__(self) -> None:
        _validate_probability("entity preservation risk", self.preservation_risk)


@dataclass(frozen=True, slots=True)
class RiskBearingTranscript:
    transcript_id: str
    text: str
    words: tuple[AssessedWord, ...]
    entities: tuple[AssessedEntity, ...]
    action: DecisionAction
    reason_codes: tuple[str, ...]
    model_version: str
    risk_model_version: str
    policy_version: str
    calibration_status: CalibrationStatus
