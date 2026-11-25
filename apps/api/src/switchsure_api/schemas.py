"""Strict external API schemas and domain mappings."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from switchsure_core import (
    Alternative,
    Criticality,
    EntityEvidence,
    EntityType,
    LanguageProbability,
    RiskBearingTranscript,
    TranscriptEvidence,
    WordEvidence,
)

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LanguageProbabilityIn(StrictModel):
    code: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=16)]
    probability: float = Field(ge=0.0, le=1.0)


class AlternativeIn(StrictModel):
    surface: ShortText
    probability: float = Field(ge=0.0, le=1.0)


class WordEvidenceIn(StrictModel):
    surface: ShortText
    start_ms: int = Field(ge=0, le=1_800_000)
    end_ms: int = Field(ge=0, le=1_800_000)
    decoder_confidence: float = Field(ge=0.0, le=1.0)
    acoustic_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    boundary_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    languages: list[LanguageProbabilityIn] = Field(default_factory=list, max_length=10)
    alternatives: list[AlternativeIn] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_word(self) -> WordEvidenceIn:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        if self.languages:
            total = sum(item.probability for item in self.languages)
            if abs(total - 1.0) > 1e-4:
                raise ValueError("language probabilities must sum to 1")
        return self


class EntityEvidenceIn(StrictModel):
    start_word: int = Field(ge=0, le=199)
    end_word: int = Field(ge=0, le=199)
    entity_type: EntityType
    surface: ShortText
    criticality: Criticality
    detector_confidence: float = Field(ge=0.0, le=1.0)
    canonical_value: ShortText | None = None
    alternatives: list[AlternativeIn] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_span(self) -> EntityEvidenceIn:
        if self.end_word < self.start_word:
            raise ValueError("end_word must be greater than or equal to start_word")
        return self


class AnalyzeEvidenceRequest(StrictModel):
    transcript_id: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)] | None
    ) = None
    model_version: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
    ] = "evidence-demo-v1"
    words: list[WordEvidenceIn] = Field(min_length=1, max_length=200)
    entities: list[EntityEvidenceIn] = Field(default_factory=list, max_length=50)

    def to_domain(
        self, transcript_id: str
    ) -> tuple[TranscriptEvidence, tuple[EntityEvidence, ...]]:
        words = tuple(
            WordEvidence(
                index=index,
                surface=item.surface,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                decoder_confidence=item.decoder_confidence,
                acoustic_quality=item.acoustic_quality,
                boundary_probability=item.boundary_probability,
                languages=tuple(
                    LanguageProbability(code=language.code, probability=language.probability)
                    for language in item.languages
                ),
                alternatives=tuple(
                    Alternative(surface=alternative.surface, probability=alternative.probability)
                    for alternative in item.alternatives
                ),
            )
            for index, item in enumerate(self.words)
        )
        entities = tuple(
            EntityEvidence(
                start_word=item.start_word,
                end_word=item.end_word,
                entity_type=item.entity_type,
                surface=item.surface,
                criticality=item.criticality,
                detector_confidence=item.detector_confidence,
                canonical_value=item.canonical_value,
                alternatives=tuple(
                    Alternative(surface=alternative.surface, probability=alternative.probability)
                    for alternative in item.alternatives
                ),
            )
            for item in self.entities
        )
        return (
            TranscriptEvidence(
                transcript_id=transcript_id,
                words=words,
                model_version=self.model_version,
            ),
            entities,
        )


class LanguageProbabilityOut(StrictModel):
    code: str
    probability: float


class WordAssessmentOut(StrictModel):
    index: int
    surface: str
    start_ms: int
    end_ms: int
    recognition_risk: float
    boundary_uncertainty: float
    languages: list[LanguageProbabilityOut]


class EntityAssessmentOut(StrictModel):
    start_word: int
    end_word: int
    entity_type: EntityType
    surface: str
    canonical_value: str | None
    criticality: Criticality
    preservation_risk: float


class TranscriptAssessmentOut(StrictModel):
    transcript_id: str
    text: str
    action: str
    reason_codes: list[str]
    calibration_status: str
    model_version: str
    risk_model_version: str
    policy_version: str
    words: list[WordAssessmentOut]
    entities: list[EntityAssessmentOut]

    @classmethod
    def from_domain(cls, transcript: RiskBearingTranscript) -> TranscriptAssessmentOut:
        return cls(
            transcript_id=transcript.transcript_id,
            text=transcript.text,
            action=transcript.action,
            reason_codes=list(transcript.reason_codes),
            calibration_status=transcript.calibration_status,
            model_version=transcript.model_version,
            risk_model_version=transcript.risk_model_version,
            policy_version=transcript.policy_version,
            words=[
                WordAssessmentOut(
                    index=word.evidence.index,
                    surface=word.evidence.surface,
                    start_ms=word.evidence.start_ms,
                    end_ms=word.evidence.end_ms,
                    recognition_risk=word.recognition_risk,
                    boundary_uncertainty=word.boundary_uncertainty,
                    languages=[
                        LanguageProbabilityOut(
                            code=language.code,
                            probability=language.probability,
                        )
                        for language in word.evidence.languages
                    ],
                )
                for word in transcript.words
            ],
            entities=[
                EntityAssessmentOut(
                    start_word=entity.evidence.start_word,
                    end_word=entity.evidence.end_word,
                    entity_type=entity.evidence.entity_type,
                    surface=entity.evidence.surface,
                    canonical_value=entity.evidence.canonical_value,
                    criticality=entity.evidence.criticality,
                    preservation_risk=entity.preservation_risk,
                )
                for entity in transcript.entities
            ],
        )


class HealthOut(StrictModel):
    status: str
    service: str


class JobStatusOut(StrictModel):
    job_id: str
    state: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    attempts: int
    max_attempts: int
    retention_class: Literal["ephemeral"]
    error_code: str | None
    result: TranscriptAssessmentOut | None = None


class ApiErrorOut(StrictModel):
    code: str
    message: str
    request_id: str
