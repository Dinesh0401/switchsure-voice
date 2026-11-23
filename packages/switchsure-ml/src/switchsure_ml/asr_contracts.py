"""Model-neutral contracts for ASR evidence.

The types in this module deliberately call decoder outputs *evidence*, not
confidence.  A separate, held-out calibration stage is required before any
score can be presented as a probability of correctness.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable


class AsrTask(StrEnum):
    """Supported speech generation tasks."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class DecoderScoreKind(StrEnum):
    """Semantics of a raw score emitted by a decoder."""

    PROBABILITY = "probability"
    LOG_PROBABILITY = "log_probability"
    LOGIT = "logit"
    SEQUENCE_SCORE = "sequence_score"


@dataclass(frozen=True, slots=True)
class DecoderScore:
    """A raw decoder feature that has not been calibrated for correctness."""

    value: float
    kind: DecoderScoreKind
    calibrated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("decoder score must be finite")
        if self.kind is DecoderScoreKind.PROBABILITY and not 0.0 <= self.value <= 1.0:
            raise ValueError("decoder probability evidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TimeSpan:
    """Half-open-ish media span in seconds as reported by the backend."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_seconds) or not math.isfinite(self.end_seconds):
            raise ValueError("timestamps must be finite")
        if self.start_seconds < 0.0:
            raise ValueError("timestamp start cannot be negative")
        if self.end_seconds < self.start_seconds:
            raise ValueError("timestamp end cannot precede its start")


@dataclass(frozen=True, slots=True)
class AudioSource:
    """Ephemeral audio supplied as encoded bytes or a local path.

    Audio and paths are excluded from ``repr`` to reduce accidental leakage to
    logs and tracebacks. Remote URLs are intentionally outside this contract.
    """

    data: bytes | None = field(default=None, repr=False)
    path: Path | None = field(default=None, repr=False)
    media_type: str | None = None

    def __post_init__(self) -> None:
        if (self.data is None) == (self.path is None):
            raise ValueError("provide exactly one of audio data or a local path")
        if self.data is not None and not self.data:
            raise ValueError("audio data cannot be empty")
        if self.path is not None and not self.path.is_absolute():
            raise ValueError("audio path must be absolute")
        if self.media_type is not None and not self.media_type.startswith("audio/"):
            raise ValueError("media_type must use the audio/* form")

    def backend_value(self) -> bytes | str:
        """Return a backend-compatible value without copying audio bytes."""

        if self.data is not None:
            return self.data
        assert self.path is not None
        return str(self.path)


@dataclass(frozen=True, slots=True)
class AsrOptions:
    """Portable decoding options shared across ASR implementations."""

    task: AsrTask = AsrTask.TRANSCRIBE
    language_hints: tuple[str, ...] = ()
    word_timestamps: bool = True
    multilingual: bool = True
    vad_filter: bool = True
    beam_size: int = 5
    max_alternatives: int = 3
    hotwords: tuple[str, ...] = ()
    initial_prompt: str | None = None

    def __post_init__(self) -> None:
        if self.beam_size < 1:
            raise ValueError("beam_size must be at least 1")
        if self.max_alternatives < 1:
            raise ValueError("max_alternatives must be at least 1")
        if any(not value.strip() for value in self.language_hints):
            raise ValueError("language hints cannot be blank")
        if any(not value.strip() for value in self.hotwords):
            raise ValueError("hotwords cannot be blank")
        if self.initial_prompt is not None and not self.initial_prompt.strip():
            raise ValueError("initial_prompt cannot be blank")


@dataclass(frozen=True, slots=True)
class AsrRequest:
    """One inference request. ``source_id`` is a non-sensitive corpus key."""

    request_id: str
    source_id: str
    audio: AudioSource
    options: AsrOptions = field(default_factory=AsrOptions)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id cannot be blank")
        if not self.source_id.strip():
            raise ValueError("source_id cannot be blank")


@dataclass(frozen=True, slots=True)
class LanguageEvidence:
    """Backend language evidence for an utterance or time span."""

    language: str
    decoder_score: DecoderScore | None = None
    span: TimeSpan | None = None
    scope: str = "utterance"

    def __post_init__(self) -> None:
        if not self.language.strip():
            raise ValueError("language cannot be blank")
        if not self.scope.strip():
            raise ValueError("language-evidence scope cannot be blank")


@dataclass(frozen=True, slots=True)
class TokenEvidence:
    """A decoder token and its optional raw score and timestamp."""

    surface: str
    token_id: int | None = None
    span: TimeSpan | None = None
    decoder_score: DecoderScore | None = None


@dataclass(frozen=True, slots=True)
class WordEvidence:
    """A surface word with immutable timing and decoder evidence."""

    surface: str
    span: TimeSpan
    decoder_score: DecoderScore | None = None
    language_evidence: tuple[LanguageEvidence, ...] = ()
    tokens: tuple[TokenEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.surface:
            raise ValueError("word surface cannot be empty")


@dataclass(frozen=True, slots=True)
class SegmentEvidence:
    """A backend segment preserving words, tokens, language, and raw scores."""

    surface: str
    span: TimeSpan
    words: tuple[WordEvidence, ...] = ()
    tokens: tuple[TokenEvidence, ...] = ()
    decoder_score: DecoderScore | None = None
    language_evidence: tuple[LanguageEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class TranscriptAlternative:
    """One transcript candidate from N-best or repeated decoding evidence."""

    surface: str
    decoder_score: DecoderScore | None = None
    words: tuple[WordEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.surface:
            raise ValueError("alternative surface cannot be empty")


@dataclass(frozen=True, slots=True)
class AsrProvenance:
    """Pinned model/adapter metadata needed to reproduce an inference."""

    adapter: str
    model_id: str
    model_revision: str | None = None
    backend_version: str | None = None
    device: str | None = None
    decoding_parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.adapter.strip() or not self.model_id.strip():
            raise ValueError("adapter and model_id cannot be blank")
        keys = [key for key, _ in self.decoding_parameters]
        if any(not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("decoding parameter keys must be unique and non-empty")

    @classmethod
    def from_parameters(
        cls,
        *,
        adapter: str,
        model_id: str,
        parameters: Mapping[str, object],
        model_revision: str | None = None,
        backend_version: str | None = None,
        device: str | None = None,
    ) -> AsrProvenance:
        return cls(
            adapter=adapter,
            model_id=model_id,
            model_revision=model_revision,
            backend_version=backend_version,
            device=device,
            decoding_parameters=tuple(
                sorted((key, str(value)) for key, value in parameters.items())
            ),
        )


@dataclass(frozen=True, slots=True)
class AsrEvidence:
    """Complete, uncalibrated recognition evidence emitted by an adapter."""

    transcript: str
    segments: tuple[SegmentEvidence, ...]
    language_evidence: tuple[LanguageEvidence, ...]
    alternatives: tuple[TranscriptAlternative, ...]
    provenance: AsrProvenance
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.transcript:
            raise ValueError("transcript cannot be empty")
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("warnings cannot be blank")


@runtime_checkable
class AsrEvidencePort(Protocol):
    """Inference boundary implemented by every recognition backend."""

    def transcribe(self, request: AsrRequest) -> AsrEvidence:
        """Return immutable decoder evidence for one request."""
