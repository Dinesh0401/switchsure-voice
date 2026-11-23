"""Dependency-light ASR adapters that emit the shared evidence contract."""

from __future__ import annotations

import importlib
import io
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .asr_contracts import (
    AsrEvidence,
    AsrProvenance,
    AsrRequest,
    DecoderScore,
    DecoderScoreKind,
    LanguageEvidence,
    SegmentEvidence,
    TimeSpan,
    TranscriptAlternative,
    WordEvidence,
)


class AsrAdapterError(RuntimeError):
    """Safe adapter failure that never includes audio or transcript content."""


class MissingOptionalDependencyError(AsrAdapterError):
    """Raised when an optional backend has not been installed."""


class AdapterOutputError(AsrAdapterError):
    """Raised when a backend returns an unusable response shape."""


class FixtureAsrAdapter:
    """Deterministic adapter for tests and reproducible interface demos."""

    def __init__(self, fixtures: Mapping[str, AsrEvidence]) -> None:
        self._fixtures = dict(fixtures)

    def transcribe(self, request: AsrRequest) -> AsrEvidence:
        try:
            return self._fixtures[request.source_id]
        except KeyError as error:
            raise AsrAdapterError(
                f"no deterministic ASR fixture registered for source_id={request.source_id!r}"
            ) from error


class FasterWhisperAdapter:
    """Optional faster-whisper baseline with lazy dependency and model loading."""

    def __init__(
        self,
        model_id: str,
        *,
        model_revision: str | None = None,
        device: str = "auto",
        compute_type: str = "default",
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be blank")
        if model_revision is not None and model_factory is None:
            raise ValueError(
                "the default faster-whisper loader cannot enforce model_revision; "
                "use an immutable local model path or inject a revision-aware model_factory"
            )
        self._model_id = model_id
        self._model_revision = model_revision
        self._device = device
        self._compute_type = compute_type
        self._model_factory = model_factory
        self._model: Any | None = None
        self._backend_version: str | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        factory = self._model_factory
        if factory is None:
            try:
                backend = importlib.import_module("faster_whisper")
            except ModuleNotFoundError as error:
                raise MissingOptionalDependencyError(
                    "faster-whisper is optional; install the SwitchSure ASR baseline "
                    "environment with `uv add --optional asr faster-whisper`"
                ) from error
            factory = backend.WhisperModel
            self._backend_version = getattr(backend, "__version__", None)
        self._model = factory(
            self._model_id,
            device=self._device,
            compute_type=self._compute_type,
        )
        return self._model

    def transcribe(self, request: AsrRequest) -> AsrEvidence:
        model = self._load_model()
        language = (
            request.options.language_hints[0] if len(request.options.language_hints) == 1 else None
        )
        kwargs: dict[str, object] = {
            "beam_size": request.options.beam_size,
            "multilingual": request.options.multilingual,
            "vad_filter": request.options.vad_filter,
            "word_timestamps": request.options.word_timestamps,
            "task": request.options.task.value,
        }
        if language is not None:
            kwargs["language"] = language
        if request.options.hotwords:
            kwargs["hotwords"] = " ".join(request.options.hotwords)
        if request.options.initial_prompt is not None:
            kwargs["initial_prompt"] = request.options.initial_prompt

        try:
            audio: object = request.audio.backend_value()
            if isinstance(audio, bytes):
                audio = io.BytesIO(audio)
            raw_segments, info = model.transcribe(audio, **kwargs)
            segments = tuple(self._convert_segment(segment) for segment in raw_segments)
        except AsrAdapterError:
            raise
        except Exception as error:
            raise AsrAdapterError(
                f"faster-whisper inference failed for request_id={request.request_id!r}; "
                "audio and transcript content were omitted"
            ) from error
        transcript = "".join(segment.surface for segment in segments).strip()
        if not transcript:
            raise AdapterOutputError("faster-whisper returned no transcript text")

        languages = self._convert_languages(info)
        alternatives = self._top_alternative(transcript, segments)
        provenance = AsrProvenance.from_parameters(
            adapter="faster-whisper",
            model_id=self._model_id,
            model_revision=self._model_revision,
            backend_version=self._backend_version,
            device=self._device,
            parameters=kwargs,
        )
        return AsrEvidence(
            transcript=transcript,
            segments=segments,
            language_evidence=languages,
            alternatives=alternatives,
            provenance=provenance,
            warnings=("decoder scores are uncalibrated evidence",),
        )

    @staticmethod
    def _convert_segment(segment: Any) -> SegmentEvidence:
        span = TimeSpan(float(segment.start), float(segment.end))
        words = tuple(
            WordEvidence(
                surface=str(word.word),
                span=TimeSpan(float(word.start), float(word.end)),
                decoder_score=DecoderScore(float(word.probability), DecoderScoreKind.PROBABILITY)
                if getattr(word, "probability", None) is not None
                else None,
            )
            for word in (getattr(segment, "words", None) or ())
        )
        average_log_probability = getattr(segment, "avg_logprob", None)
        return SegmentEvidence(
            surface=str(segment.text),
            span=span,
            words=words,
            decoder_score=DecoderScore(
                float(average_log_probability), DecoderScoreKind.LOG_PROBABILITY
            )
            if average_log_probability is not None
            else None,
        )

    @staticmethod
    def _convert_languages(info: Any) -> tuple[LanguageEvidence, ...]:
        all_probabilities = getattr(info, "all_language_probs", None)
        if all_probabilities:
            return tuple(
                LanguageEvidence(
                    language=str(language),
                    decoder_score=DecoderScore(float(score), DecoderScoreKind.PROBABILITY),
                )
                for language, score in all_probabilities
            )
        language = getattr(info, "language", None)
        if language is None:
            return ()
        probability = getattr(info, "language_probability", None)
        return (
            LanguageEvidence(
                language=str(language),
                decoder_score=DecoderScore(float(probability), DecoderScoreKind.PROBABILITY)
                if probability is not None
                else None,
            ),
        )

    @staticmethod
    def _top_alternative(
        transcript: str, segments: tuple[SegmentEvidence, ...]
    ) -> tuple[TranscriptAlternative, ...]:
        scored = [
            segment.decoder_score.value
            for segment in segments
            if segment.decoder_score is not None
            and segment.decoder_score.kind is DecoderScoreKind.LOG_PROBABILITY
        ]
        score = (
            DecoderScore(sum(scored) / len(scored), DecoderScoreKind.SEQUENCE_SCORE)
            if scored
            else None
        )
        return (
            TranscriptAlternative(
                surface=transcript,
                decoder_score=score,
                words=tuple(word for segment in segments for word in segment.words),
            ),
        )


class HuggingFaceWhisperAdapter:
    """Optional Transformers Whisper pipeline with lazy dependency loading."""

    def __init__(
        self,
        model_id: str,
        *,
        model_revision: str | None = None,
        device: str | int | None = None,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be blank")
        self._model_id = model_id
        self._model_revision = model_revision
        self._device = device
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any | None = None
        self._backend_version: str | None = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        factory = self._pipeline_factory
        if factory is None:
            try:
                backend = importlib.import_module("transformers")
            except ModuleNotFoundError as error:
                raise MissingOptionalDependencyError(
                    "Hugging Face Whisper is optional; install the SwitchSure ASR baseline "
                    "environment with `uv add --optional asr transformers torch`"
                ) from error
            factory = backend.pipeline
            self._backend_version = getattr(backend, "__version__", None)
        kwargs: dict[str, object] = {
            "task": "automatic-speech-recognition",
            "model": self._model_id,
        }
        if self._model_revision is not None:
            kwargs["revision"] = self._model_revision
        if self._device is not None:
            kwargs["device"] = self._device
        self._pipeline = factory(**kwargs)
        return self._pipeline

    def transcribe(self, request: AsrRequest) -> AsrEvidence:
        pipeline = self._load_pipeline()
        generate_kwargs: dict[str, object] = {"task": request.options.task.value}
        if len(request.options.language_hints) == 1:
            generate_kwargs["language"] = request.options.language_hints[0]
        if request.options.initial_prompt is not None:
            generate_kwargs["prompt"] = request.options.initial_prompt
        try:
            raw = pipeline(
                request.audio.backend_value(),
                return_timestamps="word" if request.options.word_timestamps else False,
                generate_kwargs=generate_kwargs,
            )
        except Exception as error:
            raise AsrAdapterError(
                f"Hugging Face Whisper inference failed for request_id={request.request_id!r}; "
                "audio and transcript content were omitted"
            ) from error
        if not isinstance(raw, Mapping):
            raise AdapterOutputError("Hugging Face Whisper returned a non-mapping response")
        transcript = str(raw.get("text", "")).strip()
        if not transcript:
            raise AdapterOutputError("Hugging Face Whisper returned no transcript text")

        words = self._convert_chunks(raw.get("chunks"))
        span = self._span_for_words(words)
        segments = (SegmentEvidence(surface=transcript, span=span, words=words),)
        languages = self._language_evidence(raw)
        alternatives = self._alternatives(raw, transcript, words)
        provenance = AsrProvenance.from_parameters(
            adapter="huggingface-whisper",
            model_id=self._model_id,
            model_revision=self._model_revision,
            backend_version=self._backend_version,
            device=str(self._device) if self._device is not None else None,
            parameters={
                "return_timestamps": request.options.word_timestamps,
                **generate_kwargs,
            },
        )
        return AsrEvidence(
            transcript=transcript,
            segments=segments,
            language_evidence=languages,
            alternatives=alternatives,
            provenance=provenance,
            warnings=tuple(
                warning
                for warning in (
                    "decoder scores are uncalibrated evidence",
                    "VAD is not applied by the Hugging Face Whisper adapter"
                    if request.options.vad_filter
                    else None,
                )
                if warning is not None
            ),
        )

    @staticmethod
    def _convert_chunks(value: object) -> tuple[WordEvidence, ...]:
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
            return ()
        words: list[WordEvidence] = []
        for chunk in value:
            if not isinstance(chunk, Mapping):
                continue
            timestamp = chunk.get("timestamp")
            if (
                not isinstance(timestamp, (tuple, list))
                or len(timestamp) != 2
                or timestamp[0] is None
                or timestamp[1] is None
            ):
                continue
            score = chunk.get("score")
            words.append(
                WordEvidence(
                    surface=str(chunk.get("text", "")),
                    span=TimeSpan(float(str(timestamp[0])), float(str(timestamp[1]))),
                    decoder_score=DecoderScore(float(score), DecoderScoreKind.PROBABILITY)
                    if score is not None
                    else None,
                )
            )
        return tuple(words)

    @staticmethod
    def _span_for_words(words: tuple[WordEvidence, ...]) -> TimeSpan:
        if not words:
            return TimeSpan(0.0, 0.0)
        return TimeSpan(words[0].span.start_seconds, words[-1].span.end_seconds)

    @staticmethod
    def _language_evidence(raw: Mapping[object, object]) -> tuple[LanguageEvidence, ...]:
        language = raw.get("language")
        if language is None:
            return ()
        language_score = raw.get("language_score")
        return (
            LanguageEvidence(
                language=str(language),
                decoder_score=DecoderScore(float(str(language_score)), DecoderScoreKind.PROBABILITY)
                if language_score is not None
                else None,
            ),
        )

    @staticmethod
    def _alternatives(
        raw: Mapping[object, object],
        transcript: str,
        words: tuple[WordEvidence, ...],
    ) -> tuple[TranscriptAlternative, ...]:
        candidates: list[TranscriptAlternative] = [
            TranscriptAlternative(surface=transcript, words=words)
        ]
        value = raw.get("alternatives")
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
            for alternative in value:
                if not isinstance(alternative, Mapping):
                    continue
                text = str(alternative.get("text", "")).strip()
                if not text or text == transcript:
                    continue
                score = alternative.get("score")
                candidates.append(
                    TranscriptAlternative(
                        surface=text,
                        decoder_score=DecoderScore(float(score), DecoderScoreKind.SEQUENCE_SCORE)
                        if score is not None
                        else None,
                    )
                )
        return tuple(candidates)
