from __future__ import annotations

import importlib
import io
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from switchsure_ml.asr_adapters import (
    AdapterOutputError,
    AsrAdapterError,
    FasterWhisperAdapter,
    HuggingFaceWhisperAdapter,
    MissingOptionalDependencyError,
)
from switchsure_ml.asr_contracts import (
    AsrOptions,
    AsrRequest,
    AudioSource,
    DecoderScoreKind,
)


def make_request() -> AsrRequest:
    return AsrRequest(
        request_id="safe-request-id",
        source_id="safe-source-id",
        audio=AudioSource(data=b"secret-audio", media_type="audio/wav"),
        options=AsrOptions(
            language_hints=("ta",),
            beam_size=7,
            hotwords=("Metformin", "500 mg"),
            initial_prompt="medical visit",
        ),
    )


class RecordingFactory:
    def __init__(self, product: object) -> None:
        self.product = product
        self.calls: list[dict[str, object]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append({"args": args, **kwargs})
        return self.product


class FakeFasterModel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def transcribe(self, audio: object, **kwargs: object) -> tuple[object, object]:
        self.calls.append((audio, kwargs))
        word = SimpleNamespace(word=" 500 mg", start=0.5, end=1.1, probability=0.72)
        segment = SimpleNamespace(
            text=" மருந்து 500 mg",
            start=0.0,
            end=1.2,
            avg_logprob=-0.42,
            words=(word,),
        )
        info = SimpleNamespace(
            language="ta",
            language_probability=0.82,
            all_language_probs=(("ta", 0.82), ("en", 0.18)),
        )
        return iter((segment,)), info


def test_faster_whisper_adapter_is_lazy_and_preserves_evidence() -> None:
    model = FakeFasterModel()
    factory = RecordingFactory(model)
    adapter = FasterWhisperAdapter(
        "small",
        model_revision="model-revision",
        device="cpu",
        compute_type="int8",
        model_factory=factory,
    )

    assert factory.calls == []
    evidence = adapter.transcribe(make_request())

    assert len(factory.calls) == 1
    assert evidence.transcript == "மருந்து 500 mg"
    assert evidence.segments[0].span.end_seconds == 1.2
    assert evidence.segments[0].words[0].surface == " 500 mg"
    word_score = evidence.segments[0].words[0].decoder_score
    assert word_score is not None
    assert word_score.kind is DecoderScoreKind.PROBABILITY
    assert word_score.calibrated is False
    assert [item.language for item in evidence.language_evidence] == ["ta", "en"]
    assert evidence.alternatives[0].surface == evidence.transcript
    assert evidence.provenance.model_revision == "model-revision"
    assert evidence.warnings == ("decoder scores are uncalibrated evidence",)

    audio, kwargs = model.calls[0]
    assert isinstance(audio, io.BytesIO)
    assert audio.getvalue() == b"secret-audio"
    assert kwargs["language"] == "ta"
    assert kwargs["beam_size"] == 7
    assert kwargs["hotwords"] == "Metformin 500 mg"
    assert kwargs["multilingual"] is True
    assert kwargs["vad_filter"] is True


def test_faster_whisper_model_is_loaded_once() -> None:
    factory = RecordingFactory(FakeFasterModel())
    adapter = FasterWhisperAdapter("tiny", model_factory=factory)

    adapter.transcribe(make_request())
    adapter.transcribe(make_request())

    assert len(factory.calls) == 1


def test_default_faster_whisper_loader_refuses_unenforced_revision() -> None:
    with pytest.raises(ValueError, match="cannot enforce model_revision"):
        FasterWhisperAdapter("small", model_revision="claimed-but-not-applied")


def test_optional_faster_whisper_dependency_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> Any:
        assert name == "faster_whisper"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(MissingOptionalDependencyError, match=r"faster-whisper.*uv add"):
        FasterWhisperAdapter("small").transcribe(make_request())


def test_faster_whisper_failure_redacts_audio() -> None:
    class FailingModel:
        def transcribe(self, _audio: object, **_kwargs: object) -> None:
            raise RuntimeError("backend failure")

    adapter = FasterWhisperAdapter("small", model_factory=RecordingFactory(FailingModel()))

    with pytest.raises(AsrAdapterError) as error:
        adapter.transcribe(make_request())

    assert "safe-request-id" in str(error.value)
    assert "secret-audio" not in str(error.value)


def test_huggingface_adapter_is_lazy_and_maps_words_languages_and_nbest() -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def pipeline(audio: object, **kwargs: object) -> dict[str, object]:
        calls.append((audio, kwargs))
        return {
            "text": "மருந்து 500 mg",
            "chunks": [
                {"text": "மருந்து", "timestamp": (0.0, 0.5), "score": 0.91},
                {"text": " 500 mg", "timestamp": (0.5, 1.1), "score": 0.68},
            ],
            "language": "ta",
            "language_score": 0.79,
            "alternatives": [
                {"text": "மருந்து 50 mg", "score": -1.4},
                {"text": "மருந்து 500 mg", "score": -0.5},
            ],
        }

    factory = RecordingFactory(pipeline)
    adapter = HuggingFaceWhisperAdapter(
        "openai/whisper-small",
        model_revision="revision-1",
        device=-1,
        pipeline_factory=factory,
    )

    assert factory.calls == []
    evidence = adapter.transcribe(make_request())

    assert factory.calls[0]["task"] == "automatic-speech-recognition"
    assert factory.calls[0]["revision"] == "revision-1"
    assert evidence.segments[0].span.end_seconds == 1.1
    assert len(evidence.segments[0].words) == 2
    assert evidence.language_evidence[0].language == "ta"
    assert [item.surface for item in evidence.alternatives] == [
        "மருந்து 500 mg",
        "மருந்து 50 mg",
    ]
    assert evidence.alternatives[1].decoder_score is not None
    assert evidence.alternatives[1].decoder_score.calibrated is False
    assert calls[0][1]["return_timestamps"] == "word"
    generate_kwargs = calls[0][1]["generate_kwargs"]
    assert isinstance(generate_kwargs, dict)
    assert generate_kwargs["language"] == "ta"
    assert "VAD is not applied" in evidence.warnings[1]


def test_huggingface_optional_dependency_and_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> Any:
        assert name == "transformers"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(MissingOptionalDependencyError, match="transformers torch"):
        HuggingFaceWhisperAdapter("openai/whisper-small").transcribe(make_request())

    pipeline_factory: Callable[..., object] = RecordingFactory(lambda *_args, **_kwargs: [])
    with pytest.raises(AdapterOutputError, match="non-mapping"):
        HuggingFaceWhisperAdapter(
            "openai/whisper-small", pipeline_factory=pipeline_factory
        ).transcribe(make_request())


def test_huggingface_failure_redacts_audio() -> None:
    def pipeline(_audio: object, **_kwargs: object) -> object:
        raise RuntimeError("backend failure")

    adapter = HuggingFaceWhisperAdapter(
        "openai/whisper-small", pipeline_factory=RecordingFactory(pipeline)
    )

    with pytest.raises(AsrAdapterError) as error:
        adapter.transcribe(make_request())

    assert "safe-request-id" in str(error.value)
    assert "secret-audio" not in str(error.value)
