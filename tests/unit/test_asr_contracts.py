from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from switchsure_ml.asr_adapters import AsrAdapterError, FixtureAsrAdapter
from switchsure_ml.asr_contracts import (
    AsrEvidence,
    AsrEvidencePort,
    AsrOptions,
    AsrProvenance,
    AsrRequest,
    AudioSource,
    DecoderScore,
    DecoderScoreKind,
    LanguageEvidence,
    SegmentEvidence,
    TimeSpan,
    TranscriptAlternative,
    WordEvidence,
)


def make_evidence() -> AsrEvidence:
    word = WordEvidence(
        surface=" 500 mg",
        span=TimeSpan(0.2, 0.8),
        decoder_score=DecoderScore(0.74, DecoderScoreKind.PROBABILITY),
    )
    return AsrEvidence(
        transcript="மருந்து 500 mg",
        segments=(
            SegmentEvidence(
                surface="மருந்து 500 mg",
                span=TimeSpan(0.0, 1.0),
                words=(word,),
            ),
        ),
        language_evidence=(
            LanguageEvidence(
                "ta",
                DecoderScore(0.81, DecoderScoreKind.PROBABILITY),
            ),
        ),
        alternatives=(TranscriptAlternative("மருந்து 500 mg", words=(word,)),),
        provenance=AsrProvenance(adapter="fixture", model_id="fixture-v1"),
    )


def make_request() -> AsrRequest:
    return AsrRequest(
        request_id="req-1",
        source_id="speaker-safe-key-1",
        audio=AudioSource(data=b"not-real-audio", media_type="audio/wav"),
    )


def test_audio_source_requires_one_private_local_source() -> None:
    source = AudioSource(data=b"private audio", media_type="audio/wav")

    assert source.backend_value() == b"private audio"
    assert "private audio" not in repr(source)
    assert "data=" not in repr(source)

    with pytest.raises(ValueError, match="exactly one"):
        AudioSource()
    with pytest.raises(ValueError, match="exactly one"):
        AudioSource(data=b"x", path=Path("C:/audio.wav"))
    with pytest.raises(ValueError, match="absolute"):
        AudioSource(path=Path("relative.wav"))
    with pytest.raises(ValueError, match=r"audio/\*"):
        AudioSource(data=b"x", media_type="application/octet-stream")


def test_decoder_scores_are_explicitly_uncalibrated_and_immutable() -> None:
    score = DecoderScore(-1.2, DecoderScoreKind.LOG_PROBABILITY)

    assert score.calibrated is False
    with pytest.raises(FrozenInstanceError):
        score.value = -0.5  # type: ignore[misc]
    with pytest.raises(ValueError, match="between 0 and 1"):
        DecoderScore(1.01, DecoderScoreKind.PROBABILITY)
    with pytest.raises(ValueError, match="finite"):
        DecoderScore(float("nan"), DecoderScoreKind.LOGIT)


def test_typed_contracts_reject_invalid_ranges_and_blank_options() -> None:
    with pytest.raises(ValueError, match="precede"):
        TimeSpan(2.0, 1.0)
    with pytest.raises(ValueError, match="beam_size"):
        AsrOptions(beam_size=0)
    with pytest.raises(ValueError, match="language hints"):
        AsrOptions(language_hints=(" ",))
    with pytest.raises(ValueError, match="source_id"):
        AsrRequest("req", " ", AudioSource(data=b"x"))


def test_provenance_parameters_are_stable_and_reproducible() -> None:
    provenance = AsrProvenance.from_parameters(
        adapter="baseline",
        model_id="openai/whisper-small",
        model_revision="abc123",
        parameters={"task": "transcribe", "beam": 5},
    )

    assert provenance.decoding_parameters == (("beam", "5"), ("task", "transcribe"))
    assert provenance.model_revision == "abc123"


def test_fixture_adapter_is_deterministic_and_satisfies_port() -> None:
    evidence = make_evidence()
    adapter = FixtureAsrAdapter({"speaker-safe-key-1": evidence})

    assert isinstance(adapter, AsrEvidencePort)
    assert adapter.transcribe(make_request()) is evidence
    assert adapter.transcribe(make_request()) is evidence


def test_fixture_adapter_reports_missing_key_without_audio_content() -> None:
    request = make_request()

    with pytest.raises(AsrAdapterError) as error:
        FixtureAsrAdapter({}).transcribe(request)

    assert request.source_id in str(error.value)
    assert "not-real-audio" not in str(error.value)
