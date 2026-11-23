from switchsure_core import (
    Criticality,
    DecisionAction,
    EntityType,
    LanguageProbability,
    TranscriptEvidence,
    WordEvidence,
    analyze_transcript,
    detect_structured_entities,
)


def word(
    index: int,
    surface: str,
    *,
    confidence: float = 0.97,
    boundary: float = 0.02,
    acoustic_quality: float = 0.95,
) -> WordEvidence:
    return WordEvidence(
        index=index,
        surface=surface,
        start_ms=index * 300,
        end_ms=(index + 1) * 300,
        decoder_confidence=confidence,
        acoustic_quality=acoustic_quality,
        boundary_probability=boundary,
        languages=(
            LanguageProbability(code="ta", probability=0.9),
            LanguageProbability(code="en", probability=0.1),
        ),
    )


def test_safe_transcript_is_accepted() -> None:
    evidence = TranscriptEvidence(
        transcript_id="safe-1",
        words=(word(0, "வணக்கம்"), word(1, "friend")),
        model_version="test-model",
    )

    result = analyze_transcript(evidence)

    assert result.action is DecisionAction.ACCEPT
    assert result.calibration_status == "uncalibrated_baseline"
    assert not result.entities


def test_uncertain_dosage_is_not_silently_accepted() -> None:
    evidence = TranscriptEvidence(
        transcript_id="dosage-1",
        words=(
            word(0, "Metformin", confidence=0.82, boundary=0.65),
            word(1, "500", confidence=0.18, boundary=0.48, acoustic_quality=0.55),
            word(2, "mg", confidence=0.40, boundary=0.51, acoustic_quality=0.60),
        ),
        model_version="test-model",
    )

    result = analyze_transcript(evidence)

    assert result.action in {DecisionAction.REVIEW, DecisionAction.DEFER}
    assert len(result.entities) == 1
    assert result.entities[0].evidence.entity_type is EntityType.DOSAGE
    assert result.entities[0].evidence.criticality is Criticality.HIGH
    assert result.entities[0].preservation_risk > 0.5


def test_structured_detector_finds_contact_and_date() -> None:
    words = (
        word(0, "+919876543210"),
        word(1, "12/11/2026"),
        word(2, "09:30am"),
    )

    entities = detect_structured_entities(words)

    assert [entity.entity_type for entity in entities] == [
        EntityType.PHONE,
        EntityType.DATE,
        EntityType.TIME,
    ]


def test_language_probabilities_must_be_normalized() -> None:
    try:
        WordEvidence(
            index=0,
            surface="hello",
            start_ms=0,
            end_ms=100,
            decoder_confidence=0.9,
            languages=(LanguageProbability(code="en", probability=0.7),),
        )
    except ValueError as error:
        assert "sum to 1" in str(error)
    else:
        raise AssertionError("expected invalid language distribution to fail")
