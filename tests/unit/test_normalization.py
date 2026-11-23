import pytest
from switchsure_eval.normalization import NORMALIZATION_VERSION, normalize_transcript


def test_normalization_preserves_surface_and_tamil_script() -> None:
    surface = "எனக்கு Metformin 500 mg வேணும்"

    result = normalize_transcript(surface)

    assert result.surface == surface
    assert result.normalized == "எனக்கு metformin 500 mg வேணும்"
    assert result.tokens == ("எனக்கு", "metformin", "500", "mg", "வேணும்")
    assert result.version == NORMALIZATION_VERSION


def test_normalization_is_unicode_stable_and_does_not_transliterate() -> None:
    decomposed = "Cafe\N{COMBINING ACUTE ACCENT} தமிழ்"

    result = normalize_transcript(decomposed)

    assert result.normalized == "café தமிழ்"
    assert "தமிழ்" in result.tokens


def test_normalization_separates_punctuation_symbols_and_whitespace() -> None:
    result = normalize_transcript("  Dose: ₹500—TODAY!\nID\u200b42  ")

    assert result.normalized == "dose 500 today id42"
    assert result.tokens == ("dose", "500", "today", "id42")


def test_normalization_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="surface must be a string"):
        normalize_transcript(500)  # type: ignore[arg-type]
