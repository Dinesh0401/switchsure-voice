"""Versioned, reversible text normalization for recognition metrics."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

NORMALIZATION_VERSION = "switchsure-normalization-v1"


@dataclass(frozen=True, slots=True)
class NormalizedTranscript:
    """Keep the immutable surface beside the normalized evaluation view."""

    surface: str
    normalized: str
    tokens: tuple[str, ...]
    version: str = NORMALIZATION_VERSION


def _metric_character(character: str) -> str:
    if character.isspace():
        return " "
    category = unicodedata.category(character)
    if category.startswith(("P", "S")):
        return " "
    if category.startswith("C"):
        return ""
    return character.casefold()


def normalize_transcript(surface: str) -> NormalizedTranscript:
    """Create the v1 metric view without transliteration or value rewriting.

    V1 applies NFC, Unicode case folding, punctuation/symbol separation, control
    removal, and whitespace collapse. Tamil characters and digits remain in
    their original script. Entity canonicalization is deliberately out of
    scope and must use a typed, reversible domain function.
    """

    if not isinstance(surface, str):
        raise TypeError("surface must be a string")

    nfc_surface = unicodedata.normalize("NFC", surface)
    metric_text = "".join(_metric_character(character) for character in nfc_surface)
    normalized = " ".join(metric_text.split())
    return NormalizedTranscript(
        surface=surface,
        normalized=normalized,
        tokens=tuple(normalized.split()),
    )
