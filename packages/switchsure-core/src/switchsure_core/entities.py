"""Deterministic high-precision detectors for structured critical values."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .domain import Criticality, EntityEvidence, EntityType, WordEvidence

_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_PHONE = re.compile(r"^\+?[0-9][0-9()\-\s]{6,}[0-9]$")
_DATE = re.compile(r"^(?:\d{1,2}[-/.]){2}\d{2,4}$")
_TIME = re.compile(r"^\d{1,2}:\d{2}(?:am|pm)?$", re.IGNORECASE)
_MONEY = re.compile(r"^(?:₹|\$|€|£)\s?\d+(?:[.,]\d+)?$")
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?$")
_DOSAGE_UNIT = re.compile(r"^(?:mg|mcg|g|kg|ml|l|iu)$", re.IGNORECASE)
_IDENTIFIER = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9-]{6,}$")


def detect_structured_entities(words: Sequence[WordEvidence]) -> tuple[EntityEvidence, ...]:
    entities: list[EntityEvidence] = []
    index = 0
    while index < len(words):
        surface = words[index].surface.strip().strip(",;()")

        if index + 1 < len(words) and _NUMBER.fullmatch(surface):
            unit = words[index + 1].surface.strip().strip(",;()")
            if _DOSAGE_UNIT.fullmatch(unit):
                combined = f"{surface} {unit}"
                entities.append(
                    EntityEvidence(
                        start_word=index,
                        end_word=index + 1,
                        entity_type=EntityType.DOSAGE,
                        surface=combined,
                        criticality=Criticality.HIGH,
                        detector_confidence=0.99,
                        canonical_value=combined.lower(),
                    )
                )
                index += 2
                continue

        detectors = (
            (_EMAIL, EntityType.EMAIL, Criticality.HIGH),
            (_PHONE, EntityType.PHONE, Criticality.HIGH),
            (_DATE, EntityType.DATE, Criticality.MEDIUM),
            (_TIME, EntityType.TIME, Criticality.MEDIUM),
            (_MONEY, EntityType.MONEY, Criticality.HIGH),
            (_IDENTIFIER, EntityType.IDENTIFIER, Criticality.HIGH),
        )
        for pattern, entity_type, criticality in detectors:
            if pattern.fullmatch(surface):
                entities.append(
                    EntityEvidence(
                        start_word=index,
                        end_word=index,
                        entity_type=entity_type,
                        surface=surface,
                        criticality=criticality,
                        detector_confidence=0.99,
                        canonical_value=surface,
                    )
                )
                break
        index += 1

    return tuple(entities)
