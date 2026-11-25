"""Deterministic critical-entity preservation over ASR alternatives.

This module deliberately separates entity *surface text* from canonical values.
Decoder weights and the support values derived from them are uncalibrated
evidence: they must never be presented as probabilities of correctness.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import IntEnum, StrEnum
from itertools import combinations


class EntityKind(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    DATE = "date"
    TIME = "time"
    MONEY = "money"
    DOSAGE = "dosage"
    QUANTITY = "quantity"
    IDENTIFIER = "identifier"


class EntitySeverity(IntEnum):
    MEDIUM = 2
    HIGH = 4


class EvidenceOrigin(StrEnum):
    DETERMINISTIC_RECOGNIZER = "deterministic_recognizer"


class EdgeRelation(StrEnum):
    COMPETING_VALUE = "competing_value"
    COMPETING_ENTITY_TYPE = "competing_entity_type"
    COEXISTING_ENTITY = "coexisting_entity"


class EntityRoute(StrEnum):
    PRESERVE_SURFACE = "preserve_surface"
    REVIEW = "review"


class ReviewReason(StrEnum):
    COMPETING_VALUES = "competing_values"
    COMPETING_ENTITY_TYPES = "competing_entity_types"
    MISSING_ENTITY_ALTERNATIVE = "missing_entity_alternative"
    NO_POSITIVE_SUPPORT = "no_positive_support"
    INVALID_CANONICAL_VALUE = "invalid_canonical_value"
    MISSING_PINNED_PROVENANCE = "missing_pinned_provenance"


_CRITICALITY: dict[EntityKind, EntitySeverity] = {
    EntityKind.EMAIL: EntitySeverity.HIGH,
    EntityKind.PHONE: EntitySeverity.HIGH,
    EntityKind.DATE: EntitySeverity.MEDIUM,
    EntityKind.TIME: EntitySeverity.MEDIUM,
    EntityKind.MONEY: EntitySeverity.HIGH,
    EntityKind.DOSAGE: EntitySeverity.HIGH,
    EntityKind.QUANTITY: EntitySeverity.MEDIUM,
    EntityKind.IDENTIFIER: EntitySeverity.HIGH,
}

_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_PHONE = re.compile(r"^\+?[0-9\u0be6-\u0bef][0-9\u0be6-\u0bef()\-\s]{6,}[0-9\u0be6-\u0bef]$")
_DATE = re.compile(r"^(\d{1,4})([-/.])(\d{1,2})\2(\d{1,4})$")
_TIME = re.compile(r"^(\d{1,2}):(\d{2})(?:\s?(am|pm))?$", re.IGNORECASE)
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)$")
_MONEY_JOINED = re.compile(r"^(₹|\$|€|£)\s?([+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+))$")
_IDENTIFIER = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9-]{6,}$")
_BOUNDARY_PUNCTUATION = " \t\r\n,;!?()[]{}\"'"
_IMMUTABLE_MODEL_REVISION = re.compile(
    r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64}|sha256:[0-9a-fA-F]{64})$"
)

_DOSAGE_UNITS = {
    "mg": "mg",
    "mcg": "mcg",
    "g": "g",
    "ml": "ml",
    "iu": "IU",
    "மில்லிகிராம்": "mg",
    "மைக்ரோகிராம்": "mcg",
    "கிராம்": "g",
    "மில்லிலிட்டர்": "ml",
}
_QUANTITY_UNITS = {
    "kg": "kg",
    "l": "L",
    "cm": "cm",
    "mm": "mm",
    "km": "km",
    "கிலோ": "kg",
    "லிட்டர்": "L",
}
_CURRENCY_CODES = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP"}


def _stable_id(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts).encode()
    return f"{prefix}-{hashlib.sha256(material).hexdigest()[:16]}"


def _validate_finite_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EntityWord:
    """One time-aligned word. Surface text is intentionally hidden from repr."""

    index: int
    surface: str = field(repr=False)
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("word index cannot be negative")
        if not self.surface:
            raise ValueError("word surface cannot be empty")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("word timestamps are invalid")


@dataclass(frozen=True, slots=True)
class HypothesisProvenance:
    """Pinned origin for one ASR hypothesis."""

    adapter: str
    model_id: str
    model_revision: str | None
    decoding_id: str

    def __post_init__(self) -> None:
        if not self.adapter.strip() or not self.model_id.strip() or not self.decoding_id.strip():
            raise ValueError("adapter, model_id, and decoding_id cannot be blank")
        if self.model_revision is not None and not self.model_revision.strip():
            raise ValueError("model_revision cannot be blank")

    @property
    def is_pinned(self) -> bool:
        return self.model_revision is not None and bool(
            _IMMUTABLE_MODEL_REVISION.fullmatch(self.model_revision.strip())
        )


@dataclass(frozen=True, slots=True)
class EntityHypothesis:
    """A time-aligned ASR alternative with uncalibrated relative weight."""

    hypothesis_id: str
    words: tuple[EntityWord, ...] = field(repr=False)
    provenance: HypothesisProvenance
    relative_weight: float = 1.0
    calibrated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id cannot be blank")
        if not self.words:
            raise ValueError("hypothesis must contain words")
        if [word.index for word in self.words] != list(range(len(self.words))):
            raise ValueError("word indexes must be contiguous and start at zero")
        if any(
            current.start_ms < previous.start_ms or current.end_ms < previous.end_ms
            for previous, current in zip(self.words, self.words[1:], strict=False)
        ):
            raise ValueError("word timestamps must be monotonic")
        _validate_finite_non_negative("relative_weight", self.relative_weight)


@dataclass(frozen=True, slots=True)
class CanonicalizationStep:
    """One reversible transformation in a typed canonicalization trace."""

    operation: str
    before: str = field(repr=False)
    after: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.operation.strip():
            raise ValueError("canonicalization operation cannot be blank")


@dataclass(frozen=True, slots=True)
class CanonicalEntity:
    """Canonical view that never replaces the immutable captured surface."""

    entity_kind: EntityKind
    surface: str = field(repr=False)
    canonical_value: str | None = field(repr=False)
    locale: str
    method: str
    method_version: str
    steps: tuple[CanonicalizationStep, ...] = field(default_factory=tuple, repr=False)
    validation_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.surface:
            raise ValueError("canonicalization surface cannot be empty")
        if not self.locale.strip() or not self.method.strip() or not self.method_version.strip():
            raise ValueError("locale, method, and method_version cannot be blank")
        current = self.surface
        for step in self.steps:
            if step.before != current:
                raise ValueError("canonicalization steps must form a reversible chain")
            current = step.after
        if self.canonical_value is not None and current != self.canonical_value:
            raise ValueError("canonicalization chain must end at canonical_value")
        if self.canonical_value is None and not self.validation_errors:
            raise ValueError("invalid canonicalization must explain its validation error")
        if self.canonical_value is not None and self.validation_errors:
            raise ValueError("valid canonicalization cannot contain validation errors")

    @property
    def is_valid(self) -> bool:
        return self.canonical_value is not None

    def restore_surface(self) -> str:
        """Reverse the trace and return the exact original surface."""

        current = self.canonical_value
        if current is None:
            return self.surface
        for step in reversed(self.steps):
            if step.after != current:
                raise RuntimeError("canonicalization trace is no longer reversible")
            current = step.before
        return current


class _CanonicalBuilder:
    def __init__(self, surface: str) -> None:
        self.surface = surface
        self.current = surface
        self.steps: list[CanonicalizationStep] = []

    def apply(self, operation: str, after: str) -> None:
        if after != self.current:
            self.steps.append(CanonicalizationStep(operation, self.current, after))
            self.current = after


def _ascii_digits(value: str) -> str:
    converted: list[str] = []
    for character in value:
        try:
            converted.append(str(unicodedata.decimal(character)))
        except (TypeError, ValueError):
            converted.append(character)
    return "".join(converted)


def _decimal_text(value: str) -> str:
    normalized = value.replace(",", ".")
    try:
        number = Decimal(normalized)
    except InvalidOperation as error:
        raise ValueError("invalid decimal value") from error
    rendered = format(number.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def canonicalize_entity(
    surface: str,
    entity_kind: EntityKind,
    *,
    locale: str = "en-IN",
) -> CanonicalEntity:
    """Return a reversible typed view without mutating ``surface``.

    ``en-IN`` uses day/month/year for slash, dash, and dot dates. Other
    locales must use ISO ``YYYY-MM-DD`` to avoid silently resolving an
    ambiguous critical date.
    """

    if not surface:
        raise ValueError("surface cannot be empty")
    if not locale.strip():
        raise ValueError("locale cannot be blank")

    builder = _CanonicalBuilder(surface)
    builder.apply("trim_boundary_punctuation", builder.current.strip(_BOUNDARY_PUNCTUATION))
    builder.apply("unicode_nfc", unicodedata.normalize("NFC", builder.current))
    builder.apply("map_unicode_decimal_digits", _ascii_digits(builder.current))
    value = builder.current
    error: str | None = None

    try:
        if entity_kind is EntityKind.EMAIL:
            if not _EMAIL.fullmatch(value):
                raise ValueError("invalid email structure")
            local, domain = value.rsplit("@", 1)
            builder.apply("lowercase_email_domain", f"{local}@{domain.casefold()}")
        elif entity_kind is EntityKind.PHONE:
            prefix = "+" if value.startswith("+") else ""
            digits = "".join(character for character in value if character.isdigit())
            if not 7 <= len(digits) <= 15:
                raise ValueError("phone must contain 7 to 15 digits")
            builder.apply("remove_phone_separators", prefix + digits)
        elif entity_kind is EntityKind.DATE:
            match = _DATE.fullmatch(value)
            if match is None:
                raise ValueError("invalid numeric date structure")
            first, _, second, third = match.groups()
            if len(first) == 4:
                year, month, day = int(first), int(second), int(third)
            elif locale.casefold() in {"en-in", "ta-in"}:
                day, month, year = int(first), int(second), int(third)
                year = year + 2000 if year < 100 else year
            else:
                raise ValueError("ambiguous non-ISO date requires an explicit supported locale")
            canonical_date = date(year, month, day).isoformat()
            builder.apply("date_to_iso8601", canonical_date)
        elif entity_kind is EntityKind.TIME:
            match = _TIME.fullmatch(value)
            if match is None:
                raise ValueError("invalid time structure")
            hour, minute = int(match.group(1)), int(match.group(2))
            meridiem = (match.group(3) or "").casefold()
            if minute > 59 or hour > (12 if meridiem else 23) or (hour == 0 and meridiem):
                raise ValueError("time is outside its valid range")
            if meridiem == "pm" and hour != 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
            builder.apply("time_to_24_hour", f"{hour:02d}:{minute:02d}")
        elif entity_kind is EntityKind.MONEY:
            match = _MONEY_JOINED.fullmatch(value)
            if match is None:
                pieces = value.split()
                if len(pieces) != 2 or pieces[0] not in _CURRENCY_CODES:
                    raise ValueError("invalid money structure")
                symbol, amount = pieces
            else:
                symbol, amount = match.groups()
            builder.apply(
                "money_to_iso_currency", f"{_CURRENCY_CODES[symbol]} {_decimal_text(amount)}"
            )
        elif entity_kind in {EntityKind.DOSAGE, EntityKind.QUANTITY}:
            pieces = value.split()
            if len(pieces) != 2 or not _NUMBER.fullmatch(pieces[0]):
                raise ValueError("quantity requires a number and unit")
            units = _DOSAGE_UNITS if entity_kind is EntityKind.DOSAGE else _QUANTITY_UNITS
            unit = units.get(pieces[1].casefold())
            if unit is None:
                raise ValueError("unit is not in the deterministic allowlist")
            builder.apply("quantity_to_decimal_and_si_unit", f"{_decimal_text(pieces[0])} {unit}")
        elif entity_kind is EntityKind.IDENTIFIER:
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError("identifier must mix ASCII letters and digits")
            # Identifier case can be semantically meaningful. Validation is safe;
            # case-folding would merge distinct critical values.
        else:
            raise ValueError(f"unsupported entity kind: {entity_kind}")
    except (ValueError, KeyError) as caught:
        error = str(caught)

    return CanonicalEntity(
        entity_kind=entity_kind,
        surface=surface,
        canonical_value=None if error else builder.current,
        locale=locale,
        method="switchsure-deterministic-canonicalizer",
        method_version="1.0.0",
        steps=tuple(builder.steps),
        validation_errors=() if error is None else (error,),
    )


@dataclass(frozen=True, slots=True)
class EntityMention:
    mention_id: str
    entity_kind: EntityKind
    surface: str = field(repr=False)
    canonical: CanonicalEntity = field(repr=False)
    start_word: int
    end_word: int
    start_ms: int
    end_ms: int
    severity: EntitySeverity
    hypothesis_id: str
    provenance: HypothesisProvenance
    origin: EvidenceOrigin = EvidenceOrigin.DETERMINISTIC_RECOGNIZER
    detector_id: str = "switchsure-structured-entities"
    detector_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.mention_id.strip() or not self.hypothesis_id.strip():
            raise ValueError("mention_id and hypothesis_id cannot be blank")
        if self.start_word < 0 or self.end_word < self.start_word:
            raise ValueError("entity word span is invalid")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("entity timestamps are invalid")
        if (
            self.canonical.surface != self.surface
            or self.canonical.entity_kind is not self.entity_kind
        ):
            raise ValueError(
                "canonicalization must describe the immutable mention surface and type"
            )


def _clean(surface: str) -> str:
    return surface.strip(_BOUNDARY_PUNCTUATION)


def recognize_structured_entities(
    hypothesis: EntityHypothesis,
    *,
    locale: str = "en-IN",
) -> tuple[EntityMention, ...]:
    """Run deterministic, dependency-free recognizers over one hypothesis."""

    mentions: list[EntityMention] = []

    def add(kind: EntityKind, start: int, end: int) -> None:
        selected = hypothesis.words[start : end + 1]
        surface = " ".join(word.surface for word in selected)
        canonical = canonicalize_entity(surface, kind, locale=locale)
        mentions.append(
            EntityMention(
                mention_id=_stable_id(
                    "mention", hypothesis.hypothesis_id, kind, start, end, surface
                ),
                entity_kind=kind,
                surface=surface,
                canonical=canonical,
                start_word=start,
                end_word=end,
                start_ms=selected[0].start_ms,
                end_ms=selected[-1].end_ms,
                severity=_CRITICALITY[kind],
                hypothesis_id=hypothesis.hypothesis_id,
                provenance=hypothesis.provenance,
            )
        )

    index = 0
    while index < len(hypothesis.words):
        current = _clean(hypothesis.words[index].surface)
        ascii_current = _ascii_digits(current)
        next_surface = (
            _clean(hypothesis.words[index + 1].surface)
            if index + 1 < len(hypothesis.words)
            else None
        )

        if next_surface is not None and _NUMBER.fullmatch(ascii_current):
            unit = next_surface.casefold()
            if unit in _DOSAGE_UNITS:
                add(EntityKind.DOSAGE, index, index + 1)
                index += 2
                continue
            if unit in _QUANTITY_UNITS:
                add(EntityKind.QUANTITY, index, index + 1)
                index += 2
                continue

        matched_kind: EntityKind | None = None
        if _EMAIL.fullmatch(current):
            matched_kind = EntityKind.EMAIL
        elif _PHONE.fullmatch(current):
            matched_kind = EntityKind.PHONE
        elif _DATE.fullmatch(ascii_current):
            matched_kind = EntityKind.DATE
        elif _TIME.fullmatch(ascii_current):
            matched_kind = EntityKind.TIME
        elif _MONEY_JOINED.fullmatch(ascii_current):
            matched_kind = EntityKind.MONEY
        elif _IDENTIFIER.fullmatch(ascii_current):
            matched_kind = EntityKind.IDENTIFIER
        if matched_kind is not None:
            add(matched_kind, index, index)
        index += 1

    return tuple(mentions)


@dataclass(frozen=True, slots=True)
class EntityCandidate:
    candidate_id: str
    entity_kind: EntityKind
    canonical_value: str | None = field(repr=False)
    representative_surface: str = field(repr=False)
    mentions: tuple[EntityMention, ...] = field(repr=False)
    start_ms: int
    end_ms: int
    relative_support: float
    normalized_support: float
    calibrated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.mentions:
            raise ValueError("candidate requires an id and mentions")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("candidate timestamps are invalid")
        _validate_finite_non_negative("relative_support", self.relative_support)
        if not 0.0 <= self.normalized_support <= 1.0:
            raise ValueError("normalized support must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class LatticeEdge:
    source_candidate_id: str
    target_candidate_id: str
    relation: EdgeRelation
    temporal_iou: float
    surface_similarity: float
    compatibility_weight: float
    calibrated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.source_candidate_id == self.target_candidate_id:
            raise ValueError("lattice edge cannot be a self-loop")
        for name, value in (
            ("temporal_iou", self.temporal_iou),
            ("surface_similarity", self.surface_similarity),
            ("compatibility_weight", self.compatibility_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class EntityDecision:
    decision_id: str
    entity_kind: EntityKind
    start_ms: int
    end_ms: int
    route: EntityRoute
    selected_candidate_id: str | None
    candidate_ids: tuple[str, ...]
    reason_codes: tuple[ReviewReason, ...]

    def __post_init__(self) -> None:
        if not self.decision_id.strip() or not self.candidate_ids:
            raise ValueError("decision requires an id and candidate ids")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("decision candidate ids must be unique")
        if self.route is EntityRoute.PRESERVE_SURFACE:
            if self.selected_candidate_id is None or self.reason_codes:
                raise ValueError("preserve decision requires one selection and no review reason")
            if self.selected_candidate_id not in self.candidate_ids:
                raise ValueError("selected candidate must be exposed in candidate ids")
        elif self.selected_candidate_id is not None or not self.reason_codes:
            raise ValueError("review decision must expose alternatives and reason codes")


@dataclass(frozen=True, slots=True)
class EntityLattice:
    candidates: tuple[EntityCandidate, ...]
    edges: tuple[LatticeEdge, ...]
    decisions: tuple[EntityDecision, ...]
    recognizer_version: str = "1.0.0"


def _interval_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    if union == 0:
        return 1.0 if left == right else 0.0
    return intersection / union


def _surface_similarity(left: str, right: str) -> float:
    left_chars = set(unicodedata.normalize("NFC", left).casefold())
    right_chars = set(unicodedata.normalize("NFC", right).casefold())
    union = left_chars | right_chars
    return len(left_chars & right_chars) / len(union) if union else 1.0


def _cluster_mentions(mentions: Sequence[EntityMention]) -> tuple[tuple[EntityMention, ...], ...]:
    """Cluster alternative mentions of the same type by time overlap."""

    clusters: list[list[EntityMention]] = []
    for mention in sorted(
        mentions, key=lambda item: (item.entity_kind, item.start_ms, item.end_ms)
    ):
        compatible = [
            cluster
            for cluster in clusters
            if cluster[0].entity_kind is mention.entity_kind
            and any(
                _interval_iou((mention.start_ms, mention.end_ms), (item.start_ms, item.end_ms))
                >= 0.5
                for item in cluster
            )
        ]
        if compatible:
            compatible[0].append(mention)
        else:
            clusters.append([mention])
    return tuple(tuple(cluster) for cluster in clusters)


def _hypothesis_covers_interval(hypothesis: EntityHypothesis, start_ms: int, end_ms: int) -> bool:
    """Return whether an alternative supplies evidence over an entity interval.

    N-best alternatives normally span the same utterance. An alternative that
    covers an interval but emits no matching entity is deletion evidence and
    must not be silently ignored by the entity policy.
    """

    return hypothesis.words[0].start_ms <= start_ms and hypothesis.words[-1].end_ms >= end_ms


def build_entity_lattice(
    hypotheses: Iterable[EntityHypothesis],
    *,
    locale: str = "en-IN",
) -> EntityLattice:
    """Aggregate structured candidates and conservatively route ambiguity.

    Distinct overlapping critical values are always returned for review. No
    score threshold is allowed to silently rewrite or discard an alternative.
    """

    items = tuple(hypotheses)
    if not items:
        raise ValueError("at least one hypothesis is required")
    ids = [item.hypothesis_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("hypothesis ids must be unique")

    weights = {item.hypothesis_id: item.relative_weight for item in items}
    mentions = tuple(
        mention
        for hypothesis in items
        for mention in recognize_structured_entities(hypothesis, locale=locale)
    )
    all_candidates: list[EntityCandidate] = []
    all_edges: list[LatticeEdge] = []
    decisions: list[EntityDecision] = []

    for cluster in _cluster_mentions(mentions):
        cluster_start_ms = min(item.start_ms for item in cluster)
        cluster_end_ms = max(item.end_ms for item in cluster)
        by_value: dict[tuple[EntityKind, str], list[EntityMention]] = defaultdict(list)
        for mention in cluster:
            key_value = mention.canonical.canonical_value
            if key_value is None:
                key_value = f"invalid:{mention.surface}"
            by_value[(mention.entity_kind, key_value)].append(mention)

        supports = {
            key: sum(weights[mention.hypothesis_id] for mention in value)
            for key, value in by_value.items()
        }
        total_support = sum(supports.values())
        cluster_candidates: list[EntityCandidate] = []
        for (kind, key_value), grouped_mentions in sorted(
            by_value.items(), key=lambda item: item[0][1]
        ):
            canonical_value = grouped_mentions[0].canonical.canonical_value
            support = supports[(kind, key_value)]
            normalized = support / total_support if total_support > 0.0 else 0.0
            candidate = EntityCandidate(
                candidate_id=_stable_id(
                    "candidate",
                    kind,
                    cluster_start_ms,
                    cluster_end_ms,
                    key_value,
                ),
                entity_kind=kind,
                canonical_value=canonical_value,
                representative_surface=grouped_mentions[0].surface,
                mentions=tuple(grouped_mentions),
                start_ms=min(item.start_ms for item in grouped_mentions),
                end_ms=max(item.end_ms for item in grouped_mentions),
                relative_support=support,
                normalized_support=normalized,
            )
            cluster_candidates.append(candidate)
            all_candidates.append(candidate)

        for left, right in combinations(cluster_candidates, 2):
            temporal = _interval_iou((left.start_ms, left.end_ms), (right.start_ms, right.end_ms))
            similarity = _surface_similarity(
                left.representative_surface, right.representative_surface
            )
            relation = (
                EdgeRelation.COMPETING_VALUE
                if left.entity_kind is right.entity_kind
                else EdgeRelation.COEXISTING_ENTITY
            )
            all_edges.append(
                LatticeEdge(
                    source_candidate_id=left.candidate_id,
                    target_candidate_id=right.candidate_id,
                    relation=relation,
                    temporal_iou=temporal,
                    surface_similarity=similarity,
                    compatibility_weight=(temporal + similarity) / 2.0,
                )
            )

        ordered = tuple(
            candidate.candidate_id
            for candidate in sorted(
                cluster_candidates,
                key=lambda item: (-item.relative_support, item.candidate_id),
            )
        )
        invalid = any(candidate.canonical_value is None for candidate in cluster_candidates)
        unpinned = any(not mention.provenance.is_pinned for mention in cluster)
        mentioning_hypotheses = {mention.hypothesis_id for mention in cluster}
        missing_entity_alternative = any(
            item.hypothesis_id not in mentioning_hypotheses
            and _hypothesis_covers_interval(item, cluster_start_ms, cluster_end_ms)
            for item in items
        )
        reasons: list[ReviewReason] = []
        if len(cluster_candidates) > 1:
            reasons.append(ReviewReason.COMPETING_VALUES)
        if missing_entity_alternative:
            reasons.append(ReviewReason.MISSING_ENTITY_ALTERNATIVE)
        if total_support <= 0.0:
            reasons.append(ReviewReason.NO_POSITIVE_SUPPORT)
        if invalid:
            reasons.append(ReviewReason.INVALID_CANONICAL_VALUE)
        if unpinned:
            reasons.append(ReviewReason.MISSING_PINNED_PROVENANCE)
        route = EntityRoute.REVIEW if reasons else EntityRoute.PRESERVE_SURFACE
        decisions.append(
            EntityDecision(
                decision_id=_stable_id(
                    "decision",
                    cluster[0].entity_kind,
                    cluster_start_ms,
                    cluster_end_ms,
                ),
                entity_kind=cluster[0].entity_kind,
                start_ms=cluster_start_ms,
                end_ms=cluster_end_ms,
                route=route,
                selected_candidate_id=ordered[0] if route is EntityRoute.PRESERVE_SURFACE else None,
                candidate_ids=ordered,
                reason_codes=tuple(reasons),
            )
        )

    # A time-aligned alternative can interpret the same acoustic region as a
    # different critical type (for example, a phone number versus an account
    # identifier). Treat that as competition, not two independently safe
    # entities. Cross-type coexistence needs an explicit downstream semantic
    # model and cannot be inferred from decoder output alone.
    cross_type_decisions: set[str] = set()
    candidates_by_id = {candidate.candidate_id: candidate for candidate in all_candidates}
    for left_decision, right_decision in combinations(decisions, 2):
        if left_decision.entity_kind is right_decision.entity_kind:
            continue
        temporal = _interval_iou(
            (left_decision.start_ms, left_decision.end_ms),
            (right_decision.start_ms, right_decision.end_ms),
        )
        if temporal < 0.5:
            continue
        cross_type_decisions.update((left_decision.decision_id, right_decision.decision_id))
        for left_id in left_decision.candidate_ids:
            for right_id in right_decision.candidate_ids:
                left_candidate = candidates_by_id[left_id]
                right_candidate = candidates_by_id[right_id]
                similarity = _surface_similarity(
                    left_candidate.representative_surface,
                    right_candidate.representative_surface,
                )
                all_edges.append(
                    LatticeEdge(
                        source_candidate_id=left_id,
                        target_candidate_id=right_id,
                        relation=EdgeRelation.COMPETING_ENTITY_TYPE,
                        temporal_iou=temporal,
                        surface_similarity=similarity,
                        compatibility_weight=(temporal + similarity) / 2.0,
                    )
                )

    decisions = [
        EntityDecision(
            decision_id=decision.decision_id,
            entity_kind=decision.entity_kind,
            start_ms=decision.start_ms,
            end_ms=decision.end_ms,
            route=EntityRoute.REVIEW,
            selected_candidate_id=None,
            candidate_ids=decision.candidate_ids,
            reason_codes=(*decision.reason_codes, ReviewReason.COMPETING_ENTITY_TYPES),
        )
        if decision.decision_id in cross_type_decisions
        and ReviewReason.COMPETING_ENTITY_TYPES not in decision.reason_codes
        else decision
        for decision in decisions
    ]

    return EntityLattice(
        candidates=tuple(all_candidates),
        edges=tuple(all_edges),
        decisions=tuple(decisions),
    )
