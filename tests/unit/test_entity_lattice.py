import pytest
from switchsure_ml.entity_lattice import (
    EdgeRelation,
    EntityHypothesis,
    EntityKind,
    EntityRoute,
    EntityWord,
    HypothesisProvenance,
    ReviewReason,
    build_entity_lattice,
    canonicalize_entity,
    recognize_structured_entities,
)


def provenance(*, revision: str | None = "model-sha-123") -> HypothesisProvenance:
    return HypothesisProvenance(
        adapter="fixture",
        model_id="whisper-test",
        model_revision=revision,
        decoding_id="beam-5",
    )


def hypothesis(
    hypothesis_id: str,
    surfaces: tuple[str, ...],
    *,
    weight: float = 1.0,
    revision: str | None = "model-sha-123",
    offset_ms: int = 0,
) -> EntityHypothesis:
    words = tuple(
        EntityWord(
            index=index,
            surface=surface,
            start_ms=offset_ms + index * 300,
            end_ms=offset_ms + (index + 1) * 300,
        )
        for index, surface in enumerate(surfaces)
    )
    return EntityHypothesis(
        hypothesis_id=hypothesis_id,
        words=words,
        provenance=provenance(revision=revision),
        relative_weight=weight,
    )


def test_tamil_digits_and_unit_are_canonicalized_reversibly() -> None:
    original = "௫௦௦ மில்லிகிராம்"

    canonical = canonicalize_entity(original, EntityKind.DOSAGE, locale="ta-IN")

    assert canonical.surface == original
    assert canonical.canonical_value == "500 mg"
    assert canonical.is_valid
    assert canonical.restore_surface() == original
    assert [step.operation for step in canonical.steps] == [
        "map_unicode_decimal_digits",
        "quantity_to_decimal_and_si_unit",
    ]


@pytest.mark.parametrize(
    ("surface", "kind", "expected"),
    [
        ("User@EXAMPLE.COM", EntityKind.EMAIL, "User@example.com"),
        ("+91 (98765) 43210", EntityKind.PHONE, "+919876543210"),
        ("12/11/2026", EntityKind.DATE, "2026-11-12"),
        ("09:30pm", EntityKind.TIME, "21:30"),
        ("₹500.00", EntityKind.MONEY, "INR 500"),
        ("ab-12345", EntityKind.IDENTIFIER, "ab-12345"),
    ],
)
def test_typed_canonicalizers_preserve_original_surface(
    surface: str, kind: EntityKind, expected: str
) -> None:
    canonical = canonicalize_entity(surface, kind)

    assert canonical.canonical_value == expected
    assert canonical.restore_surface() == surface


def test_recognizer_is_deterministic_and_preserves_span_provenance() -> None:
    item = hypothesis("h1", ("மருந்து", "௫௦௦", "மில்லிகிராம்", "09:30am"))

    first = recognize_structured_entities(item, locale="ta-IN")
    second = recognize_structured_entities(item, locale="ta-IN")

    assert first == second
    assert [mention.entity_kind for mention in first] == [EntityKind.DOSAGE, EntityKind.TIME]
    dosage = first[0]
    assert dosage.surface == "௫௦௦ மில்லிகிராம்"
    assert dosage.start_word == 1
    assert dosage.end_word == 2
    assert dosage.start_ms == 300
    assert dosage.end_ms == 900
    assert dosage.provenance.model_revision == "model-sha-123"


def test_identical_values_aggregate_evidence_without_becoming_confidence() -> None:
    first = hypothesis("beam-1", ("call", "+919876543210"), weight=0.8)
    second = hypothesis("beam-2", ("call", "+91-98765-43210"), weight=0.2)

    lattice = build_entity_lattice((first, second))

    assert len(lattice.candidates) == 1
    candidate = lattice.candidates[0]
    assert candidate.canonical_value == "+919876543210"
    assert len(candidate.mentions) == 2
    assert candidate.relative_support == pytest.approx(1.0)
    assert candidate.normalized_support == pytest.approx(1.0)
    assert candidate.calibrated is False
    assert lattice.decisions[0].route is EntityRoute.PRESERVE_SURFACE
    assert lattice.decisions[0].selected_candidate_id == candidate.candidate_id


def test_competing_critical_values_return_all_alternatives_for_review() -> None:
    first = hypothesis("beam-1", ("call", "+919876543210"), weight=0.75)
    second = hypothesis("beam-2", ("call", "+919876543211"), weight=0.25)

    lattice = build_entity_lattice((first, second))

    assert len(lattice.candidates) == 2
    assert sum(candidate.normalized_support for candidate in lattice.candidates) == pytest.approx(
        1.0
    )
    assert len(lattice.edges) == 1
    assert lattice.edges[0].relation is EdgeRelation.COMPETING_VALUE
    assert lattice.edges[0].calibrated is False
    decision = lattice.decisions[0]
    assert decision.route is EntityRoute.REVIEW
    assert decision.selected_candidate_id is None
    assert set(decision.candidate_ids) == {
        candidate.candidate_id for candidate in lattice.candidates
    }
    assert decision.reason_codes == (ReviewReason.COMPETING_VALUES,)


def test_invalid_locale_dependent_date_is_routed_to_review() -> None:
    item = hypothesis("h1", ("on", "12/11/2026"))

    lattice = build_entity_lattice((item,), locale="en-US")

    assert lattice.candidates[0].canonical_value is None
    assert lattice.decisions[0].route is EntityRoute.REVIEW
    assert lattice.decisions[0].reason_codes == (ReviewReason.INVALID_CANONICAL_VALUE,)


def test_unpinned_model_provenance_cannot_auto_preserve_critical_value() -> None:
    item = hypothesis("h1", ("ID", "AB-12345"), revision=None)

    lattice = build_entity_lattice((item,))

    assert lattice.decisions[0].route is EntityRoute.REVIEW
    assert lattice.decisions[0].reason_codes == (ReviewReason.MISSING_PINNED_PROVENANCE,)


def test_repeated_value_at_distinct_times_remains_two_events() -> None:
    words = (
        EntityWord(0, "+919876543210", 0, 300),
        EntityWord(1, "then", 300, 600),
        EntityWord(2, "+919876543210", 900, 1200),
    )
    item = EntityHypothesis("h1", words, provenance())

    lattice = build_entity_lattice((item,))

    assert len(lattice.candidates) == 2
    assert len(lattice.decisions) == 2
    assert all(decision.route is EntityRoute.PRESERVE_SURFACE for decision in lattice.decisions)


def test_zero_weight_alternatives_are_still_not_silently_discarded() -> None:
    first = hypothesis("beam-1", ("₹500",), weight=0.0)
    second = hypothesis("beam-2", ("₹900",), weight=0.0)

    lattice = build_entity_lattice((first, second))

    assert [candidate.normalized_support for candidate in lattice.candidates] == [0.0, 0.0]
    assert lattice.decisions[0].route is EntityRoute.REVIEW
    assert len(lattice.decisions[0].candidate_ids) == 2


def test_single_zero_weight_candidate_is_routed_to_review() -> None:
    item = hypothesis("beam-1", ("₹500",), weight=0.0)

    lattice = build_entity_lattice((item,))

    assert lattice.decisions[0].route is EntityRoute.REVIEW
    assert lattice.decisions[0].reason_codes == (ReviewReason.NO_POSITIVE_SUPPORT,)


def test_no_detected_entity_produces_an_empty_lattice() -> None:
    item = hypothesis("h1", ("வணக்கம்", "friend"))

    lattice = build_entity_lattice((item,))

    assert lattice.candidates == ()
    assert lattice.edges == ()
    assert lattice.decisions == ()


def test_sensitive_surface_and_canonical_values_are_excluded_from_repr() -> None:
    secret = "+919876543210"
    item = hypothesis("h1", (secret,))
    lattice = build_entity_lattice((item,))

    assert secret not in repr(item)
    assert secret not in repr(lattice)


def test_contract_rejects_duplicate_hypothesis_ids_and_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        hypothesis("bad", ("hello",), weight=-0.1)

    duplicate = hypothesis("same", ("hello",))
    with pytest.raises(ValueError, match="unique"):
        build_entity_lattice((duplicate, duplicate))
