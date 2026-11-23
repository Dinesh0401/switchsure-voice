from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
from switchsure_ml.dataset_manifest import (
    DataRecord,
    DatasetManifest,
    DatasetSource,
    DataSplit,
    EvidenceStatus,
    IssueSeverity,
    ManifestValidationError,
)

_AUDIO_HASH = "a" * 64


def _source(**changes: object) -> DatasetSource:
    values: dict[str, object] = {
        "dataset_id": "local-pilot",
        "title": "Consented local pilot",
        "version": "2026-11-23",
        "source_uri": "registry://switchsure/local-pilot",
        "provenance": "Internal data register entry DSR-001",
        "license_status": EvidenceStatus.VERIFIED,
        "consent_status": EvidenceStatus.VERIFIED,
        "license_expression": "Internal research agreement DSR-001",
        "consent_statement": "Consent form revision 2 permits this declared experiment",
    }
    values.update(changes)
    return DatasetSource(**cast(Any, values))


def _record(
    record_id: str = "utt-001",
    split: DataSplit = DataSplit.TRAIN,
    **changes: object,
) -> DataRecord:
    values: dict[str, object] = {
        "record_id": record_id,
        "dataset_id": "local-pilot",
        "audio_path": f"audio/{record_id}.wav",
        "audio_sha256": _AUDIO_HASH,
        "split": split,
        "speaker_id": f"speaker-{record_id}",
        "session_id": f"session-{record_id}",
        "duration_ms": 1_250,
        "languages": ("ta", "en"),
        "transcript_path": f"transcripts/{record_id}.json",
        "metadata": (("device", "phone"), ("noise_band", "quiet")),
    }
    values.update(changes)
    return DataRecord(**cast(Any, values))


def _manifest(*records: DataRecord, source: DatasetSource | None = None) -> DatasetManifest:
    return DatasetManifest(
        manifest_id="switchsure-phase1",
        schema_version="1.0.0",
        normalization_version="ta-en-v1",
        sources=(source or _source(),),
        records=records or (_record(),),
    )


def test_verified_source_requires_license_and_consent_evidence() -> None:
    with pytest.raises(ValueError, match="license_expression"):
        _source(license_expression=None)
    with pytest.raises(ValueError, match="consent_statement"):
        _source(consent_statement=None)

    assert _source().approved_for_experiment is True
    assert _source(license_status=EvidenceStatus.UNVERIFIED).approved_for_experiment is False


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"audio_path": "../private.wav"}, "relative path"),
        ({"audio_path": "audio\\sample.wav"}, "POSIX separators"),
        ({"audio_sha256": "not-a-digest"}, "64-character"),
        ({"duration_ms": 0}, "greater than zero"),
        ({"languages": ()}, "at least one"),
        ({"languages": ("ta", "ta")}, "duplicates"),
        ({"metadata": (("z", "last"), ("a", "first"))}, "lexical order"),
    ],
)
def test_data_record_rejects_non_reproducible_fields(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _record(**cast(Any, changes))


def test_data_record_rejects_unknown_split_even_when_typing_is_bypassed() -> None:
    with pytest.raises(ValueError, match="train, calibration, validation, or test"):
        _record(split=cast(Any, "development"))


def test_clean_manifest_passes_strict_validation() -> None:
    manifest = _manifest(
        _record("train", DataSplit.TRAIN),
        _record("cal", DataSplit.CALIBRATION),
        _record("val", DataSplit.VALIDATION),
        _record("test", DataSplit.TEST),
    )

    assert manifest.issues(require_governance_approval=True) == ()
    manifest.validate(require_governance_approval=True)


def test_unverified_governance_is_visible_and_strict_mode_blocks_it() -> None:
    source = _source(
        license_status=EvidenceStatus.UNVERIFIED,
        consent_status=EvidenceStatus.UNVERIFIED,
        license_expression=None,
        consent_statement=None,
    )
    manifest = _manifest(source=source)

    warning = manifest.issues()[0]
    assert warning.code == "governance_not_verified"
    assert warning.severity is IssueSeverity.WARNING
    manifest.validate()

    with pytest.raises(ManifestValidationError) as error:
        manifest.validate(require_governance_approval=True)
    assert [issue.code for issue in error.value.issues] == ["governance_not_verified"]


def test_rejected_governance_always_blocks_the_manifest() -> None:
    source = _source(
        license_status=EvidenceStatus.REJECTED,
        license_expression="Review DSR-002 found use prohibited",
    )
    manifest = _manifest(source=source)

    with pytest.raises(ManifestValidationError) as error:
        manifest.validate()
    assert [issue.code for issue in error.value.issues] == ["governance_rejected"]


def test_speaker_and_session_leakage_are_reported_across_any_splits() -> None:
    train = _record(
        "train",
        DataSplit.TRAIN,
        speaker_id="speaker-shared",
        session_id="session-shared",
    )
    calibration = _record(
        "calibration",
        DataSplit.CALIBRATION,
        speaker_id="speaker-shared",
        session_id="session-shared",
    )
    manifest = _manifest(train, calibration)

    issues = manifest.issues()
    assert [issue.code for issue in issues] == [
        "speaker_split_leakage",
        "session_split_leakage",
    ]
    assert "calibration, train" in issues[0].message
    with pytest.raises(ManifestValidationError, match="speaker_split_leakage"):
        manifest.validate()


def test_duplicate_identifiers_paths_and_unknown_sources_are_errors() -> None:
    first = _record("same", audio_path="audio/shared.wav")
    second = _record(
        "same",
        audio_path="audio/shared.wav",
        dataset_id="missing-source",
        speaker_id="other-speaker",
        session_id="other-session",
    )

    codes = [issue.code for issue in _manifest(first, second).issues()]
    assert codes == ["duplicate_record_id", "duplicate_audio_path", "unknown_dataset_id"]


def test_manifest_and_split_hashes_are_stable_and_order_independent() -> None:
    train = _record("train", DataSplit.TRAIN)
    test = _record("test", DataSplit.TEST)
    first = _manifest(train, test)
    reordered = replace(first, records=(test, train))

    assert first.stable_hash() == reordered.stable_hash()
    assert first.split_hash(DataSplit.TRAIN) == reordered.split_hash(DataSplit.TRAIN)
    assert len(first.stable_hash()) == 64
    assert len(first.split_hash(DataSplit.CALIBRATION)) == 64


def test_split_hash_changes_only_when_its_own_content_or_context_changes() -> None:
    train = _record("train", DataSplit.TRAIN)
    test = _record("test", DataSplit.TEST)
    baseline = _manifest(train, test)
    test_changed = _manifest(train, replace(test, duration_ms=2_000))
    normalization_changed = replace(baseline, normalization_version="ta-en-v2")

    assert baseline.stable_hash() != test_changed.stable_hash()
    assert baseline.split_hash(DataSplit.TRAIN) == test_changed.split_hash(DataSplit.TRAIN)
    assert baseline.split_hash(DataSplit.TEST) != test_changed.split_hash(DataSplit.TEST)
    assert baseline.split_hash(DataSplit.TRAIN) != normalization_changed.split_hash(DataSplit.TRAIN)
