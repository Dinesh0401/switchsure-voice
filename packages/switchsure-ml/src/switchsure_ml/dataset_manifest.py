"""Deterministic dataset manifests with explicit governance and leakage checks.

This module intentionally does not touch audio or download datasets.  It defines
the evidence required before an experiment may claim that its data is usable.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Final

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


class DataSplit(StrEnum):
    """Permitted experiment partitions."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    VALIDATION = "validation"
    TEST = "test"


class EvidenceStatus(StrEnum):
    """Review state for license or consent evidence."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class IssueSeverity(StrEnum):
    """Whether an issue blocks an experiment or records a governance warning."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ManifestIssue:
    """Machine-readable manifest validation result."""

    code: str
    message: str
    severity: IssueSeverity


class ManifestValidationError(ValueError):
    """Raised when a manifest does not meet its requested validation policy."""

    def __init__(self, issues: tuple[ManifestIssue, ...]) -> None:
        self.issues = issues
        summary = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        super().__init__(summary)


def _require_text(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and have no surrounding whitespace")


def _validate_relative_path(value: str, field_name: str) -> None:
    _require_text(value, field_name)
    if "\\" in value:
        raise ValueError(f"{field_name} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("./"):
        raise ValueError(f"{field_name} must be a normalized relative path without traversal")
    if str(path) != value:
        raise ValueError(f"{field_name} must be a normalized relative POSIX path")


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """Governance evidence for a dataset contributing one or more records."""

    dataset_id: str
    title: str
    version: str
    source_uri: str
    provenance: str
    license_status: EvidenceStatus
    consent_status: EvidenceStatus
    license_expression: str | None = None
    consent_statement: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.license_status, EvidenceStatus):
            raise ValueError("license_status must be an EvidenceStatus")
        if not isinstance(self.consent_status, EvidenceStatus):
            raise ValueError("consent_status must be an EvidenceStatus")
        for field_name in ("dataset_id", "title", "version", "source_uri", "provenance"):
            _require_text(getattr(self, field_name), field_name)
        if self.license_expression is not None:
            _require_text(self.license_expression, "license_expression")
        if self.consent_statement is not None:
            _require_text(self.consent_statement, "consent_statement")
        if self.license_status is EvidenceStatus.VERIFIED and self.license_expression is None:
            raise ValueError("verified license_status requires license_expression")
        if self.consent_status is EvidenceStatus.VERIFIED and self.consent_statement is None:
            raise ValueError("verified consent_status requires consent_statement")

    @property
    def approved_for_experiment(self) -> bool:
        """Return true only when both license and consent evidence are verified."""

        return (
            self.license_status is EvidenceStatus.VERIFIED
            and self.consent_status is EvidenceStatus.VERIFIED
        )


@dataclass(frozen=True, slots=True)
class DataRecord:
    """One immutable audio example and its split-critical identity metadata."""

    record_id: str
    dataset_id: str
    audio_path: str
    audio_sha256: str
    split: DataSplit
    speaker_id: str
    session_id: str
    duration_ms: int
    languages: tuple[str, ...]
    transcript_path: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.split, DataSplit):
            raise ValueError("split must be one of train, calibration, validation, or test")
        for field_name in ("record_id", "dataset_id", "speaker_id", "session_id"):
            _require_text(getattr(self, field_name), field_name)
        _validate_relative_path(self.audio_path, "audio_path")
        if self.transcript_path is not None:
            _validate_relative_path(self.transcript_path, "transcript_path")
        if not _SHA256_PATTERN.fullmatch(self.audio_sha256):
            raise ValueError("audio_sha256 must be a lowercase 64-character SHA-256 digest")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be greater than zero")
        if not self.languages:
            raise ValueError("languages must contain at least one declared language")
        if len(set(self.languages)) != len(self.languages):
            raise ValueError("languages must not contain duplicates")
        for language in self.languages:
            _require_text(language, "languages entry")

        metadata_keys: list[str] = []
        for key, value in self.metadata:
            _require_text(key, "metadata key")
            _require_text(value, f"metadata value for {key}")
            metadata_keys.append(key)
        if metadata_keys != sorted(metadata_keys) or len(set(metadata_keys)) != len(metadata_keys):
            raise ValueError("metadata must have unique keys in lexical order")


def _canonical_source(source: DatasetSource) -> dict[str, str | None]:
    return {
        "consent_statement": source.consent_statement,
        "consent_status": source.consent_status.value,
        "dataset_id": source.dataset_id,
        "license_expression": source.license_expression,
        "license_status": source.license_status.value,
        "provenance": source.provenance,
        "source_uri": source.source_uri,
        "title": source.title,
        "version": source.version,
    }


def _canonical_record(record: DataRecord) -> dict[str, object]:
    return {
        "audio_path": record.audio_path,
        "audio_sha256": record.audio_sha256,
        "dataset_id": record.dataset_id,
        "duration_ms": record.duration_ms,
        "languages": list(record.languages),
        "metadata": {key: value for key, value in record.metadata},
        "record_id": record.record_id,
        "session_id": record.session_id,
        "speaker_id": record.speaker_id,
        "split": record.split.value,
        "transcript_path": record.transcript_path,
    }


def _stable_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Reproducible collection of sources and speaker-disjoint records."""

    manifest_id: str
    schema_version: str
    normalization_version: str
    sources: tuple[DatasetSource, ...]
    records: tuple[DataRecord, ...]

    def __post_init__(self) -> None:
        for field_name in ("manifest_id", "schema_version", "normalization_version"):
            _require_text(getattr(self, field_name), field_name)
        if not self.sources:
            raise ValueError("sources cannot be empty")
        if not self.records:
            raise ValueError("records cannot be empty")

    def issues(self, *, require_governance_approval: bool = False) -> tuple[ManifestIssue, ...]:
        """Return deterministic structural, governance, and split-isolation issues."""

        issues: list[ManifestIssue] = []
        source_ids = [source.dataset_id for source in self.sources]
        duplicate_source_ids = sorted(
            source_id for source_id in set(source_ids) if source_ids.count(source_id) > 1
        )
        for source_id in duplicate_source_ids:
            issues.append(
                ManifestIssue(
                    "duplicate_dataset_id",
                    f"dataset_id {source_id!r} occurs more than once",
                    IssueSeverity.ERROR,
                )
            )

        known_sources = set(source_ids)
        for source in sorted(self.sources, key=lambda item: item.dataset_id):
            governance_rejected = (
                source.license_status is EvidenceStatus.REJECTED
                or source.consent_status is EvidenceStatus.REJECTED
            )
            if governance_rejected:
                issues.append(
                    ManifestIssue(
                        "governance_rejected",
                        f"dataset {source.dataset_id!r} has rejected license or consent evidence",
                        IssueSeverity.ERROR,
                    )
                )
            elif not source.approved_for_experiment:
                severity = (
                    IssueSeverity.ERROR if require_governance_approval else IssueSeverity.WARNING
                )
                issues.append(
                    ManifestIssue(
                        "governance_not_verified",
                        f"dataset {source.dataset_id!r} lacks verified license "
                        "and consent evidence",
                        severity,
                    )
                )

        record_ids = [record.record_id for record in self.records]
        audio_paths = [record.audio_path for record in self.records]
        for value in sorted(value for value in set(record_ids) if record_ids.count(value) > 1):
            issues.append(
                ManifestIssue(
                    "duplicate_record_id",
                    f"record_id {value!r} occurs more than once",
                    IssueSeverity.ERROR,
                )
            )
        for value in sorted(value for value in set(audio_paths) if audio_paths.count(value) > 1):
            issues.append(
                ManifestIssue(
                    "duplicate_audio_path",
                    f"audio_path {value!r} occurs more than once",
                    IssueSeverity.ERROR,
                )
            )
        for record in sorted(self.records, key=lambda item: item.record_id):
            if record.dataset_id not in known_sources:
                issues.append(
                    ManifestIssue(
                        "unknown_dataset_id",
                        f"record {record.record_id!r} references unknown dataset "
                        f"{record.dataset_id!r}",
                        IssueSeverity.ERROR,
                    )
                )

        issues.extend(self._leakage_issues("speaker", lambda record: record.speaker_id))
        issues.extend(self._leakage_issues("session", lambda record: record.session_id))
        return tuple(issues)

    def _leakage_issues(
        self,
        identity_name: str,
        identity: Callable[[DataRecord], str],
    ) -> tuple[ManifestIssue, ...]:
        memberships: dict[str, set[DataSplit]] = {}
        for record in self.records:
            value = identity(record)
            memberships.setdefault(value, set()).add(record.split)
        leaked = sorted((value, splits) for value, splits in memberships.items() if len(splits) > 1)
        return tuple(
            ManifestIssue(
                f"{identity_name}_split_leakage",
                f"{identity_name} {value!r} appears in splits "
                + ", ".join(sorted(split.value for split in splits)),
                IssueSeverity.ERROR,
            )
            for value, splits in leaked
        )

    def validate(self, *, require_governance_approval: bool = False) -> None:
        """Raise for errors under the selected governance policy."""

        errors = tuple(
            issue
            for issue in self.issues(require_governance_approval=require_governance_approval)
            if issue.severity is IssueSeverity.ERROR
        )
        if errors:
            raise ManifestValidationError(errors)

    def stable_hash(self) -> str:
        """Hash all reproducibility-relevant content independent of tuple ordering."""

        return _stable_digest(
            {
                "manifest_id": self.manifest_id,
                "normalization_version": self.normalization_version,
                "records": [
                    _canonical_record(record)
                    for record in sorted(self.records, key=lambda item: item.record_id)
                ],
                "schema_version": self.schema_version,
                "sources": [
                    _canonical_source(source)
                    for source in sorted(self.sources, key=lambda item: item.dataset_id)
                ],
            }
        )

    def split_hash(self, split: DataSplit) -> str:
        """Hash one split plus its source governance and normalization context."""

        records = sorted(
            (record for record in self.records if record.split is split),
            key=lambda item: item.record_id,
        )
        used_source_ids = {record.dataset_id for record in records}
        sources = sorted(
            (source for source in self.sources if source.dataset_id in used_source_ids),
            key=lambda item: item.dataset_id,
        )
        return _stable_digest(
            {
                "normalization_version": self.normalization_version,
                "records": [_canonical_record(record) for record in records],
                "schema_version": self.schema_version,
                "sources": [_canonical_source(source) for source in sources],
                "split": split.value,
            }
        )
