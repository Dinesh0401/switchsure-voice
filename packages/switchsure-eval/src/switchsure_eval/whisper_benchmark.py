"""Fail-closed, privacy-safe execution for the Phase-1 Whisper baseline.

The runner deliberately keeps transcripts in memory.  Its artifact contains
only aggregate/edit statistics and salted record keys, never audio paths,
reference text, hypotheses, or entity surfaces.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from switchsure_ml.asr_adapters import AsrAdapterError
from switchsure_ml.asr_contracts import AsrEvidencePort, AsrOptions, AsrRequest, AudioSource
from switchsure_ml.dataset_manifest import (
    DataRecord,
    DatasetManifest,
    DataSplit,
    ManifestValidationError,
)

from .metrics import EditCounts, word_error_counts
from .normalization import NORMALIZATION_VERSION, normalize_transcript

_SHA256_PREFIX: Final = "sha256:"


class BenchmarkBackend(StrEnum):
    FASTER_WHISPER = "faster-whisper"
    FIXTURE = "fixture"


class RunStatus(StrEnum):
    BLOCKED = "blocked"
    SMOKE = "smoke"
    COMPLETE = "complete"


class RecordOutcome(StrEnum):
    SUCCESS = "success"
    MISSING_AUDIO = "missing_audio"
    AUDIO_HASH_MISMATCH = "audio_hash_mismatch"
    MISSING_REFERENCE = "missing_reference"
    INVALID_REFERENCE = "invalid_reference"
    TIMEOUT = "timeout"
    DECODER_FAILURE = "decoder_failure"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    run_id: str
    backend: BenchmarkBackend
    model_id: str
    model_revision: str
    split: DataSplit
    seed: int
    git_commit: str
    environment_lock_hash: str
    device: str = "auto"
    compute_type: str = "default"
    beam_size: int = 5
    word_timestamps: bool = True
    vad_filter: bool = True
    multilingual: bool = True
    allow_locked_test: bool = False
    fixture_smoke_only: bool = False
    local_model_snapshot: Path | None = None

    def __post_init__(self) -> None:
        required = (
            self.run_id,
            self.model_id,
            self.model_revision,
            self.git_commit,
            self.environment_lock_hash,
            self.device,
            self.compute_type,
        )
        if any(not value.strip() for value in required):
            raise ValueError("benchmark provenance fields cannot be blank")
        if self.beam_size < 1:
            raise ValueError("beam_size must be positive")
        if self.backend is BenchmarkBackend.FIXTURE and not self.fixture_smoke_only:
            raise ValueError("fixture backend must be explicitly marked smoke-only")
        if self.split is DataSplit.TEST and not self.allow_locked_test:
            raise ValueError("locked test execution requires explicit authorization")


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class HardwareSnapshot:
    operating_system: str
    machine: str
    processor: str
    python_version: str
    logical_cpu_count: int | None
    accelerator: str | None


@dataclass(frozen=True, slots=True)
class PrivateEditSummary:
    substitutions: int
    deletions: int
    insertions: int
    correct: int

    @classmethod
    def from_counts(cls, counts: EditCounts) -> PrivateEditSummary:
        return cls(
            substitutions=counts.substitutions,
            deletions=counts.deletions,
            insertions=counts.insertions,
            correct=counts.correct,
        )

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def reference_length(self) -> int:
        return self.substitutions + self.deletions + self.correct


@dataclass(frozen=True, slots=True)
class RecordResult:
    record_key: str
    outcome: RecordOutcome
    duration_ms: int
    latency_ms: float | None = None
    real_time_factor: float | None = None
    normalized_word_edits: PrivateEditSummary | None = None
    surface_word_edits: PrivateEditSummary | None = None
    normalized_character_edits: PrivateEditSummary | None = None
    failure_code: str | None = None


@dataclass(frozen=True, slots=True)
class AggregateResult:
    record_count: int
    outcome_counts: tuple[tuple[str, int], ...]
    normalized_wer: float | None
    surface_wer: float | None
    normalized_cer: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    cold_start_ms: float | None
    real_time_factor: float | None
    peak_memory_mb: float | None
    peak_memory_method: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkArtifact:
    schema_version: str
    status: RunStatus
    run_id: str
    config_hash: str
    git_commit: str
    environment_lock_hash: str
    dataset_manifest_hash: str
    split_hash: str
    normalization_version: str
    backend: str
    backend_version: str | None
    model_id: str
    model_revision: str
    seed: int
    hardware: HardwareSnapshot
    preflight_issues: tuple[PreflightIssue, ...]
    aggregate: AggregateResult | None
    records: tuple[RecordResult, ...]
    privacy_notice: str = (
        "No audio path, transcript, hypothesis, or entity surface is stored in this artifact."
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_directory(path: Path) -> str:
    """Hash a local model snapshot without exposing its absolute path."""

    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError("model snapshot contains no files")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_digest_file(candidate)))
    return digest.hexdigest()


def _config_hash(config: BenchmarkConfig) -> str:
    public = {
        "backend": config.backend.value,
        "beam_size": config.beam_size,
        "compute_type": config.compute_type,
        "device": config.device,
        "fixture_smoke_only": config.fixture_smoke_only,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "multilingual": config.multilingual,
        "run_id": config.run_id,
        "seed": config.seed,
        "split": config.split.value,
        "vad_filter": config.vad_filter,
        "word_timestamps": config.word_timestamps,
    }
    encoded = json.dumps(public, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _SHA256_PREFIX + _digest_bytes(encoded)


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _accelerator_name() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    return ", ".join(names) or None


def hardware_snapshot() -> HardwareSnapshot:
    return HardwareSnapshot(
        operating_system=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        logical_cpu_count=os.cpu_count(),
        accelerator=_accelerator_name(),
    )


def _peak_memory_mb() -> tuple[float | None, str | None]:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            if ok:
                return counters.PeakWorkingSetSize / (1024 * 1024), "process_peak_working_set"
        except (AttributeError, OSError):
            return None, None
    return None, None


def preflight(
    manifest: DatasetManifest,
    config: BenchmarkConfig,
    data_root: Path,
) -> tuple[PreflightIssue, ...]:
    """Return every blocker; an experiment may run only when this tuple is empty."""

    issues: list[PreflightIssue] = []
    try:
        manifest.validate(require_governance_approval=True)
    except ManifestValidationError as error:
        issues.extend(PreflightIssue(item.code, item.message) for item in error.issues)

    if manifest.normalization_version != NORMALIZATION_VERSION:
        issues.append(
            PreflightIssue(
                "normalization_version_mismatch",
                "manifest normalization version does not match the evaluator",
            )
        )
    if config.model_revision.upper() == "UNPINNED":
        issues.append(PreflightIssue("model_revision_unpinned", "model revision must be immutable"))

    selected = tuple(record for record in manifest.records if record.split is config.split)
    if not selected:
        issues.append(PreflightIssue("empty_split", "selected split has no records"))

    if config.backend is BenchmarkBackend.FASTER_WHISPER:
        if importlib.util.find_spec("faster_whisper") is None:
            issues.append(
                PreflightIssue(
                    "backend_missing",
                    "faster-whisper is not installed in this environment",
                )
            )
        snapshot = config.local_model_snapshot
        if snapshot is None:
            issues.append(
                PreflightIssue(
                    "immutable_model_snapshot_required",
                    "faster-whisper execution requires a pinned local model snapshot",
                )
            )
        elif not snapshot.is_absolute() or not snapshot.is_dir():
            issues.append(
                PreflightIssue("model_snapshot_missing", "local model snapshot is unavailable")
            )
        elif config.model_revision.startswith(_SHA256_PREFIX):
            observed = hash_directory(snapshot)
            if observed != config.model_revision.removeprefix(_SHA256_PREFIX):
                issues.append(
                    PreflightIssue(
                        "model_hash_mismatch",
                        "local model snapshot hash does not match",
                    )
                )
        else:
            issues.append(
                PreflightIssue(
                    "model_revision_not_content_hash",
                    "local faster-whisper model revision must use sha256:<digest>",
                )
            )

    for record in selected:
        audio_path = data_root / Path(record.audio_path)
        if not audio_path.is_file():
            issues.append(
                PreflightIssue("missing_audio", "one or more audio files are unavailable")
            )
            break
    for record in selected:
        if record.transcript_path is None or not (data_root / record.transcript_path).is_file():
            issues.append(
                PreflightIssue(
                    "missing_reference",
                    "one or more reference transcripts are unavailable",
                )
            )
            break
    return tuple(issues)


def _read_reference(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("transcript"), str):
            raise ValueError("reference JSON must contain a transcript string")
        text = payload["transcript"]
    if not text.strip():
        raise ValueError("reference transcript is empty")
    return text


def _record_key(manifest_hash: str, record_id: str) -> str:
    return _digest_bytes(f"{manifest_hash}:{record_id}".encode())


def _empty_result(record: DataRecord, key: str, outcome: RecordOutcome) -> RecordResult:
    return RecordResult(
        record_key=key,
        outcome=outcome,
        duration_ms=record.duration_ms,
        failure_code=outcome.value,
    )


def _aggregate(results: Sequence[RecordResult]) -> AggregateResult:
    normalized = [item.normalized_word_edits for item in results if item.normalized_word_edits]
    surface = [item.surface_word_edits for item in results if item.surface_word_edits]
    characters = [
        item.normalized_character_edits for item in results if item.normalized_character_edits
    ]

    def error_rate(items: Sequence[PrivateEditSummary]) -> float | None:
        denominator = sum(item.reference_length for item in items)
        return sum(item.errors for item in items) / denominator if denominator else None

    latencies = [item.latency_ms for item in results if item.latency_ms is not None]
    durations = [item.duration_ms for item in results if item.latency_ms is not None]
    peak_memory, peak_method = _peak_memory_mb()
    counts = Counter(item.outcome.value for item in results)
    return AggregateResult(
        record_count=len(results),
        outcome_counts=tuple(sorted(counts.items())),
        normalized_wer=error_rate(normalized),
        surface_wer=error_rate(surface),
        normalized_cer=error_rate(characters),
        latency_p50_ms=_percentile(latencies, 0.5),
        latency_p95_ms=_percentile(latencies, 0.95),
        cold_start_ms=latencies[0] if latencies else None,
        real_time_factor=(sum(latencies) / sum(durations))
        if durations and sum(durations)
        else None,
        peak_memory_mb=peak_memory,
        peak_memory_method=peak_method,
    )


def _backend_version(backend: BenchmarkBackend) -> str | None:
    if backend is BenchmarkBackend.FIXTURE:
        return "deterministic-fixture-v1"
    try:
        from importlib.metadata import version

        return version("faster-whisper")
    except Exception:  # pragma: no cover - defensive metadata lookup
        return None


def blocked_artifact(
    manifest: DatasetManifest,
    config: BenchmarkConfig,
    issues: tuple[PreflightIssue, ...],
) -> BenchmarkArtifact:
    return BenchmarkArtifact(
        schema_version="1.0",
        status=RunStatus.BLOCKED,
        run_id=config.run_id,
        config_hash=_config_hash(config),
        git_commit=config.git_commit,
        environment_lock_hash=config.environment_lock_hash,
        dataset_manifest_hash=_SHA256_PREFIX + manifest.stable_hash(),
        split_hash=_SHA256_PREFIX + manifest.split_hash(config.split),
        normalization_version=NORMALIZATION_VERSION,
        backend=config.backend.value,
        backend_version=_backend_version(config.backend),
        model_id=config.model_id,
        model_revision=config.model_revision,
        seed=config.seed,
        hardware=hardware_snapshot(),
        preflight_issues=issues,
        aggregate=None,
        records=(),
    )


def run_benchmark(
    manifest: DatasetManifest,
    config: BenchmarkConfig,
    data_root: Path,
    adapter: AsrEvidencePort,
    *,
    monotonic: Callable[[], float] = time.perf_counter,
) -> BenchmarkArtifact:
    """Run a validated split and return a content-redacted research artifact."""

    issues = preflight(manifest, config, data_root)
    if issues:
        return blocked_artifact(manifest, config, issues)

    manifest_hash = manifest.stable_hash()
    results: list[RecordResult] = []
    for record in manifest.records:
        if record.split is not config.split:
            continue
        key = _record_key(manifest_hash, record.record_id)
        audio_path = (data_root / record.audio_path).resolve()
        reference_path = (
            (data_root / record.transcript_path).resolve() if record.transcript_path else None
        )
        if not audio_path.is_file():
            results.append(_empty_result(record, key, RecordOutcome.MISSING_AUDIO))
            continue
        if _digest_file(audio_path) != record.audio_sha256:
            results.append(_empty_result(record, key, RecordOutcome.AUDIO_HASH_MISMATCH))
            continue
        if reference_path is None or not reference_path.is_file():
            results.append(_empty_result(record, key, RecordOutcome.MISSING_REFERENCE))
            continue
        try:
            reference = _read_reference(reference_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            results.append(_empty_result(record, key, RecordOutcome.INVALID_REFERENCE))
            continue

        request = AsrRequest(
            request_id=key,
            source_id=record.record_id,
            audio=AudioSource(path=audio_path),
            options=AsrOptions(
                beam_size=config.beam_size,
                word_timestamps=config.word_timestamps,
                vad_filter=config.vad_filter,
                multilingual=config.multilingual,
            ),
        )
        started = monotonic()
        try:
            evidence = adapter.transcribe(request)
        except TimeoutError:
            results.append(_empty_result(record, key, RecordOutcome.TIMEOUT))
            continue
        except (AsrAdapterError, OSError, RuntimeError):
            results.append(_empty_result(record, key, RecordOutcome.DECODER_FAILURE))
            continue
        latency_ms = (monotonic() - started) * 1000
        normalized_reference = normalize_transcript(reference)
        normalized_hypothesis = normalize_transcript(evidence.transcript)
        normalized_words = word_error_counts(
            normalized_reference.tokens, normalized_hypothesis.tokens
        )
        surface_words = word_error_counts(reference.split(), evidence.transcript.split())
        characters = word_error_counts(
            tuple(normalized_reference.normalized), tuple(normalized_hypothesis.normalized)
        )
        results.append(
            RecordResult(
                record_key=key,
                outcome=RecordOutcome.SUCCESS,
                duration_ms=record.duration_ms,
                latency_ms=latency_ms,
                real_time_factor=latency_ms / record.duration_ms,
                normalized_word_edits=PrivateEditSummary.from_counts(normalized_words),
                surface_word_edits=PrivateEditSummary.from_counts(surface_words),
                normalized_character_edits=PrivateEditSummary.from_counts(characters),
            )
        )

    status = RunStatus.SMOKE if config.fixture_smoke_only else RunStatus.COMPLETE
    return BenchmarkArtifact(
        schema_version="1.0",
        status=status,
        run_id=config.run_id,
        config_hash=_config_hash(config),
        git_commit=config.git_commit,
        environment_lock_hash=config.environment_lock_hash,
        dataset_manifest_hash=_SHA256_PREFIX + manifest_hash,
        split_hash=_SHA256_PREFIX + manifest.split_hash(config.split),
        normalization_version=NORMALIZATION_VERSION,
        backend=config.backend.value,
        backend_version=_backend_version(config.backend),
        model_id=config.model_id,
        model_revision=config.model_revision,
        seed=config.seed,
        hardware=hardware_snapshot(),
        preflight_issues=(),
        aggregate=_aggregate(results),
        records=tuple(results),
    )


def write_artifact(path: Path, artifact: BenchmarkArtifact) -> None:
    """Atomically write a redacted artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(artifact.to_json(), encoding="utf-8", newline="\n")
    temporary.replace(path)
