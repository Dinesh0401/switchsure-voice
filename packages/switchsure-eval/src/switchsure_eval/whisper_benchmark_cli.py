"""Command-line entry point for a pinned, privacy-safe Whisper benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from switchsure_ml.asr_adapters import FasterWhisperAdapter
from switchsure_ml.dataset_manifest import (
    DataRecord,
    DatasetManifest,
    DatasetSource,
    DataSplit,
    EvidenceStatus,
)

from .whisper_benchmark import (
    BenchmarkBackend,
    BenchmarkConfig,
    blocked_artifact,
    preflight,
    run_benchmark,
    write_artifact,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _load_manifest(path: Path) -> DatasetManifest:
    root = _mapping(json.loads(path.read_text(encoding="utf-8")), "manifest")
    sources: list[DatasetSource] = []
    raw_sources = root.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("manifest sources must be a list")
    for raw in raw_sources:
        item = _mapping(raw, "dataset source")
        sources.append(
            DatasetSource(
                dataset_id=_text(item.get("dataset_id"), "dataset_id"),
                title=_text(item.get("title"), "title"),
                version=_text(item.get("version"), "version"),
                source_uri=_text(item.get("source_uri"), "source_uri"),
                provenance=_text(item.get("provenance"), "provenance"),
                license_status=EvidenceStatus(_text(item.get("license_status"), "license_status")),
                consent_status=EvidenceStatus(_text(item.get("consent_status"), "consent_status")),
                license_expression=_optional_text(
                    item.get("license_expression"), "license_expression"
                ),
                consent_statement=_optional_text(
                    item.get("consent_statement"), "consent_statement"
                ),
            )
        )

    records: list[DataRecord] = []
    raw_records = root.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("manifest records must be a list")
    for raw in raw_records:
        item = _mapping(raw, "data record")
        languages = item.get("languages")
        metadata = item.get("metadata", {})
        if not isinstance(languages, list) or not all(
            isinstance(value, str) for value in languages
        ):
            raise ValueError("record languages must be a list of strings")
        metadata_map = _mapping(metadata, "record metadata")
        records.append(
            DataRecord(
                record_id=_text(item.get("record_id"), "record_id"),
                dataset_id=_text(item.get("dataset_id"), "dataset_id"),
                audio_path=_text(item.get("audio_path"), "audio_path"),
                audio_sha256=_text(item.get("audio_sha256"), "audio_sha256"),
                split=DataSplit(_text(item.get("split"), "split")),
                speaker_id=_text(item.get("speaker_id"), "speaker_id"),
                session_id=_text(item.get("session_id"), "session_id"),
                duration_ms=int(item.get("duration_ms", 0)),
                languages=tuple(languages),
                transcript_path=_optional_text(item.get("transcript_path"), "transcript_path"),
                metadata=tuple(
                    sorted(
                        (
                            _text(key, "metadata key"),
                            _text(value, "metadata value"),
                        )
                        for key, value in metadata_map.items()
                    )
                ),
            )
        )
    return DatasetManifest(
        manifest_id=_text(root.get("manifest_id"), "manifest_id"),
        schema_version=_text(root.get("schema_version"), "schema_version"),
        normalization_version=_text(root.get("normalization_version"), "normalization_version"),
        sources=tuple(sources),
        records=tuple(records),
    )


def _repo_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "uv.lock").is_file() and (candidate / ".git").exists():
            return candidate
    raise ValueError("could not locate repository root from benchmark config")


def _build_config(
    config_path: Path,
    *,
    split: DataSplit,
    allow_locked_test: bool,
) -> BenchmarkConfig:
    raw = _mapping(tomllib.loads(config_path.read_text(encoding="utf-8")), "config")
    backbone = _mapping(raw.get("backbone"), "backbone")
    repo_root = _repo_root(config_path)
    snapshot_value = backbone.get("local_snapshot_path")
    snapshot = None
    if isinstance(snapshot_value, str):
        snapshot = Path(snapshot_value)
        if not snapshot.is_absolute():
            snapshot = (repo_root / snapshot).resolve()
    adapter = _text(backbone.get("adapter"), "backbone.adapter")
    if adapter != BenchmarkBackend.FASTER_WHISPER.value:
        raise ValueError("benchmark CLI currently supports only faster-whisper")
    return BenchmarkConfig(
        run_id=_text(raw.get("experiment_id"), "experiment_id"),
        backend=BenchmarkBackend.FASTER_WHISPER,
        model_id=_text(backbone.get("model_id"), "backbone.model_id"),
        model_revision=_text(backbone.get("model_revision"), "backbone.model_revision"),
        split=split,
        seed=int(raw.get("seed", 0)),
        git_commit=_git_commit(repo_root),
        environment_lock_hash=_sha256_file(repo_root / "uv.lock"),
        device=_text(backbone.get("device", "auto"), "backbone.device"),
        compute_type=_text(backbone.get("compute_type", "default"), "backbone.compute_type"),
        beam_size=int(backbone.get("beam_size", 5)),
        word_timestamps=bool(backbone.get("word_timestamps", True)),
        vad_filter=bool(backbone.get("vad_filter", True)),
        multilingual=bool(backbone.get("multilingual", True)),
        allow_locked_test=allow_locked_test,
        local_model_snapshot=snapshot,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=tuple(item.value for item in DataSplit),
        default="validation",
    )
    parser.add_argument("--allow-locked-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _build_config(
        args.config.resolve(),
        split=DataSplit(args.split),
        allow_locked_test=bool(args.allow_locked_test),
    )
    manifest = _load_manifest(args.manifest.resolve())
    data_root = args.data_root.resolve()
    issues = preflight(manifest, config, data_root)
    if issues:
        artifact = blocked_artifact(manifest, config, issues)
        write_artifact(args.output.resolve(), artifact)
        print(json.dumps({"status": "blocked", "blockers": [item.code for item in issues]}))
        return 2

    assert config.local_model_snapshot is not None
    backend = importlib.import_module("faster_whisper")

    def model_factory(model_id: str, **kwargs: object) -> object:
        return backend.WhisperModel(model_id, **kwargs)

    adapter = FasterWhisperAdapter(
        str(config.local_model_snapshot),
        model_revision=config.model_revision,
        device=config.device,
        compute_type=config.compute_type,
        model_factory=model_factory,
    )
    artifact = run_benchmark(manifest, config, data_root, adapter)
    write_artifact(args.output.resolve(), artifact)
    print(json.dumps({"status": artifact.status.value, "records": len(artifact.records)}))
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution boundary
    raise SystemExit(main())
