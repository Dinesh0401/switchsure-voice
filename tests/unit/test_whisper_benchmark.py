from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from switchsure_eval import whisper_benchmark as benchmark
from switchsure_eval import whisper_benchmark_cli as benchmark_cli
from switchsure_eval.normalization import NORMALIZATION_VERSION
from switchsure_eval.whisper_benchmark import (
    BenchmarkBackend,
    BenchmarkConfig,
    RecordOutcome,
    RunStatus,
    preflight,
    run_benchmark,
    write_artifact,
)
from switchsure_ml.asr_adapters import FixtureAsrAdapter
from switchsure_ml.asr_contracts import (
    AsrEvidence,
    AsrProvenance,
    AsrRequest,
    LanguageEvidence,
    SegmentEvidence,
    TimeSpan,
    TranscriptAlternative,
)
from switchsure_ml.dataset_manifest import (
    DataRecord,
    DatasetManifest,
    DatasetSource,
    DataSplit,
    EvidenceStatus,
)


def _manifest(audio_digest: str) -> DatasetManifest:
    return DatasetManifest(
        manifest_id="fixture-manifest",
        schema_version="1.0",
        normalization_version=NORMALIZATION_VERSION,
        sources=(
            DatasetSource(
                dataset_id="consented-fixture",
                title="Consented fixture",
                version="1",
                source_uri="local://fixture",
                provenance="Generated only for unit testing",
                license_status=EvidenceStatus.VERIFIED,
                consent_status=EvidenceStatus.VERIFIED,
                license_expression="CC0-1.0",
                consent_statement="No human speech; synthetic test bytes",
            ),
        ),
        records=(
            DataRecord(
                record_id="record-secret-name",
                dataset_id="consented-fixture",
                audio_path="validation/private-audio.wav",
                audio_sha256=audio_digest,
                split=DataSplit.VALIDATION,
                speaker_id="speaker-fixture",
                session_id="session-fixture",
                duration_ms=2000,
                languages=("ta", "en"),
                transcript_path="validation/private-reference.txt",
            ),
        ),
    )


def _config(*, backend: BenchmarkBackend = BenchmarkBackend.FIXTURE) -> BenchmarkConfig:
    return BenchmarkConfig(
        run_id="fixture-smoke",
        backend=backend,
        model_id="fixture-asr" if backend is BenchmarkBackend.FIXTURE else "large-v3-turbo",
        model_revision="fixture-v1" if backend is BenchmarkBackend.FIXTURE else "UNPINNED",
        split=DataSplit.VALIDATION,
        seed=17,
        git_commit="abc123",
        environment_lock_hash="sha256:lock",
        fixture_smoke_only=backend is BenchmarkBackend.FIXTURE,
    )


def _evidence(transcript: str) -> AsrEvidence:
    segment = SegmentEvidence(surface=transcript, span=TimeSpan(0.0, 2.0))
    return AsrEvidence(
        transcript=transcript,
        segments=(segment,),
        language_evidence=(LanguageEvidence("ta"), LanguageEvidence("en")),
        alternatives=(TranscriptAlternative(transcript),),
        provenance=AsrProvenance("fixture", "fixture-asr", "fixture-v1"),
        warnings=("decoder scores are uncalibrated evidence",),
    )


def _clock(values: tuple[float, ...]) -> Iterator[float]:
    yield from values


def test_fixture_smoke_computes_metrics_without_persisting_content(tmp_path: Path) -> None:
    audio = b"not-real-audio"
    root = tmp_path / "data"
    folder = root / "validation"
    folder.mkdir(parents=True)
    (folder / "private-audio.wav").write_bytes(audio)
    reference = "எனக்கு Metformin 500 mg வேண்டும்"
    hypothesis = "எனக்கு Metformin 50 mg வேண்டும்"
    (folder / "private-reference.txt").write_text(reference, encoding="utf-8")
    manifest = _manifest(hashlib.sha256(audio).hexdigest())
    adapter = FixtureAsrAdapter({"record-secret-name": _evidence(hypothesis)})
    times = _clock((10.0, 10.2))

    artifact = run_benchmark(
        manifest,
        _config(),
        root,
        adapter,
        monotonic=lambda: next(times),
    )

    assert artifact.status is RunStatus.SMOKE
    assert artifact.aggregate is not None
    assert artifact.aggregate.normalized_wer == pytest.approx(0.2)
    assert artifact.aggregate.surface_wer == pytest.approx(0.2)
    assert artifact.aggregate.latency_p50_ms == pytest.approx(200.0)
    assert artifact.aggregate.real_time_factor == pytest.approx(0.1)
    assert artifact.records[0].outcome is RecordOutcome.SUCCESS

    serialized = artifact.to_json()
    for secret in (reference, hypothesis, "private-audio.wav", "record-secret-name"):
        assert secret not in serialized
    assert artifact.privacy_notice in serialized

    output = tmp_path / "artifacts" / "smoke.json"
    write_artifact(output, artifact)
    assert output.read_text(encoding="utf-8") == serialized
    assert not output.with_suffix(".json.tmp").exists()


def test_preflight_fails_closed_for_unapproved_real_execution(tmp_path: Path) -> None:
    manifest = _manifest("0" * 64)

    issues = preflight(manifest, _config(backend=BenchmarkBackend.FASTER_WHISPER), tmp_path)
    codes = {issue.code for issue in issues}

    assert "model_revision_unpinned" in codes
    assert "backend_missing" in codes
    assert "immutable_model_snapshot_required" in codes
    assert "missing_audio" in codes
    assert "missing_reference" in codes


def test_audio_hash_mismatch_is_a_retained_outcome(tmp_path: Path) -> None:
    root = tmp_path / "data"
    folder = root / "validation"
    folder.mkdir(parents=True)
    (folder / "private-audio.wav").write_bytes(b"tampered")
    (folder / "private-reference.txt").write_text("reference", encoding="utf-8")
    manifest = _manifest("0" * 64)
    adapter = FixtureAsrAdapter({"record-secret-name": _evidence("reference")})

    artifact = run_benchmark(manifest, _config(), root, adapter)

    assert artifact.status is RunStatus.SMOKE
    assert artifact.records[0].outcome is RecordOutcome.AUDIO_HASH_MISMATCH
    assert artifact.aggregate is not None
    assert artifact.aggregate.normalized_wer is None


def test_locked_test_execution_requires_explicit_authorization() -> None:
    with pytest.raises(ValueError, match="locked test"):
        BenchmarkConfig(
            run_id="forbidden-test",
            backend=BenchmarkBackend.FIXTURE,
            model_id="fixture",
            model_revision="fixture-v1",
            split=DataSplit.TEST,
            seed=17,
            git_commit="commit",
            environment_lock_hash="sha256:lock",
            fixture_smoke_only=True,
        )


def _write_cli_inputs(root: Path, *, adapter: str = "faster-whisper") -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").mkdir()
    (root / "uv.lock").write_text("locked", encoding="utf-8")
    config_folder = root / "configs" / "experiments"
    config_folder.mkdir(parents=True)
    config_path = config_folder / "baseline.toml"
    config_path.write_text(
        "\n".join(
            (
                'experiment_id = "cli-smoke"',
                "seed = 23",
                "[backbone]",
                f'adapter = "{adapter}"',
                'model_id = "large-v3-turbo"',
                'model_revision = "UNPINNED"',
                'device = "auto"',
                'compute_type = "auto"',
                "beam_size = 3",
                "word_timestamps = true",
                "vad_filter = true",
                "multilingual = true",
            )
        ),
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_id": "cli-manifest",
                "schema_version": "1.0",
                "normalization_version": NORMALIZATION_VERSION,
                "sources": [
                    {
                        "dataset_id": "approved-source",
                        "title": "Approved source",
                        "version": "immutable-v1",
                        "source_uri": "registry://approved-source",
                        "provenance": "Dated internal review",
                        "license_status": "verified",
                        "consent_status": "verified",
                        "license_expression": "Research agreement",
                        "consent_statement": "Declared ASR evaluation is permitted",
                    }
                ],
                "records": [
                    {
                        "record_id": "private-record",
                        "dataset_id": "approved-source",
                        "audio_path": "validation/audio.wav",
                        "audio_sha256": "a" * 64,
                        "split": "validation",
                        "speaker_id": "speaker-1",
                        "session_id": "session-1",
                        "duration_ms": 1000,
                        "languages": ["ta", "en"],
                        "transcript_path": "validation/reference.json",
                        "metadata": {"noise_band": "quiet"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path, manifest_path


def test_cli_writes_a_redacted_blocked_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, manifest_path = _write_cli_inputs(tmp_path)
    output = tmp_path / "artifacts" / "blocked.json"
    monkeypatch.setattr(benchmark_cli, "_git_commit", lambda _root: "deadbeef")

    exit_code = benchmark_cli.main(
        (
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(tmp_path / "approved-data"),
            "--output",
            str(output),
        )
    )

    assert exit_code == 2
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["status"] == "blocked"
    codes = {item["code"] for item in artifact["preflight_issues"]}
    assert {
        "backend_missing",
        "immutable_model_snapshot_required",
        "missing_audio",
        "missing_reference",
        "model_revision_unpinned",
    } <= codes
    assert artifact["git_commit"] == "deadbeef"
    serialized = output.read_text(encoding="utf-8")
    assert "private-record" not in serialized
    console = json.loads(capsys.readouterr().out)
    assert console["status"] == "blocked"


def test_cli_loads_exact_manifest_and_rejects_wrong_adapter(tmp_path: Path) -> None:
    config_path, manifest_path = _write_cli_inputs(tmp_path, adapter="unsupported")

    manifest = benchmark_cli._load_manifest(manifest_path)
    assert manifest.manifest_id == "cli-manifest"
    assert manifest.records[0].metadata == (("noise_band", "quiet"),)
    with pytest.raises(ValueError, match="only faster-whisper"):
        benchmark_cli._build_config(
            config_path,
            split=DataSplit.VALIDATION,
            allow_locked_test=False,
        )


def test_cli_rejects_malformed_manifest(tmp_path: Path) -> None:
    malformed = tmp_path / "manifest.json"
    malformed.write_text('{"sources": "not-a-list"}', encoding="utf-8")

    with pytest.raises(ValueError, match="sources must be a list"):
        benchmark_cli._load_manifest(malformed)


def test_benchmark_configuration_rejects_unsafe_values() -> None:
    with pytest.raises(ValueError, match="provenance"):
        replace(_config(), run_id="")
    with pytest.raises(ValueError, match="beam_size"):
        replace(_config(), beam_size=0)
    with pytest.raises(ValueError, match="smoke-only"):
        replace(_config(), fixture_smoke_only=False)


def test_model_snapshot_hash_and_accelerator_probe_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    with pytest.raises(ValueError, match="no files"):
        benchmark.hash_directory(snapshot)
    (snapshot / "model.bin").write_bytes(b"weights")
    first = benchmark.hash_directory(snapshot)
    assert len(first) == 64
    assert first == benchmark.hash_directory(snapshot)

    monkeypatch.setattr(
        "switchsure_eval.whisper_benchmark.shutil.which", lambda _name: "nvidia-smi"
    )
    monkeypatch.setattr(
        "switchsure_eval.whisper_benchmark.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="GPU One\nGPU Two\n"),
    )
    assert benchmark._accelerator_name() == "GPU One, GPU Two"

    def fail_probe(*_args: object, **_kwargs: object) -> object:
        raise OSError("probe unavailable")

    monkeypatch.setattr("switchsure_eval.whisper_benchmark.subprocess.run", fail_probe)
    assert benchmark._accelerator_name() is None


@pytest.mark.parametrize("error", [TimeoutError(), RuntimeError("decoder failed")])
def test_decoder_failures_remain_explicit_outcomes(tmp_path: Path, error: Exception) -> None:
    audio = b"test-audio"
    root = tmp_path / "data"
    folder = root / "validation"
    folder.mkdir(parents=True)
    (folder / "private-audio.wav").write_bytes(audio)
    (folder / "private-reference.txt").write_text("reference", encoding="utf-8")
    manifest = _manifest(hashlib.sha256(audio).hexdigest())

    class FailingAdapter:
        def transcribe(self, request: AsrRequest) -> AsrEvidence:
            del request
            raise error

    artifact = run_benchmark(manifest, _config(), root, FailingAdapter())
    if isinstance(error, TimeoutError):
        expected = RecordOutcome.TIMEOUT
    else:
        expected = RecordOutcome.DECODER_FAILURE
    assert artifact.records[0].outcome is expected


def test_invalid_reference_and_blocked_execution_are_retained(tmp_path: Path) -> None:
    audio = b"test-audio"
    root = tmp_path / "data"
    folder = root / "validation"
    folder.mkdir(parents=True)
    (folder / "private-audio.wav").write_bytes(audio)
    (folder / "private-reference.txt").write_text("", encoding="utf-8")
    manifest = _manifest(hashlib.sha256(audio).hexdigest())
    adapter = FixtureAsrAdapter({"record-secret-name": _evidence("reference")})

    artifact = run_benchmark(manifest, _config(), root, adapter)
    assert artifact.records[0].outcome is RecordOutcome.INVALID_REFERENCE

    blocked = run_benchmark(
        manifest,
        _config(backend=BenchmarkBackend.FASTER_WHISPER),
        root,
        adapter,
    )
    assert blocked.status is RunStatus.BLOCKED


def test_cli_low_level_validation_and_snapshot_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest_file = tmp_path / "lock"
    digest_file.write_bytes(b"locked")
    assert benchmark_cli._sha256_file(digest_file).startswith("sha256:")
    with pytest.raises(ValueError, match="must be a mapping"):
        benchmark_cli._mapping([], "value")
    with pytest.raises(ValueError, match="must be a string"):
        benchmark_cli._text(3, "value")
    assert benchmark_cli._optional_text(None, "value") is None
    with pytest.raises(ValueError, match="repository root"):
        benchmark_cli._repo_root(tmp_path / "missing" / "config.toml")

    config_path, _ = _write_cli_inputs(tmp_path / "repo")
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text + '\nlocal_snapshot_path = "artifacts/model"\n', encoding="utf-8")
    monkeypatch.setattr(benchmark_cli, "_git_commit", lambda _root: "deadbeef")
    config = benchmark_cli._build_config(
        config_path,
        split=DataSplit.VALIDATION,
        allow_locked_test=False,
    )
    assert config.local_model_snapshot == (tmp_path / "repo" / "artifacts" / "model").resolve()
