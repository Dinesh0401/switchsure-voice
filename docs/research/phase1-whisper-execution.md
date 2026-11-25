# Phase 1 Whisper benchmark execution

## Current evidence status

No real Tamil-English Whisper score has been produced in this workspace. On 2026-11-23 the local
preflight found all of the following blockers:

- the dataset inventory remains discovery-only; no strictly approved manifest or speech files are
  present;
- the registered model revision is `UNPINNED` and no content-addressed local CTranslate2 snapshot
  is configured;
- `faster-whisper`, `transformers`, and `torch` are not installed;
- no NVIDIA accelerator is visible to the local environment.

These are execution blockers, not failed model results. The deterministic fixture smoke test is
interface evidence only and must never be reported as model quality.

## Fail-closed runner

The runner validates governance, speaker/session separation, normalization version, selected-split
availability, audio hashes, reference availability, backend installation, and the model snapshot
hash before inference. It refuses an unpinned model and requires an explicit flag before accessing
the locked test split.

Run validation first; do not use the locked test split for setup:

```powershell
uv run python -m switchsure_eval.whisper_benchmark_cli `
  --config configs/experiments/phase1-baseline.toml `
  --manifest artifacts/manifests/phase1-manifest.json `
  --approvals configs/data/dataset-approvals.toml `
  --data-root data/approved/phase1 `
  --output artifacts/benchmarks/phase1-validation.json `
  --split validation
```

The JSON artifact captures the git commit, environment-lock hash, dataset/split hashes, model ID
and revision, decoding configuration hash, hardware, backend version, failures, normalized and
surface WER, normalized CER, p50/p95/cold-start latency, real-time factor, and peak process memory
when the operating system exposes it.

## Privacy behavior

References and hypotheses exist only in process memory. Artifacts contain salted record keys and
edit counts, never audio paths, record IDs, transcripts, hypotheses, or entity surfaces. Console
output contains only run status, record count, and blocker codes. Decoder errors are categorized
without persisting exception text.

## Conditions for a scientific result

1. Approve an exact version of a Tamil-English source for `code_switch_evaluation` in the
   machine-readable approval registry (and separately for `locked_test` before test access).
2. Materialize a strict manifest with verified license and consent evidence, hashes, and
   speaker/session-disjoint train, calibration, validation, and locked-test partitions.
3. Install and lock the backend environment.
4. Materialize a local model snapshot and set `model_revision` to its `sha256:<tree digest>`.
5. Run the validation split, resolve only pipeline failures, then authorize the one preregistered
   locked-test access. Do not use test results to choose decoding or normalization settings.

The runner currently reports corpus-level recognition and systems metrics. Switch-region, entity,
subgroup, uncertainty, and confidence-interval reports require the approved alignment/entity
annotations declared in the evaluation contract; absent annotations remain explicitly unevaluable.
