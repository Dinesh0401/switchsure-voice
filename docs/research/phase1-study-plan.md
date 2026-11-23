# Phase 1 baseline study plan

Status: **preregistered engineering contract; no empirical result yet**.

## Decision this study supports

The study decides whether a zero-shot multilingual Whisper backbone provides enough timestamped evidence to justify Phase 2 boundary-aware confidence modeling. It does not decide whether SwitchSure is safe for autonomous medical, financial, or identity workflows.

## Primary hypothesis

On speaker-disjoint Tamil-English speech, zero-shot recognition error within two words of a gold language-switch boundary is higher than error on matched non-boundary words after matching on token length, language, entity status, and acoustic-quality band.

- Real-user failure: a dosage, date, identifier, name, or amount near a language switch is silently changed.
- Primary metric: switch-region WER.
- Primary comparison: switch-region words versus matched non-boundary controls from the same declared evaluation population.
- Expected direction: switch-region WER is higher.
- Stopping rule: inspect the locked test set once for this experiment identifier. Any method change creates a new experiment and uses validation evidence until a future locked comparison is declared.

The machine-readable preregistration is [`configs/experiments/phase1-baseline.toml`](../../configs/experiments/phase1-baseline.toml).

## Data admission gates

An audio record is inadmissible until its manifest includes dataset revision, source URL, license identifier or exact terms reference, consent/provenance status, speaker ID, session ID, locale, duration, checksum, and split. Unknown license or consent status is a hard failure, not a warning.

Train, calibration, validation, and locked-test speakers and sessions must be disjoint. Synthetic speech can support training or stress tests but cannot enter the locked real-speech test set. Raw audio remains outside Git and is ephemeral unless the source terms and study protocol explicitly authorize retention.

## Baseline adapters

1. `faster-whisper` zero-shot with word timestamps, VAD, and multilingual decoding enabled.
2. Hugging Face Whisper pipeline with word timestamps as an independent evidence adapter.
3. The same declared backbone with a language hint, reported as a separate condition.

`faster-whisper` exposes word timestamps, segment log probability, no-speech probability, language probability, VAD, multilingual decoding, and hotword controls. Hugging Face Whisper exposes timestamped generation and word timestamps. These values are recorded as **uncalibrated evidence** and must not be shown as user-facing confidence ([faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Whisper documentation](https://huggingface.co/docs/transformers/model_doc/whisper)).

Every model and dataset revision must be pinned before the first declared run. `UNPINNED` in the configuration is an intentional execution blocker.

## Outputs

- immutable dataset and split hashes;
- per-example outcomes including timeout, invalid audio, missing prediction, and abstention;
- normalized and unnormalized WER/CER;
- switch-region, per-language, and monolingual-control metrics;
- critical-entity error taxonomy by type and severity;
- raw-score calibration diagnostics and risk-coverage curves;
- speaker-level paired bootstrap intervals;
- declared subgroup denominators and privacy suppression;
- p50/p95 latency, real-time factor, cold-start time, and peak memory;
- a machine-readable gate report and a human limitations section.

## Phase 1 completion boundary

Engineering readiness means adapters, manifests, leakage checks, metrics, and reproducible report generation pass offline tests. Scientific completion additionally requires admitted Tamil-English audio, pinned revisions, declared hardware, and one locked-test run. Without those artifacts, the repository may report “ready to run,” never “baseline validated.”
