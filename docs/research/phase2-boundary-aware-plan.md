# Phase 2 boundary-aware uncertainty plan

Status: **preregistered engineering implementation; no empirical result yet**.

## Decision and user failure

This study tests whether explicit switch-boundary evidence improves word-error
risk ranking over the same error head without boundary features. The product
failure is a critical Tamil or English word being silently accepted when it is
spoken near a language switch and the recognizer is unstable.

The primary metric is area under the risk-coverage curve (AURC). The primary
comparison is the boundary-aware error head versus the matched baseline error
head on the same examples. The hypothesis is supported only if speaker-level
paired confidence intervals show lower AURC without a material regression on
matched monolingual controls. The machine-readable preregistration is
[`configs/experiments/phase2-boundary-aware.toml`](../../configs/experiments/phase2-boundary-aware.toml).

## Model

The dependency-light reference implementation in `boundary_uncertainty.py`
contains two deterministic, L2-regularized logistic heads:

1. The boundary head uses token-LID entropy, adjacent-language transition
   evidence, timestamp gap, and repeated-decoding disagreement.
2. The word-error head uses decoder error evidence, LID entropy, disagreement,
   token length, and entity status. Its Phase 2 variant additionally receives
   boundary probability and normalized binary boundary entropy.

The baseline and boundary-aware variants use the same optimizer, labels, and
non-boundary feature set. This makes the boundary feature addition a matched
ablation rather than a comparison between unrelated models. Decoder evidence
is never described as calibrated confidence.

The compact heads are research baselines and portable domain contracts. A later
tensor-backed implementation may replace them behind the same output contract,
but it must reproduce the split and calibration safeguards.

## Leakage and calibration controls

- Both heads fit only on examples explicitly marked `train`.
- Temperature scaling fits once, only on the speaker- and session-disjoint
  `calibration` split.
- Validation selects hyperparameters and action thresholds; it never fits the
  calibrator.
- Locked test data is prediction-only and inspected once for this experiment.
- Duplicate example IDs, missing labels, single-class fit data, split mixing,
  and training/calibration identity overlap are hard failures.
- Per-token artifacts contain stable IDs and numeric evidence, not transcript
  surface text.

The current temperature scaler preserves ranking and calibrates word-error
logits. Boundary probabilities are still model scores until a separately
declared boundary-calibration study is run.

## Matched boundary controls

Gold boundary tokens are matched to unused non-boundary tokens within one split
on language, entity status, acoustic-quality band, and token-length band.
Same-speaker controls are preferred, then closest token length. Unmatched
boundaries are reported rather than dropped silently. Gold boundary labels are
used only for this offline analysis and never enter inference features.

The matched analysis reports boundary-versus-control error and entropy deltas,
along with absolute counts. It does not by itself establish causality; token
frequency, switch direction, and residual acoustic confounding remain declared
limitations.

## Required evaluation

- boundary precision, recall, F1, tolerance-window F1, and localization error;
- NLL, Brier score, adaptive ECE, maximum calibration error, AUROC/AUPRC, full
  risk-coverage curves, and AURC;
- matched boundary/control deltas;
- switch direction, entity type, acoustic band, device, and declared demographic
  slices with privacy suppression;
- at least three trained-head seeds;
- speaker-level paired bootstrap confidence intervals;
- latency and memory for feature extraction and both heads;
- the no-boundary ablation using the identical train/calibration/test records.

## Execution blockers

The experiment must not run against the locked set until the Tamil-English data
license and consent evidence are verified, split hashes exist, the Phase 1 ASR
artifact and model revision are pinned, and the three declared seeds have been
executed on recorded hardware. `UNPINNED` values in the configuration are
intentional blockers. This document reports engineering readiness, not improved
accuracy or product safety.
