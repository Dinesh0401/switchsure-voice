---
name: develop-switchsure-ml
description: Design and implement SwitchSure multilingual speech, token language-ID, switch-boundary, confidence calibration, candidate-lattice entity, and selective-prediction research. Use for model adapters, training, inference, data preparation, active learning, calibration, ablations, and ML code under packages/switchsure-ml.
---

# Develop SwitchSure ML

## Load the research contract

Read `../../../docs/research/mathematical-methods.md`, `../../../docs/research/evaluation-contract.md`, and `../../../docs/research/evidence-base.md`. Read `references/experiment-checklist.md` before running an experiment.

## Workflow

1. State a falsifiable hypothesis, target user failure, primary metric, baseline, and stopping rule.
2. Verify dataset license, consent, provenance, speaker/session IDs, and split hashes.
3. Establish the simplest reproducible baseline before adding a new method.
4. Implement through model-neutral ports and emit the shared evidence contract.
5. Fit learned components on training data, select hyperparameters on validation data, and fit calibration only on the speaker-disjoint calibration split.
6. Evaluate on the locked test set once per declared comparison; preserve per-example outputs allowed by policy.
7. Run ablations and inspect switch, entity, acoustic, and demographic slices.
8. Publish the configuration, seed, revisions, environment lock, hardware, latency, memory, and limitations.

## Modeling rules

- Treat decoder scores as features, not calibrated confidence.
- Separate recognition, token-language, boundary, entity, calibration, and action-policy targets.
- Include an ambiguous/neutral language state for names and shared tokens.
- Compare boundary-aware features against matched non-boundary controls.
- Preserve N-best/repeated-decoding evidence and timestamps when available.
- Keep entity surface text immutable; canonicalization is a typed, reversible step.
- Route ambiguous critical values to review; do not use a generative model to silently resolve them.
- Use budgeted second-pass inference only when its risk reduction justifies latency.

## Statistical rules

- Use speaker-level paired bootstrap confidence intervals.
- Report absolute counts, effect sizes, and uncertainty.
- Use at least three seeds for trained heads unless cost is justified and disclosed.
- Report NLL, Brier, adaptive ECE, maximum calibration error, and risk-coverage; never select from ECE alone.
- Check conditional coverage before making a conformal claim.
- Do not tune on subgroup test results or repeatedly inspect the locked set.

## Done criteria

An experiment is done when another machine can reproduce it from a manifest, failures are categorized, baseline and ablations are present, leakage checks pass, and the conclusion says what was disproved as clearly as what improved.
