---
name: evaluate-switchsure-research
description: Evaluate SwitchSure models and policies reproducibly. Use when computing ASR, code-switch, language-ID, critical-entity, calibration, selective-risk, fairness, latency, or user-burden metrics; comparing experiments; enforcing release gates; or generating research reports under packages/switchsure-eval.
---

# Evaluate SwitchSure research

## Load the evaluation contract

Read `../../../docs/research/evaluation-contract.md` and `../../../docs/research/mathematical-methods.md`. Use `references/report-checklist.md` before accepting a result.

## Workflow

1. Validate manifest, split hashes, schema, provenance, and normalization version.
2. Refuse evaluation when speaker/session leakage or forbidden test tuning is detected.
3. Align hypotheses and references once with a versioned normalization pipeline.
4. Compute recognition, switching, entity, uncertainty, risk-coverage, latency, and review-burden metrics.
5. Slice only declared, adequately sized groups and suppress privacy-risking cells.
6. Use paired speaker-level bootstrap intervals for comparisons.
7. Generate machine-readable results and a human report with counts, intervals, plots, failures, and limitations.
8. Apply release gates mechanically; do not convert a failed gate into a pass through prose.

## Metric rules

- Report normalized and unnormalized WER/CER together.
- Report switch-region performance separately from monolingual controls.
- Distinguish entity detection, span matching, normalized value accuracy, and accepted high-severity errors.
- Measure calibration with NLL, Brier, adaptive ECE, maximum calibration error, and reliability plots.
- Measure selective behavior with coverage, risk, AURC, and user review burden.
- Include latency distribution, cold start, real-time factor, and reference hardware.
- Record undefined metrics and empty denominators explicitly; never silently coerce them to zero.

## Comparison rules

- Pair examples and speakers across systems.
- Use the predeclared primary comparison and correct exploratory multiple comparisons.
- Separate statistical significance, practical effect, and product relevance.
- Treat missing examples, decoder failures, and timeouts as outcomes, not removable noise.

## Done criteria

The report is complete only when it is reproducible from pinned artifacts, includes confidence intervals and failure counts, shows release-gate status, and states where the result does not generalize.
