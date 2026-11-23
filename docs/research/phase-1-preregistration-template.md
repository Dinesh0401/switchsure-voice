# Phase-1 baseline preregistration and report

Complete this document and serialize the matching `BaselineExperimentSpec` **before**
inspecting locked-test results. Replace every `TBD`; an unfilled field makes the result
non-releasable.

## Reproducibility identity

- Experiment ID: `TBD`
- Git commit: `TBD`
- Environment lock SHA-256: `TBD`
- Dataset manifest SHA-256: `TBD`
- Normalization version: `TBD`
- Train/calibration/validation/locked-test split hashes: `TBD`
- Model, calibrator, and policy revisions: `TBD`
- Seeds: `TBD` (at least three for trained heads or fine-tuning)
- Reference hardware and runtime: `TBD`

Speaker and session keys must be non-identifying, unique within a partition, and disjoint
across train, calibration, validation, and test. The test partition must not appear in the
tuning-partition declaration.

## Preregistered claim

- Primary system and comparator: `TBD`
- Primary metric and direction: `TBD`
- Practical-effect threshold: `TBD`
- Paired speaker-level bootstrap seed/resamples/confidence level: `TBD`
- Exploratory comparisons and multiplicity correction: `TBD`
- Declared subgroups and minimum unique-speaker suppression threshold: `TBD`
- P95 latency target and reference hardware: `TBD`

## Dataset and failure accounting

Record raw and retained example/speaker counts, licenses, consent constraints, exclusions,
and normalization decisions. Report counts for `success`, `missing`, `timeout`,
`invalid_audio`, `decoder_failure`, and `abstention`; none may be silently removed. Report
the declared error taxonomy, including entity detection/span/value errors, switch-region
errors, contextual-bias false positives, and high-severity accepted errors.

## Results

Report normalized and unnormalized recognition metrics, switching, critical-entity,
calibration, risk-coverage, latency, review-burden, and accessibility results with absolute
counts and denominators. Every comparison needs a paired speaker-level 95% interval. Every
subgroup row must show example count, unique-speaker count, denominator, outcome counts, and
whether its metric was privacy-suppressed.

Attach the machine-readable `ReleaseDecision`. A `fail` or `not_evaluable` gate remains
non-passing regardless of narrative context. Passing does not authorize autonomous
high-stakes use.

## Limitations and generalization

- Populations, languages, switch types, devices, and acoustic conditions not represented:
  `TBD`
- Distribution shifts or conformal assumptions not supported by evidence: `TBD`
- Observed evidence versus inference: `TBD`
- Mitigation for any major subgroup above the severity-ratio gate: `TBD`
