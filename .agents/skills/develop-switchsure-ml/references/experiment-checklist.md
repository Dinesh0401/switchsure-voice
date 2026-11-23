# Experiment checklist

- Hypothesis and real-user failure are explicit.
- Primary metric, baseline, expected direction, and stopping rule are declared.
- Dataset license/consent and data card are available.
- Train, calibration, validation, and locked-test speaker/session sets are disjoint.
- Dataset, split, normalization, model, code, and environment revisions are pinned.
- Seeds, hardware, precision, batch size, latency, and peak memory are captured.
- Calibration is fit after model selection and without test access.
- Aggregate, switch-region, entity-type, acoustic, and declared subgroup metrics are reported.
- Paired speaker bootstrap intervals and per-example error classes are produced.
- Negative results, failed assumptions, and deployment limitations are documented.
