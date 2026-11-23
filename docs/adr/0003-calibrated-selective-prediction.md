# ADR-0003: Calibrated selective prediction

- Status: accepted
- Date: 2026-11-20

## Context

Modern speech decoders can be overconfident. A single fixed confidence threshold also ignores the higher cost of errors in critical entities.

## Decision

Never expose raw decoder probability as confidence. Fit a versioned post-hoc correctness model and calibrator on a held-out calibration set. Apply a cost-aware policy that accepts, reviews, or defers. Select thresholds from risk-coverage behavior and declared harm weights.

## Consequences

- Calibration data becomes a first-class artifact.
- Model, calibrator, and policy versions are independently tracked.
- The product can trade coverage for lower accepted risk in a measurable way.
