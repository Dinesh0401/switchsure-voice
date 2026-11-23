# ADR-0002: Risk-bearing transcript contract

- Status: accepted
- Date: 2026-11-20

## Context

A plain transcript string discards the evidence needed to understand code switches, calibration, critical entities, and user review.

## Decision

The canonical output is a typed transcript containing word timestamps, language posteriors, recognition and boundary risks, entity evidence, alternatives, provenance, and a policy action. Surface forms remain immutable; normalized values are linked fields.

## Consequences

- Product, evaluation, and audit use the same semantic object.
- Storage and API payloads are larger but interpretable.
- Adapters must map model-specific outputs into the common evidence contract.
