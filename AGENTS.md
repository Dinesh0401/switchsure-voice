# SwitchSure engineering contract

## Mission

Build an uncertainty-aware multilingual voice accessibility platform. Optimize for calibrated safety and critical-entity preservation, not transcript fluency alone.

## Non-negotiable rules

- Keep the domain layer independent of FastAPI, model vendors, and storage implementations.
- Treat raw decoder probabilities as uncalibrated evidence, never as user-facing confidence.
- Preserve the original surface form, timestamps, alternatives, provenance, and confidence for every critical entity.
- Never use an LLM to silently rewrite a critical entity.
- Require speaker-disjoint train, calibration, validation, and test partitions.
- Report aggregate metrics and subgroup slices. Do not claim improvement without confidence intervals.
- Keep raw audio ephemeral by default and never log transcript or entity content.
- Maintain keyboard, screen-reader, contrast, and non-speech interaction paths.
- Add or update tests with every behavioral change.

## Repository-local skills

Use the relevant skill under `.agents/skills/` before changing that area:

- `build-accessible-voice-ui`
- `build-switchsure-api`
- `develop-switchsure-ml`
- `evaluate-switchsure-research`

## Definition of done

A feature is done only when its contracts, tests, observability, privacy behavior, failure behavior, and evaluation impact are explicit.
