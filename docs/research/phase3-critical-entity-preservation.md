# Phase 3: critical-entity preservation contract

## Research claim under test

**Falsifiable hypothesis:** a time-aligned candidate lattice plus deterministic
structured-entity guard will reduce critical-entity error rate relative to the
same ASR top-1 transcript without the guard, without increasing matched general
speech WER by more than the preregistered limit.

The target user failure is a fluent-looking transcript that silently changes a
phone number, identifier, dosage, date, time, quantity, or money amount. The
primary metric is critical-entity error rate. Secondary measures are exact and
normalized entity value accuracy, critical-entity recall, high-severity accepted
error rate, review burden, and p95 latency. Phase 3 stops if its speaker-level
paired confidence interval does not support the preregistered improvement or if
review burden or latency exceeds its product gate.

This document specifies an engineering mechanism, not a measured improvement.
No locked-test result has been produced yet.

## Safety invariants

1. Captured surface text is immutable and remains attached to timestamps,
   hypothesis provenance, recognizer revision, and canonicalization trace.
2. Canonicalization creates a separate typed value. Every transformation records
   before/after values and is exactly reversible to the captured surface.
3. Relative hypothesis weights, normalized candidate support, edge weights, and
   decoder scores are uncalibrated evidence—not probabilities of correctness.
4. Different overlapping critical values are all returned to review. A larger
   decoder weight cannot silently discard or rewrite an alternative.
5. Invalid locale-dependent values and values from an unpinned model revision
   are routed to review.
6. Entity and transcript text is excluded from object representations to reduce
   accidental logging. Application logs must use opaque IDs and reason codes.
7. No generative model participates in structured canonicalization or ambiguity
   resolution.

## Implemented baseline

`switchsure_ml.entity_lattice` provides a dependency-free Phase-3 baseline:

- deterministic finite-state-style recognizers for Tamil/English decimal digits,
  dosage, quantity, phone, email, date, time, money, and identifiers;
- locale-explicit typed canonicalization, including Tamil digit and unit mapping;
- immutable ASR-hypothesis and pinned provenance contracts;
- time-overlap clustering across N-best or repeated-decoding hypotheses;
- evidence aggregation by canonical value without presenting support as confidence;
- weighted competing-value edges using temporal overlap and surface similarity;
- conservative preserve/review decisions that expose every competing value.

Candidate IDs, mention IDs, and decision IDs are deterministic hashes. They are
stable for reproducibility but must not be treated as anonymization of the source
data.

## Evaluation plan

Compare on the same speaker-disjoint locked examples:

1. ASR top-1 structured recognizer only;
2. candidate lattice with equal hypothesis weights;
3. candidate lattice with decoder-relative weights;
4. lattice without transliteration/script-aware features;
5. lattice without temporal alignment;
6. full guard plus the calibrated selective-prediction policy.

Report absolute counts and speaker-level paired bootstrap 95% confidence
intervals. Break results out by entity type, Tamil/English script pattern,
switch proximity, device/noise band, and declared demographic slices where
collection and sample size permit. Include distractor false positives, invalid
canonicalizations, competing-value review rate, and errors caused by missing or
incorrect timestamps.

The deterministic baseline intentionally does not recognize people,
organizations, locations, medicines, or addresses. Those types require a
separately evaluated multilingual span model and lexicon retrieval. Their output
must enter the same immutable lattice contract and may not overwrite a captured
surface.

## Product integration boundary

The lattice returns evidence and a conservative route; it does not make a
high-stakes product authorization. `preserve_surface` means retain the recognized
surface and provenance. `review` means present the smallest affected span and all
candidate surfaces through accessible confirmation UI. Domain policy may always
escalate either route to deferral.

Before any Phase-3 gain is claimed, the run must pin dataset/split/normalization,
model, code, and environment revisions; fit calibration only on the calibration
split; run the declared ablations; and pass the evaluation contract.
