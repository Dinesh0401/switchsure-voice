# Phase 1 ASR baseline pre-registration

## Status

This document declares the first baseline study before model outputs are inspected. The adapter
implementation is an evidence-collection boundary, not evidence that either backend is safe or
accurate. Raw decoder probabilities and log probabilities remain uncalibrated features.

## Falsifiable hypothesis

On speaker-disjoint Tamil-English code-switched speech, at least one pinned Whisper baseline will
show materially higher word error rate within two words of a gold switch boundary than on matched
non-boundary words after matching language, entity status, token length, and acoustic-quality
bands.

The real-user failure is substitution or deletion of a critical medicine, dosage, date, time,
phone number, name, or identifier near a language switch while the product presents the transcript
without review.

## Declared comparison

- Baselines: pinned faster-whisper and Hugging Face Whisper revisions using matched decoding
  settings where the backends support them.
- Primary metric: switch-region WER minus matched non-boundary WER, with a speaker-level paired
  bootstrap 95% confidence interval.
- Safety metric: critical-entity error rate, reported by entity type and proximity to a switch.
- Evidence diagnostics: raw word/segment decoder scores, language evidence, timestamp availability,
  N-best availability, latency, peak memory, and backend failures.
- Expected direction: switch-region WER and critical-entity error rate are higher near switches.
- Stopping rule: reject the hypothesis if the confidence interval for the declared WER difference
  includes zero or reverses direction. Do not add a boundary model until dataset and alignment
  failures have been categorized and the baseline artifact is reproducible.

## Data gates before execution

1. Confirm license, consent, intended-use compatibility, and data retention policy.
2. Create train, calibration, validation, and locked-test partitions disjoint by speaker and, where
   speaker identity is uncertain, by recording session.
3. Pin dataset manifest, split hashes, normalization version, model revisions, environment lock,
   hardware, seeds, and decoding configuration.
4. Annotate or verify word spans, token language including ambiguous/neutral, switch boundaries,
   entity spans/types, criticality, and acceptable orthographic variants.
5. Keep raw audio ephemeral and never log transcript or critical-entity content.

## Adapter evidence contract

`AsrEvidencePort` accepts an `AsrRequest` whose audio bytes and local path are excluded from object
representations. An adapter returns immutable transcript, segment, word, token, language,
alternative, timestamp, score, and provenance records. Every `DecoderScore` is structurally marked
`calibrated = False`; calibration belongs to a later speaker-disjoint calibration stage.

The deterministic fixture adapter supports reproducible tests. Optional faster-whisper and Hugging
Face adapters import and initialize their heavy backends only on first inference. Missing packages
produce an installation-oriented error, while backend failures omit audio and transcript content.

## Evidence that must accompany results

- normalized and unnormalized WER, CER, language-specific WER, switch-region WER, and edit counts;
- entity value accuracy, entity WER, critical-entity recall, and high-severity accepted errors;
- aggregate and declared language/acoustic/entity slices with absolute counts;
- speaker-level paired bootstrap confidence intervals and per-example failure categories;
- p50/p95 latency, real-time factor, cold start, peak memory, backend/version provenance;
- negative results, unavailable evidence fields, invalid assumptions, and deployment limitations.

No locked-test result may be used to choose a model, decoder setting, normalization rule, threshold,
or subgroup definition.
