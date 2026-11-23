# Product brief

## Problem

Multilingual speakers naturally switch languages within a sentence. Speech systems often fail precisely at those boundaries and can still present the result with unjustified confidence. Errors involving a medicine, dosage, person, address, account number, date, currency amount, or emergency term can be materially more harmful than ordinary word errors.

## Initial user

The first lane serves Tamil-English speakers using voice input in high-friction accessibility contexts such as healthcare, public services, education, and forms. The product must also work for users with non-standard accents, older speakers, transgender speakers, and noisy or low-bandwidth environments.

## Jobs to be done

- Convert multilingual or code-switched speech to an editable transcript.
- Make uncertainty understandable without exposing model jargon.
- Protect critical entities and let the user confirm them with minimal effort.
- Offer typed input and correction as equal alternatives to speech.
- Export a traceable, user-approved transcript without retaining raw audio by default.

## Product principles

1. **Calibrated honesty:** uncertainty is useful only when a stated confidence corresponds to observed error frequency.
2. **Risk over cosmetics:** prioritize harmful entity substitutions and deletions above punctuation or stylistic normalization.
3. **Selective automation:** accept safe spans, review risky spans, and defer when evidence is insufficient.
4. **User agency:** the user sees alternatives and makes the final choice for high-risk content.
5. **Language dignity:** preserve the speaker's language choices, script, and code-switching; do not force translation.
6. **Accessible by construction:** every speech action has a keyboard/text equivalent.

## Scope

### Phase 1

- uploaded or recorded short-form audio;
- Tamil-English transcription with word timestamps;
- switch-boundary and word-risk visualization;
- configurable critical-entity types;
- focused confirmation workflow;
- research-grade evaluation reports.

### Later

- low-latency streaming;
- more Indian and global language pairs;
- domain lexicons with contextual biasing;
- on-device or private-cloud inference;
- speech-to-speech accessibility modes.

## Explicit non-goals for the first release

- diagnosis, clinical decision support, or legal advice;
- speaker identification;
- emotion or protected-attribute inference;
- fully autonomous submission of high-stakes forms;
- claiming support for a language pair without a declared benchmark.
