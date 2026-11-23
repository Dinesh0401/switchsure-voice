# Phase 1 dataset inventory and manifest template

## Governance status (reviewed 2026-11-23)

The machine-readable decision record is
[`configs/data/dataset-approvals.toml`](../../configs/data/dataset-approvals.toml). Approval is scoped
to an exact version and experiment role; it never implies that access has been granted. No external
audio was downloaded during this review.

Two real-speech sources are approved **only for monolingual controls**: Common Voice Scripted Speech
27.0 Tamil and English. No natural Tamil-English source has yet cleared access, consent, annotation,
speaker/session, and immutable-version review, so the locked code-switch benchmark remains blocked.
This is a governance result, not missing paperwork that may be bypassed.

## Candidate inventory

| Candidate | Potential role | Tamil-English evidence | Governance evidence | Speaker/session suitability | Decision |
|---|---|---|---|---|---|
| Purpose-built SwitchSure pilot | Critical-entity and accessibility target set | Must be collected | Protocol, consent, withdrawal and retention terms not drafted | Must be assigned before collection | **Blocked** |
| [Common Voice 27 Tamil](https://mozilladatacollective.com/datasets/cmu5x7cbc00dqo1071wdidlpf) | Tamil real-speech control | Monolingual read speech, not code-switched | CC0-1.0; contribution terms reviewed; no redistribution/re-identification | 984 speakers reported; regenerate disjoint splits from delivered metadata | **Approved: monolingual control only** |
| [Common Voice 27 English](https://mozilladatacollective.com/datasets?locale=en&q=Common+Voice+Scripted+Speech) | English real-speech control | Monolingual read speech, not code-switched | CC0-1.0; same Common Voice restrictions | Must pin delivered metadata and prefer declared South-Asian accent controls | **Approved: monolingual control only** |
| [Zhagaram Tamil/Tanglish](https://huggingface.co/datasets/zhagaramlabs/tamil-conversational-speech) | Natural target corpus | Card claims 603.3 hours of Tanglish | Gated custom license; public evidence does not complete consent/use review | Conversation IDs are described; stable speakers and human references need inspection | **Conditional; blocked pending access and review** |
| [Tamil-English podcast diarization](https://huggingface.co/datasets/Rangasuthan/tamil-english-podcast-diarization) | Exploratory natural audio | Two code-mixed podcasts | Card mixes CC-BY-4.0 with research/non-commercial restrictions; reference transcripts are not contracted | Two recordings, narrow domain, unsuitable independence | **Rejected for ASR benchmark** |
| [MUCS 2021](https://navana-tech.github.io/MUCS2021/data.html) Tamil | Previously proposed benchmark | Authoritative page/paper make Tamil monolingual; only Hindi-English and Bengali-English are code-switched | Dataset terms do not cure the language-pair mismatch | Not evaluated because target role is invalid | **Rejected as Tamil-English code-switch data** |
| [IMDA National Speech Corpus](https://www.imda.gov.sg/-/media/imda/files/industry-development/infrastructure/technology/digital-services-lab/2018-11-22-fact-sheet-for-ai-open-source-subfinal-v3_2.pdf) conversational collection | Natural target candidate | Secondary descriptions indicate English/Tamil switching may exist | Singapore Open Data License evidence found; exact acquired part and participant terms not inspected | Requires exact package, speaker/session, transcript audit | **Conditional; blocked pending official access** |
| [Controlled-access 2026 Tamil-English corpus](https://doi.org/10.1186/s13636-026-00477-y) | Natural target candidate | Paper describes 10 hours / 1,034 real utterances | Institutional approval and DUA required; artifact absent | Paper claims speaker-disjoint data; delivery still must be audited | **Conditional; blocked pending DUA and artifact** |
| FLEURS, IndicVoices, LT-EDI resources | Auxiliary candidates | Not established as natural Tamil-English switching | Exact release/consent review incomplete | Not reviewed | **Unverified; no use approved** |

The Phase 1 baseline must combine an approved natural Tamil-English target set with the approved
Tamil and English controls. Synthetic switching may be used only as a separately labelled training
ablation; it must never enter the locked real-speech test set. Public availability, a repository
license, or a paper claim alone is insufficient for approval.

## Required manifest entry

Each source must record:

```yaml
dataset_id: stable-local-identifier
title: exact-dataset-title
version: immutable-version-or-snapshot
source_uri: authoritative-source-or-internal-registry-entry
provenance: dated-review-record-and-retrieval-chain
license_status: unverified  # verified | unverified | rejected
license_expression: null    # exact version-specific expression/evidence when verified
consent_status: unverified  # verified | unverified | rejected
consent_statement: null     # reviewed evidence and allowed purpose when verified
```

Each audio record must record:

```yaml
record_id: immutable-record-identifier
dataset_id: stable-local-identifier
audio_path: relative/posix/path.wav
audio_sha256: lowercase-64-character-digest
split: train  # train | calibration | validation | test
speaker_id: stable-source-scoped-speaker-id
session_id: stable-source-scoped-session-id
duration_ms: 1250
languages: [ta, en]
transcript_path: relative/posix/transcript.json
metadata:  # sorted string keys; omit sensitive attributes unless approved and necessary
  device: phone
  noise_band: quiet
```

## Admission checklist

A source can move from discovery to experiment-ready only when all of these are recorded:

1. Exact version and immutable provenance are pinned.
2. License evidence is reviewed for training, evaluation, derived artifacts, and redistribution.
3. Consent and participant terms are reviewed for the declared research and retention policy.
4. Speaker and recording-session identifiers are present or can be created without identity leakage.
5. Direct identifiers and unnecessary sensitive metadata are removed or access-controlled.
6. Audio and annotations have content hashes; exclusions and known blind spots are documented.
7. Train, calibration, validation, and locked-test speakers and sessions are disjoint.
8. Split and complete-manifest hashes are captured before training.

If any item remains unknown, keep its status `unverified`; strict manifest validation must block the
experiment rather than converting missing evidence into implied approval.
