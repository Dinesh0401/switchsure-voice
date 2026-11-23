# Phase 1 dataset inventory and manifest template

## Governance status

This is a discovery inventory, not permission to download, train on, redistribute, or publish any
dataset. Every external candidate below is **unverified** until a dated review records the exact
dataset version, authoritative terms, consent basis, provenance, speaker/session identifiers, and
the intended use. A model-card license, paper-access license, or repository license does not by
itself establish permission for the underlying speech data.

No data has been downloaded as part of this inventory.

## Candidate inventory

| Candidate | Potential role | Tamil-English code-switch evidence | License | Consent / participant terms | Speaker/session IDs | Decision |
|---|---|---|---|---|---|---|
| Purpose-built SwitchSure pilot | Primary natural Tamil-English target set with critical entities and accessibility tasks | Must be collected and annotated | Not yet drafted | Not yet drafted; requires accessible informed consent and withdrawal/retention policy | Must be assigned before collection | Blocked pending protocol and ethics/privacy review |
| [MUCS challenge data](https://arxiv.org/abs/2104.00235) | Indian multilingual/code-switch benchmark candidate | Paper indicates Indian multilingual and code-switching tasks; exact Tamil-English subset must be verified | **Unknown / unverified** | **Unknown / unverified** | **Unknown / unverified** | Discovery only |
| [Mozilla Common Voice Tamil](https://commonvoice.mozilla.org/en/datasets) | Tamil monolingual control and acoustic diversity candidate | Not assumed to be code-switched | **Version-specific terms unverified** | **Version-specific collection/consent terms unverified** | Availability and stability unverified | Discovery only |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) Tamil (`ta_in`) | Tamil monolingual control and adapter smoke tests | Not assumed to be code-switched | **Dataset/audio terms unverified for intended use** | **Collection and participant terms unverified** | Availability and stability unverified | Discovery only |
| [IndicVoices](https://indicvoices.ai4bharat.org/) | Natural Indian-language speech and acoustic-control candidate | Tamil availability and natural Tamil-English switching must be verified | **Version-specific terms unverified** | **Collection and participant terms unverified** | Availability and stability unverified | Discovery only |
| [LT-EDI Tamil ASR 2025](https://aclanthology.org/2025.ltedi-1.16/) resources | Accessibility-oriented Tamil evaluation candidate, including declared speaker slices | Dataset availability and English switching must be verified | **Unknown / unverified** | **Unknown / unverified** | **Unknown / unverified** | Paper evidence only; no data use approved |

The Phase 1 baseline should combine a verified natural Tamil-English target set with verified Tamil
and English monolingual controls. Synthetic switching may be used only as a separately labeled
training ablation; it must never enter the locked real-speech test set.

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
