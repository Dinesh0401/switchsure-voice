# Security and privacy threat model

## Assets

- raw voice recordings and voiceprints;
- transcripts and critical entities;
- user correction history and optional lexicons;
- model, calibration, and evaluation artifacts;
- API credentials and deployment configuration.

## Trust boundaries

Browser input crosses into the API, media decoding crosses into native libraries, the API crosses into model workers and storage, and research artifacts cross from raw/annotated data into evaluation outputs.

## Principal threats and controls

| Threat | Primary controls |
|---|---|
| Malicious or oversized media | allowlisted types, byte/duration limits, content sniffing, sandboxed decode, timeouts |
| Audio/transcript leakage through logs | structured content-free logs, field allowlist, telemetry tests |
| Unauthorized retention | ephemeral default, explicit consent, TTL deletion, deletion audit |
| Prompt/context injection through transcript | treat transcript as data, no tool authority, schema-bound post-processing |
| Entity hallucination or silent rewrite | immutable evidence spans, provenance, alternatives, user confirmation |
| Model or dataset supply-chain compromise | pinned revisions and hashes, signed artifacts where possible, license/data manifests |
| Membership or speaker inference | no speaker-identification feature, minimize embeddings, access control, deletion |
| Cross-user job access | opaque random IDs, authenticated ownership checks when accounts exist |
| Denial of service | quotas, bounded concurrency, backpressure, cancellation, upload streaming |
| Unsafe automation from a wrong transcript | accept/review/defer policy and downstream confirmation boundary |

## Privacy defaults

- Do not persist raw audio.
- Do not log transcript or entity values.
- Do not train on user audio without separate, revocable, purpose-specific consent.
- Store only the minimum product state and attach a retention class to every stored object.
- Prefer aggregate metrics with privacy review for small demographic slices.

## Required pre-pilot work

- data-flow inventory and retention matrix;
- dependency and media-decoder security review;
- authentication/authorization design;
- encryption and key-management design;
- abuse, rate-limit, and incident-response runbooks;
- privacy notice and research consent reviewed for the deployment jurisdiction;
- red-team tests for malformed audio, transcript injection, cross-user access, and telemetry leakage.
