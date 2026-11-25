# ADR-0004: Ephemeral job content with durable lifecycle metadata

- Status: accepted
- Date: 2026-11-25

## Context

Pilot inference needs bounded backpressure, retries, cancellation, ownership checks, and recovery
after API or worker restarts. Persisting raw audio or transcript-bearing job payloads by default
would violate SwitchSure's data-minimization contract and would introduce encryption, access, and
deletion obligations before the controlled pilot has earned them.

## Decision

Persist only content-free job lifecycle metadata in SQLite: opaque job ID, hashed owner and
idempotency identifiers, state, timestamps, retry count, retention class, lease deadline, and safe
error code. Keep payloads and results in process memory with redacted representations. Bound active
jobs, use expiring worker leases, require a fresh payload for retry, and delete terminal metadata
after its TTL. On process restart, fail any job whose ephemeral payload or result is unavailable.

This SQLite implementation is the modular-monolith pilot adapter. A later Redis/PostgreSQL worker
boundary must preserve the same domain contract and receive a separate security review.

## Consequences

- Queue overload is explicit instead of consuming unbounded memory.
- Metadata survives restart without retaining audio, transcripts, or entities.
- A restart may require the user to retry because sensitive content is intentionally unavailable.
- Cross-process inference is not yet supported; production GPU workers require an encrypted,
  access-controlled payload transport and verified deletion design.
