---
name: build-switchsure-api
description: Build and review SwitchSure's Python/FastAPI delivery layer and service orchestration. Use when changing API endpoints, audio upload and validation, typed transcript contracts, adapters, persistence, queues, errors, privacy-safe telemetry, or backend tests under apps/api and packages/switchsure-core.
---

# Build the SwitchSure API

## Start with the contracts

Read `../../../docs/architecture/system-design.md`, `../../../docs/security/threat-model.md`, and the relevant ADRs. The core domain must not import FastAPI, model SDKs, database clients, or queue clients.

## Workflow

1. Define or update the domain command/result and port.
2. Write domain and contract tests before transport wiring.
3. Add the application orchestration with explicit timeouts and cancellation.
4. Map transport input to the command and domain failures to stable error codes.
5. Add privacy-safe metrics and traces using an allowlist.
6. Exercise success, invalid input, unavailable model, low evidence, timeout, cancellation, and retry behavior.

## API rules

- Use Pydantic models with strict bounds and explicit enums.
- Validate media type by content, byte limit, duration, channel count, and sample rate policy.
- Stream uploads to a bounded temporary file; do not read unbounded bodies into memory.
- Generate opaque random job IDs. Never derive identifiers from audio or transcript content.
- Return typed errors with a request ID and safe user message; keep internal details out of responses.
- Version external contracts deliberately. Do not leak model-specific response objects.
- Make idempotency and ownership checks explicit before adding durable jobs.

## Privacy and reliability rules

- Never log audio, transcript text, entity values, user lexicons, file names, or tokens.
- Delete temporary audio in a `finally` path and test deletion.
- Bound model calls with timeout, concurrency, and backpressure.
- Prefer explicit `defer` or typed fallback over fabricated content.
- Track model, calibrator, and policy versions independently.
- Keep health/readiness probes content-free and dependency-aware.

## Done criteria

Run unit, contract, and API tests; verify OpenAPI shape; inspect logs for content leakage; verify cleanup on success and failure; and record any architecture decision that changes a trust boundary.
