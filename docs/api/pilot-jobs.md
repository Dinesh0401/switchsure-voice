# Pilot asynchronous job API

The pilot job API makes bounded backpressure, cancellation, retries, retention, and failure states
explicit while keeping transcript-bearing payloads and results ephemeral.

## Contract

1. Generate a cryptographically random 32-128 character capability for the browser session. Keep it
   in memory; do not put it in a URL or log.
2. Submit model-neutral evidence with `POST /v1/jobs/analyze-evidence` and the
   `X-Job-Capability` header. An optional `Idempotency-Key` prevents duplicate work for that
   capability.
3. Poll `GET /v1/jobs/{job_id}` with the same capability. States are `queued`, `running`,
   `succeeded`, `failed`, or `cancelled`.
4. Cancel queued/running work with `POST /v1/jobs/{job_id}/cancel`.
5. A failed job may be retried once by posting the complete fresh payload to
   `POST /v1/jobs/{job_id}/retry`. The service does not retain the original payload for retry.

Queue saturation returns `503` with `Retry-After`. Ownership mismatch and an unknown ID return the
same content-free `404`. Validation and operational errors contain a safe code, safe message, and
random request ID. They never include transcript text, entity values, filenames, or internal model
exceptions.

## Pilot configuration and limits

- `SWITCHSURE_JOB_DB_PATH` selects the SQLite metadata database. The default is process-local
  in-memory metadata for development; set an explicit protected path for a controlled pilot.
- The default processor is the deterministic evidence analyzer, not a deployed Whisper/GPU worker.
- A capability is an anonymous pilot isolation mechanism, not account authentication. Public or
  multi-user deployment still requires authenticated principals, authorization, rate limits,
  encrypted transport/storage, and an abuse review.
- Payloads and results are process-local. A restart changes affected jobs to a safe failed state and
  requires resubmission.
