# API review checklist

- Domain layer has no framework, model, storage, or transport import.
- Payload sizes, strings, collections, audio duration, and timeouts are bounded.
- Error codes are stable, documented, and covered by tests.
- Temporary data is deleted on success, failure, timeout, and cancellation.
- Logs and traces use an allowlist and contain no user content.
- Requests carry opaque correlation IDs and do not expose stack traces.
- Model, calibrator, policy, and schema versions are returned where reproducibility needs them.
- Retries cannot duplicate a user-visible high-stakes action.
- Readiness fails when a mandatory dependency cannot serve traffic.
- Rate limiting and ownership are addressed before public deployment.
