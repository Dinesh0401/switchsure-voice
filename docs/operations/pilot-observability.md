# Pilot observability contract

## Purpose

The controlled pilot needs enough operational evidence to detect latency, backpressure, unsafe
fallbacks, and model failures without collecting speech content. The in-process registry is a
bounded adapter for pilot and test environments; it is not a replacement for the Phase B
OpenTelemetry backend.

## Permitted metrics

| Metric family | Fixed dimensions | Values |
|---|---|---|
| Stage latency | pipeline stage | count, total milliseconds, maximum milliseconds |
| Decision action | `accept`, `review`, `defer` | count |
| Error | pipeline stage, safe error class | count |
| Queue | none | observations, current depth/capacity, high watermark, full observations |

Pipeline stages and error classes are enums. Arbitrary strings cannot become labels. Latency,
queue depth, and queue capacity are finite and bounded. The registry aggregates values and never
keeps an event history, so its number of series is independent of traffic volume.

## Forbidden data

Audio, transcripts, entity surfaces or canonical values, file names, and tokens are forbidden.
The event factory rejects those field names before checking the allowlist. It also rejects every
other field not required by the selected metric family. Request IDs, job IDs, owner identifiers,
user lexicons, payloads, results, exceptions, and free-form error messages are intentionally absent
from this adapter.

Callers must map failures to `SafeErrorClass`; they must not pass exception text. Version metadata
belongs in separately reviewed deployment resource attributes, not high-cardinality event labels.

## Load-safety properties

- Eight pipeline-stage buckets, three action buckets, and the Cartesian product of stage and safe
  error-class buckets are allocated once.
- Queue metrics occupy one aggregate series.
- No samples, content, identifiers, or arbitrary label values are retained.
- Recording and snapshots are protected by a process-local lock.
- Counters are monotonically increasing for the life of the process; the external metrics backend
  is responsible for scrape intervals and durable retention.

Before pilot deployment, wire this adapter at orchestration boundaries, export it through an
authenticated or network-restricted endpoint, and load-test the complete service. An export must
preserve this field allowlist and must not add request-derived resource attributes.
