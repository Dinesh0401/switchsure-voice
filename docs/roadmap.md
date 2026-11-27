# Roadmap

## Phase 0 — Contracts and safe vertical slice ✅

- Freeze domain and API contracts for a risk-bearing transcript.
- Implement deterministic baseline risk aggregation and entity policy.
- Ship an accessible browser workflow using a deterministic demo adapter.
- Establish unit, contract, accessibility, and architecture checks.

## Phase 1 — Reproducible baseline study 🟡

Engineering status: manifests, leakage checks, adapter contracts, fixture-tested optional backends,
normalization, preregistration, bootstrap comparison, error taxonomy, release-gate evaluation, and a
privacy-safe fail-closed benchmark runner are implemented. Common Voice v27 Tamil/English are
approved only for monolingual controls. Scientific status: awaiting approved natural Tamil-English
data, a pinned local model snapshot, declared reference hardware, and the single locked-test run.

- Complete dataset license/consent inventory and manifests.
- Build speaker-disjoint Tamil-English and monolingual control splits.
- Integrate faster-whisper and Hugging Face Whisper evidence adapters.
- Produce zero-shot baseline, error taxonomy, and subgroup report.

## Phase 2 — Boundary-aware uncertainty

Engineering status: dependency-light class-balanced boundary/error heads, baseline ablation,
calibration-only temperature scaling, split-isolation checks, and matched controls are implemented
and preregistered. Scientific status: no real-corpus training or three-seed comparison yet.

- Annotate or derive token LID and switch boundaries.
- Train token-LID/boundary head and word-confidence estimator.
- Calibrate on held-out speakers and evaluate risk-coverage behavior.
- Run feature and calibration ablations.

## Phase 3 — Entity preservation

Engineering status: deterministic Tamil-English structured-value recognition, immutable surfaces,
reversible canonicalization, candidate-lattice evidence, and conservative review routing are
implemented. Scientific status: contextual-bias, distractor, per-type risk, and user-study results
remain pending approved data.

- Add structured-value recognizers and multilingual span model.
- Add candidate-lattice evidence, phonetic matching, and contextual biasing.
- Evaluate distractor robustness and per-type critical-entity risk.
- Tune focused clarification policy with user studies.

## Phase 4 — Pilot-quality product

Engineering status: the modular-monolith pilot now has a bounded asynchronous queue, durable
content-free SQLite lifecycle metadata, ephemeral payload/result retention, TTL deletion, worker
leases, cancellation, retry with fresh payload, typed errors, capability-scoped anonymous access,
fixed-cardinality content-free telemetry, and accessible queued/running/error/retry/cancel/typed
fallback states. Pilot status: not deployable yet; a real GPU worker, authenticated authorization,
encrypted cross-process payload transport, OpenTelemetry export, full-service load/security tests,
and manual WCAG/assistive-technology verification remain pending.

- GPU worker and bounded queue, durable metadata, explicit retention.
- Full WCAG 2.2 AA review and assistive-technology testing.
- Security/privacy review, observability, load tests, canary evaluation.
- Add language pairs only after dataset and release gates are satisfied.

## Phase 5 — Research publication artifact

- Pre-register primary hypotheses and locked-test protocol.
- Release permitted code, manifests, model cards, and evaluation outputs.
- Document negative results, limitations, and deployment constraints.
