# Roadmap

## Phase 0 — Contracts and safe vertical slice ✅

- Freeze domain and API contracts for a risk-bearing transcript.
- Implement deterministic baseline risk aggregation and entity policy.
- Ship an accessible browser workflow using a deterministic demo adapter.
- Establish unit, contract, accessibility, and architecture checks.

## Phase 1 — Reproducible baseline study 🟡

Engineering status: manifests, leakage checks, adapter contracts, fixture-tested optional backends,
normalization, preregistration, bootstrap comparison, error taxonomy, and release-gate evaluation are
implemented. Scientific status: awaiting verified data admission, pinned model revisions, declared
reference hardware, and the single locked-test baseline run.

- Complete dataset license/consent inventory and manifests.
- Build speaker-disjoint Tamil-English and monolingual control splits.
- Integrate faster-whisper and Hugging Face Whisper evidence adapters.
- Produce zero-shot baseline, error taxonomy, and subgroup report.

## Phase 2 — Boundary-aware uncertainty

- Annotate or derive token LID and switch boundaries.
- Train token-LID/boundary head and word-confidence estimator.
- Calibrate on held-out speakers and evaluate risk-coverage behavior.
- Run feature and calibration ablations.

## Phase 3 — Entity preservation

- Add structured-value recognizers and multilingual span model.
- Add candidate-lattice evidence, phonetic matching, and contextual biasing.
- Evaluate distractor robustness and per-type critical-entity risk.
- Tune focused clarification policy with user studies.

## Phase 4 — Pilot-quality product

- GPU worker and bounded queue, durable metadata, explicit retention.
- Full WCAG 2.2 AA review and assistive-technology testing.
- Security/privacy review, observability, load tests, canary evaluation.
- Add language pairs only after dataset and release gates are satisfied.

## Phase 5 — Research publication artifact

- Pre-register primary hypotheses and locked-test protocol.
- Release permitted code, manifests, model cards, and evaluation outputs.
- Document negative results, limitations, and deployment constraints.
