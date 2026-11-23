# Evaluation contract

## 1. Purpose

This contract prevents a fluent demo from being mistaken for a reliable system. Results are valid only when data lineage, split policy, normalization, metrics, uncertainty, and subgroup slices are reproducible.

## 2. Data protocol

- Start with Tamil-English code-switched speech and matched Tamil and English monolingual controls.
- Use only datasets whose license and consent terms permit the intended experiment.
- Create speaker-disjoint train, calibration, validation, and locked test splits.
- Keep recording sessions disjoint when speaker IDs are uncertain.
- Annotate word spans, language tags, switch boundaries, entity spans/types, criticality, and acceptable orthographic variants.
- Retain code-switch topology: inter-sentential, intra-sentential, tag switch, and ambiguous/neutral token.
- Publish a data card with exclusions, demographics where ethically available, acoustic conditions, and known blind spots.
- Never place the locked test set in prompts, tuning loops, or threshold selection.

## 3. Baselines

1. Whisper/faster-whisper zero-shot baseline.
2. Same backbone with language hints and no entity guard.
3. Same backbone with contextual biasing only.
4. SwitchSure with boundary features but no entity policy.
5. Full SwitchSure with calibrated boundary and entity risk.
6. Optional SeamlessM4T or CTC/Conformer comparison where language coverage and compute permit.

## 4. Primary metrics

### Recognition

- normalized and unnormalized WER;
- CER for Tamil and mixed-script robustness;
- code-switch WER and per-language WER;
- switch-region WER within ±2 words of a gold boundary;
- insertion, deletion, and substitution rates.

### Switching

- token language-ID macro F1;
- switch-boundary precision, recall, F1, and tolerance-window F1;
- boundary localization mean absolute error in milliseconds;
- posterior entropy at boundary vs matched non-boundary positions.

### Critical entities

- exact and normalized entity span F1;
- critical-entity recall;
- entity word error rate;
- entity value accuracy by type;
- high-severity accepted error rate;
- contextual-bias distractor false-positive rate.

### Uncertainty and selective prediction

- negative log likelihood and Brier score;
- expected and maximum calibration error with declared binning;
- calibration plots overall, by language pair, and near/far from switches;
- AUROC/AUPRC for word error detection;
- area under the risk-coverage curve;
- risk at 50%, 75%, 90%, and 95% coverage;
- review burden: questions per minute and percent of utterances interrupted.

### Product and systems

- end-to-end p50/p95 latency and real-time factor;
- model cold-start time and peak memory;
- task completion, correction count, and time-to-confirm;
- keyboard-only and screen-reader task completion;
- typed fallback success rate.

## 5. Statistical protocol

- Report paired bootstrap 95% confidence intervals at speaker level.
- Predeclare the primary comparison and metric before reading locked-test results.
- Correct for multiple comparisons in exploratory language-pair/model grids.
- Include effect sizes and absolute counts, not only relative improvements.
- Run at least three seeds for trained heads or fine-tuning and report mean plus dispersion.

## 6. Subgroup slices

Where ethically collected and sufficiently sized, report by language direction, code-switch density, switch type, entity type, accent/region, age band, gender group including transgender speakers, device, noise band, and speech-rate band. Suppress tiny slices that create privacy or unreliable estimates; document the suppression threshold.

## 7. Phase-1 release gates

These are initial engineering gates and may be tightened after the baseline study:

- no regression greater than 1 absolute WER point versus the selected backbone on matched general speech;
- at least 20% relative reduction in critical-entity error rate versus the unguarded baseline;
- at least 15% relative reduction in switch-region WER or a statistically supported reduction in high-severity accepted errors;
- word-error ECE at or below 0.05 overall and 0.08 on declared major slices;
- lower AURC than raw decoder confidence with a speaker-level 95% confidence interval excluding zero;
- no declared major subgroup with more than 1.5× the overall high-severity accepted error rate without a documented mitigation;
- p95 short-utterance latency target defined and met on the declared reference hardware;
- zero critical accessibility violations in the automated and manual launch checklist.

Passing a metric does not authorize high-stakes autonomous use.

## 8. Ablations

- remove boundary features;
- remove N-best disagreement;
- remove acoustic-quality features;
- remove contextual biasing;
- replace calibrated confidence with raw probability;
- replace candidate-lattice entity detection with top-1 text only;
- evaluate global vs language-pair-specific calibrators.

## 9. Reproducibility artifact

Every report must include git commit, environment lock hash, model identifiers and revisions, policy/calibrator versions, dataset manifests, split hashes, normalization version, seed, reference hardware, command configuration, and machine-readable per-example outputs stripped of disallowed personal data.
