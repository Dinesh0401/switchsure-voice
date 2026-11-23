# Report acceptance checklist

- Git commit, environment lock, model/calibrator/policy revisions, seed, hardware, and split hashes are present.
- Primary comparison and metric match the predeclared plan.
- Missing predictions, timeouts, invalid audio, and abstentions are counted.
- WER/CER, switch metrics, entity metrics, calibration, risk-coverage, latency, and review burden are included where applicable.
- Paired speaker-level 95% bootstrap intervals accompany comparisons.
- Declared subgroup slices include denominators and privacy suppression.
- Reliability and risk-coverage plots use consistent axes and expose sample count.
- Release gates are machine-evaluated and failed gates remain failed.
- Limitations distinguish observed evidence from inference.
