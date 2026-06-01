# Model Card

## Intended Use

This model scores wafer-level process records for early yield-risk screening and candidate review prioritization. Scores support engineering triage and threshold-based hold/release analysis; they are not a substitute for process-engineering validation.

## Model Summary

- Model family: random forest
- Selected threshold: 0.080
- Expected cost at selected threshold: 148.000
- Test PR-AUC: 0.193
- Test ROC-AUC: 0.758
- Test recall: 0.571
- Test precision: 0.171

## Selection Protocol

The selected model is random forest by training-only 5-fold cross-validation PR-AUC. Held-out test metrics are reported for generalization evidence and threshold evaluation, not for reopening model selection.

## Confusion Matrix

- True positives: 12
- False positives: 58
- True negatives: 235
- False negatives: 9

## Monitoring Hooks

- Missingness drift, feature distribution drift, prediction distribution drift, and high-risk-rate drift are available through the static-batch monitoring utilities.
- Monitoring output in these reports is a static-batch demonstration, not live telemetry.
