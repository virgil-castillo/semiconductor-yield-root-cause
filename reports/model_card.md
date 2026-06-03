# Model Card

## Intended Use

This model scores wafer-level process records for early yield-risk review and threshold-based hold/release simulation.

Designed output: risk score, threshold flag, and sensor ranking for engineering review.

## Model Summary

- Model family: random forest
- Selected threshold: 0.080
- Expected cost at selected threshold: 148.000
- Test PR-AUC: 0.193
- Test ROC-AUC: 0.758
- Test recall: 0.571
- Test precision: 0.171

## Selection Protocol

Selection is fixed before held-out evaluation: random forest by training-only 5-fold cross-validation PR-AUC. Held-out test metrics measure generalization and threshold performance.

## Confusion Matrix

- True positives: 12
- False positives: 58
- True negatives: 235
- False negatives: 9

## Monitoring Hooks

- Available checks: missingness drift, feature distribution drift, prediction distribution drift, and high-risk-rate drift.
- The checks compare reference and current batches from the processed dataset and surface alert counts and drift flags for engineering review.
