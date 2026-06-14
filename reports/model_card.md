# Model Card

## Intended Use

This model scores wafer-level process records for early yield-risk review and threshold-based hold/release simulation.

Designed output: risk score, threshold flag, and sensor ranking for engineering review.

## Model Summary

- Model family: random forest
- Selected threshold: 0.140
- Expected cost at selected threshold: 83.000
- Test PR-AUC: 0.222
- Test ROC-AUC: 0.793
- Test recall: 0.812
- Test precision: 0.197

## Selection Protocol

Selection is fixed before held-out evaluation: random forest by the highest mean training-only 5-fold cross-validation PR-AUC (`cv_pr_auc_mean`). Held-out test metrics measure generalization and threshold performance.

## Confusion Matrix

- True positives: 13
- False positives: 53
- True negatives: 167
- False negatives: 3

## Monitoring Hooks

- Available checks: missingness drift, feature distribution drift, prediction distribution drift, and high-risk-rate drift.
- The checks compare reference and current batches from the processed dataset and surface alert counts and drift flags for engineering review.
