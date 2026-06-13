# Model Card

## Intended Use

This model scores wafer-level process records for early yield-risk review and threshold-based hold/release simulation.

Designed output: risk score, threshold flag, and sensor ranking for engineering review.

## Model Summary

- Model family: logistic regression
- Selected threshold: 0.470
- Expected cost at selected threshold: 132.000
- Test PR-AUC: 0.147
- Test ROC-AUC: 0.724
- Test recall: 0.353
- Test precision: 0.214

## Selection Protocol

Selection is fixed before held-out evaluation: logistic regression by the highest one-sigma lower-confidence bound on training-only 5-fold cross-validation PR-AUC (`selection_score = cv_pr_auc_mean - std_penalty * cv_pr_auc_std`, with `std_penalty = 1.0`). Penalizing the CV mean by its per-fold standard deviation favors families whose performance is consistent across folds rather than driven by a single high-variance fold. Held-out test metrics measure generalization and threshold performance.

## Confusion Matrix

- True positives: 6
- False positives: 22
- True negatives: 274
- False negatives: 11

## Monitoring Hooks

- Available checks: missingness drift, feature distribution drift, prediction distribution drift, and high-risk-rate drift.
- The checks compare reference and current batches from the processed dataset and surface alert counts and drift flags for engineering review.
