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

## Benchmark Context

Published SECOM results that report very high accuracy (up to ~99%) generally apply class rebalancing such as SMOTE/ADASYN to the full dataset *before* the train/test split. This leaks minority-class structure into the held-out set and inflates every metric; restricting resampling to the training partition restores realistic performance — Park et al. (2024) report ~85% accuracy under a leak-free 70/30 protocol after the same correction, and Salem et al. (2018) establish realistic baselines across 288 method combinations [1, 2]. Under a leak-free protocol the honest ceiling on this benchmark is roughly ROC-AUC ≈ 0.8 / PR-AUC ≈ 0.2. The selected model meets that band (Test ROC-AUC 0.793, PR-AUC 0.222), against a no-skill PR-AUC floor of ≈ 0.066 (the fail prevalence). This pipeline resamples inside the cross-validation folds only, so it sits in the honest band rather than the leaked one.

References:

1. Park, H.-J.; Koo, Y.-S.; Yang, H.-Y.; Han, Y.-S.; Nam, C.-S. Study on Data Preprocessing for Machine Learning Based on Semiconductor Manufacturing Processes. *Sensors* 2024, 24 (17), 5461. https://doi.org/10.3390/s24175461
2. Salem, M.; Taheri, S.; Yuan, J.-S. An Experimental Evaluation of Fault Diagnosis from Imbalanced and Incomplete Data for Smart Semiconductor Manufacturing. *Big Data Cogn. Comput.* 2018, 2 (4), 30. https://doi.org/10.3390/bdcc2040030

## Confusion Matrix

- True positives: 13
- False positives: 53
- True negatives: 167
- False negatives: 3

## Monitoring Hooks

- Available checks: missingness drift, feature distribution drift, prediction distribution drift, and high-risk-rate drift.
- The checks compare reference and current batches from the held-out test split and surface alert counts and drift flags for engineering review.
