# Executive Summary

- Selected model: logistic regression, chosen by training-only 5-fold cross-validation PR-AUC (0.080).
- Held-out PR-AUC: 0.147.
- Operating threshold: 0.470; expected cost: 132.000.
- Top sensor to inspect first: `sensor_140` (composite score 0.831).
- XGBoost held-out challenger PR-AUC: 0.072 (reference comparison only; no cross-model sensitivity overlap is produced for this run).
- Monitoring flags: 68 missingness alerts, 416 feature drift alerts, prediction drift triggered, high-risk-rate drift triggered.