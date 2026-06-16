# Executive Summary

- Selected model: random forest, chosen by training-only 5-fold cross-validation PR-AUC (0.227).
- Held-out PR-AUC: 0.222.
- Operating threshold: 0.140; expected cost: 83.000.
- Top sensor to inspect first: `sensor_059` (mean absolute SHAP 0.010).
- XGBoost held-out challenger PR-AUC: 0.213 (reference comparison only; no cross-model sensitivity overlap is produced for this run).
- Monitoring flags: 20 missingness alerts, 330 feature drift alerts, prediction drift clear, high-risk-rate drift clear.