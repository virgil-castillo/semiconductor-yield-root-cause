# Executive Summary

- Selected model: random forest, chosen by training-only 5-fold cross-validation PR-AUC (0.217).
- Held-out PR-AUC: 0.193.
- Operating threshold: 0.080; expected cost: 148.000.
- Top sensor to inspect first: `sensor_059` (composite score 0.871).
- XGBoost held-out challenger PR-AUC: 0.261; retained as a sensitivity comparator for held-out behavior.
- Monitoring flags: 0 missingness alerts, 174 feature drift alerts, prediction drift triggered, high-risk-rate drift triggered.