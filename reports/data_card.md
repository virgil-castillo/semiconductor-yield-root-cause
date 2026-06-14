# Data Card

## Dataset Facts

- Source: public historical SECOM benchmark with static wafer records rather than a complete fab execution trace.
- Label: binary pass/fail; it does not identify the failure mode.
- Sensors: anonymous sensor identifiers; process step, tool, chamber, recipe, lot, and maintenance metadata are not included.
- Split: 1331 training rows (fail rate 0.066) and 236 test rows (fail rate 0.068).
- Sensor matrix: 590 raw sensor columns with overall missing-value rate 0.045. The train/test CSVs hold the unprocessed sensor readings (all columns, missing values intact, no feature selection). Preprocessing — missing/CV/correlation filtering plus median imputation — is fit per cross-validation fold inside the model pipeline to avoid leakage, not applied before the split.

## Batch Monitoring Checks

- Missingness alerts: 20
- Feature drift alerts: 330
- Prediction drift alert: False
- High-risk-rate alert: False
