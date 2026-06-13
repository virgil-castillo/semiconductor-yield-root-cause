# Data Card

## Dataset Facts

- Source: public historical SECOM benchmark with static wafer records rather than a complete fab execution trace.
- Label: binary pass/fail; it does not identify the failure mode.
- Sensors: anonymous sensor identifiers; process step, tool, chamber, recipe, lot, and maintenance metadata are not included.
- Split: 1254 training rows (fail rate 0.069) and 313 test rows (fail rate 0.054).
- Sensor matrix: 590 raw sensor columns with overall missing-value rate 0.045. The train/test CSVs hold the unprocessed sensor readings (all columns, missing values intact, no feature selection). Preprocessing — missing/variance/correlation filtering plus median imputation — is fit per cross-validation fold inside the model pipeline to avoid leakage, not applied before the split.

## Batch Monitoring Checks

- Missingness alerts: 68
- Feature drift alerts: 416
- Prediction drift alert: True
- High-risk-rate alert: True
