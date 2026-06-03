# Data Card

## Dataset Facts

- Source: public historical SECOM benchmark with static wafer records rather than a complete fab execution trace.
- Label: binary pass/fail; it does not identify the failure mode.
- Sensors: anonymous sensor identifiers; process step, tool, chamber, recipe, lot, and maintenance metadata are not included.
- Split: 1253 training rows (fail rate 0.066) and 314 test rows (fail rate 0.067).
- Sensor matrix: 198 sensor columns with overall missing-value rate 0.000 after preprocessing.

## Batch Monitoring Checks

- Missingness alerts: 0
- Feature drift alerts: 174
- Prediction drift alert: True
- High-risk-rate alert: True
