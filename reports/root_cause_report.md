# Root-Cause Candidate Report

`sensor_059` is the top root-cause candidate (composite score 0.871).

Why it leads: mean absolute SHAP 0.006, fail/pass lift 0.011, SPC flag rate 0.019, and composite score 0.871.

Next engineering action: map the top sensor IDs to process step, tool, chamber, recipe, lot, and maintenance context before changing process settings. The SECOM sensor IDs are anonymized, so this metadata join connects the ranking to fab action.

## Top Candidates

| sensor | mean_abs_shap | shap_lift | spc_flag_rate | composite_score |
| --- | --- | --- | --- | --- |
| sensor_059 | 0.006 | 0.011 | 0.019 | 0.871 |
| sensor_033 | 0.004 | 0.002 | 0.016 | 0.479 |
| sensor_519 | 0.003 | 0.002 | 0.025 | 0.431 |
| sensor_205 | 0.004 | 0.001 | 0.003 | 0.383 |
| sensor_031 | 0.004 | -0.000 | 0.000 | 0.357 |
| sensor_510 | 0.003 | 0.001 | 0.016 | 0.309 |
| sensor_064 | 0.002 | 0.001 | 0.016 | 0.297 |
| sensor_129 | 0.003 | 0.002 | 0.000 | 0.282 |
| sensor_065 | 0.002 | 0.001 | 0.016 | 0.275 |
| sensor_160 | 0.001 | 0.000 | 0.035 | 0.258 |

## Sensitivity

XGBoost sensitivity overlap keeps `sensor_059` in the top-k overlap sets shown below.

| top_n | overlap_count | jaccard | overlap_sensors |
| --- | --- | --- | --- |
| 5.000 | 4.000 | 0.667 | sensor_033, sensor_059, sensor_205, sensor_519 |
| 10.000 | 7.000 | 0.538 | sensor_031, sensor_033, sensor_059, sensor_129, sensor_160, sensor_205, sensor_519 |
| 20.000 | 12.000 | 0.429 | sensor_021, sensor_031, sensor_033, sensor_059, sensor_129, sensor_160, sensor_175, sensor_205, sensor_460, sensor_510, sensor_519, sensor_572 |