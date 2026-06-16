# Root-Cause Candidate Report

`sensor_059` is the top root-cause candidate (mean absolute SHAP 0.010).

Why it leads: mean absolute SHAP 0.010, fail/pass lift 0.019, and SPC flag rate 0.000.

Next engineering action: map the top sensor IDs to process step, tool, chamber, recipe, lot, and maintenance context before changing process settings. The SECOM sensor IDs are anonymized, so this metadata join connects the ranking to fab action.

## Top Candidates

| sensor | mean_abs_shap | shap_lift | spc_flag_rate |
| --- | --- | --- | --- |
| sensor_059 | 0.010 | 0.019 | 0.000 |
| sensor_033 | 0.007 | 0.004 | 0.017 |
| sensor_103 | 0.006 | 0.003 | 0.004 |
| sensor_031 | 0.006 | -0.002 | 0.000 |
| sensor_129 | 0.004 | 0.003 | 0.000 |
| sensor_130 | 0.004 | 0.002 | 0.000 |
| sensor_021 | 0.004 | 0.001 | 0.021 |
| sensor_205 | 0.004 | 0.004 | 0.004 |
| sensor_510 | 0.003 | 0.001 | 0.013 |
| sensor_064 | 0.003 | 0.004 | 0.017 |

## Sensitivity

No challenger sensitivity comparison was generated for this run. The current pipeline ranks root-cause candidates for the selected model only; a cross-model (selected vs. challenger) sensitivity overlap is not produced, so this section is intentionally omitted rather than populated from a prior run's artifacts.