# Root-Cause Candidate Report

`sensor_140` is the top root-cause candidate (composite score 0.831).

Why it leads: mean absolute SHAP 1.266, fail/pass lift 1.337, SPC flag rate 0.013, and composite score 0.831.

Next engineering action: map the top sensor IDs to process step, tool, chamber, recipe, lot, and maintenance context before changing process settings. The SECOM sensor IDs are anonymized, so this metadata join connects the ranking to fab action.

## Top Candidates

| sensor | mean_abs_shap | shap_lift | spc_flag_rate | composite_score |
| --- | --- | --- | --- | --- |
| sensor_140 | 1.266 | 1.337 | 0.013 | 0.831 |
| sensor_572 | 0.018 | 0.003 | 0.083 | 0.208 |
| sensor_040 | 0.015 | 0.006 | 0.070 | 0.177 |
| sensor_573 | 0.023 | 0.013 | 0.061 | 0.158 |
| sensor_004 | 0.167 | 0.173 | 0.013 | 0.135 |
| sensor_432 | 0.004 | 0.001 | 0.035 | 0.086 |
| sensor_523 | 0.001 | -0.002 | 0.035 | 0.086 |
| sensor_571 | 0.013 | 0.006 | 0.032 | 0.083 |
| sensor_023 | 0.006 | -0.002 | 0.032 | 0.080 |
| sensor_139 | 0.005 | -0.002 | 0.032 | 0.079 |

## Sensitivity

No challenger sensitivity comparison was generated for this run. The current pipeline ranks root-cause candidates for the selected model only; a cross-model (selected vs. challenger) sensitivity overlap is not produced, so this section is intentionally omitted rather than populated from a prior run's artifacts.