# Early Detection Study

Generated: 2026-06-17
Updated: 2026-06-18

## Status

The Optuna study is finished and the held-out Spec C comparison has been run
from the persisted 100-trial best record.

| Source | Trial count | State summary |
| --- | ---: | --- |
| `models/early_detection_study.pkl` | 100 | 100 complete |
| `reports/early_detection_trials.csv` | 100 | 100 complete |

The configured study size is `n_trials: 100` in
`configs/early_detection_config.yaml`, so the run reached the expected count.
The held-out evaluation was generated with:

```powershell
python scripts\run_early_detection.py --evaluate-existing
```

## Scope

This document is the human-facing usage and interpretation report described by
the early-detection design docs. The current artifact set includes the Optuna
study handoff artifacts plus the held-out comparison, comparison CSV, diagnostic
prefix curve CSV, curve figure, and root-cause overlap checks against the
selected-model explanation artifacts.

## Method

The experiment treats raw SECOM sensor column order as a proxy for fabrication
progress. This is a modeling assumption only: the anonymized SECOM data does not
provide real process-stage metadata.

The Optuna objective maximizes:

```text
penalized_score = mean_inner_cv_pr_auc - 0.10 * observation_fraction
```

Key run settings:

| Setting | Value |
| --- | --- |
| Sampler | Optuna `TPESampler` |
| Sampler seed | 42 |
| Trials | 100 |
| Inner CV folds | 5 |
| Primary detection metric | `pr_auc` |
| Earliness penalty alpha | 0.10 |
| Threshold policy | `tune` |
| Outer test split | 15%, held out until Spec C evaluation |
| Sensor count | 590 |
| Performance tolerance | 5% of the best full-sensor PR-AUC |

The search space includes prefix or contiguous-window sensor access,
missingness/CV/correlation preprocessing thresholds, optional feature
selection, model family and model hyperparameters, and a tuned threshold. With
the default `pr_auc` objective, the tuned threshold is recorded but does not
affect the trial score because PR-AUC is threshold-free.

## Artifacts

| Path | Contents |
| --- | --- |
| `models/early_detection_study.pkl` | Serialized Optuna study |
| `models/early_detection_best.json` | Best-trial config and provenance |
| `reports/early_detection_trials.csv` | Full Optuna trials dataframe |
| `reports/early_detection_metrics.json` | Held-out comparison metric rows |
| `reports/early_detection_comparison.json` | Held-out comparison metadata, rows, curve rows, and verdict |
| `reports/early_detection_comparison.csv` | Held-out comparison rows |
| `reports/early_detection_curve.csv` | Diagnostic prefix sweep rows |
| `reports/figures/early_detection_curve.png` | Held-out PR-AUC curve by prefix length |

## Best Trial

Best trial: `85`

| Metric | Value |
| --- | ---: |
| Penalized score | 0.197619 |
| Mean inner-CV PR-AUC | 0.223890 |
| Observation fraction | 0.262712 |
| Prefix end / sensors used | 155 |
| Sensors used | 155 / 590 |
| Mean selected features | 72.6 |
| Feasible | true |

Best configuration:

| Field | Value |
| --- | --- |
| Sensor access | Prefix length 155 (`prefix_end=155`, exclusive boundary) |
| Missing threshold | 0.434558 |
| CV threshold | 0.071714 |
| Correlation threshold | 0.929347 |
| Feature selection | none |
| Model family | random forest |
| Threshold | 0.217130 |
| Random forest `n_estimators` | 613 |
| Random forest `max_depth` | 15 |
| Random forest `min_samples_leaf` | 13 |
| Random forest `max_features` | sqrt |
| Random forest `class_weight` | balanced_subsample |

The values in this section are inner-CV search values from the training split.
They are not held-out metrics.

## Held-Out Comparison

Verdict: `baseline_preferred`.

| Role | Model | Sensors | PR-AUC | ROC-AUC | Precision | Recall | F1 | False alarm rate | Expected cost | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `early_optuna_best` | random_forest | 155 / 590 | 0.194720 | 0.759091 | 0.144578 | 0.750000 | 0.242424 | 0.322727 | 111.0 | 0.217130 |
| `full_prefix_same_config` | random_forest | 590 / 590 | 0.221676 | 0.758807 | 0.138889 | 0.625000 | 0.227273 | 0.281818 | 122.0 | 0.217130 |
| `tabular_selected_model` | random_forest | 590 / 590 | 0.221624 | 0.792614 | 0.196970 | 0.812500 | n/a | n/a | 83.0 | 0.140000 |

The selected early model uses 26.3% of the sensor sequence. Its held-out PR-AUC
is below the best full-sensor comparison by more than the configured 5%
tolerance, so the current held-out result favors the full-sensor baseline. At
the recorded thresholds, expected costs are 111.0 for the early model, 122.0 for
the full-prefix variant, and 83.0 for the selected tabular baseline.

Confusion matrices at the recorded thresholds:

| Role | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| `early_optuna_best` | 149 | 71 | 4 | 12 |
| `full_prefix_same_config` | 158 | 62 | 6 | 10 |

The selected tabular baseline is imported from `reports/model_comparison.json`;
that existing artifact records expected cost `83.0` at threshold `0.14`.

## Diagnostic Prefix Curve

The diagnostic curve reuses the winning non-access settings and refits prefix
models at configured prefix lengths, the selected early prefix length, and the
full sensor count. It is diagnostic only and does not feed back into model
choice.

| Prefix length | Sensor fraction | Held-out PR-AUC | Expected cost |
| ---: | ---: | ---: | ---: |
| 16 | 0.027119 | 0.074693 | 213.0 |
| 32 | 0.054237 | 0.090292 | 179.0 |
| 64 | 0.108475 | 0.215714 | 126.0 |
| 128 | 0.216949 | 0.205706 | 119.0 |
| 155 | 0.262712 | 0.194720 | 111.0 |
| 256 | 0.433898 | 0.223055 | 96.0 |
| 512 | 0.867797 | 0.216824 | 110.0 |
| 590 | 1.000000 | 0.221676 | 122.0 |

The best point on this descriptive curve is index 256 with held-out PR-AUC
0.223055, but this was not selected by the Optuna study and is not a replacement
model decision. The curve is still the branch's strongest engineering signal:
64 sensors already reach PR-AUC 0.215714, within the configured 5% tolerance of
the full-prefix comparison, and 256 sensors produce the best diagnostic
combination of PR-AUC and expected cost in this artifact set.

## Root-Cause Overlap

The selected early prefix contains most of the sensors already highlighted by
the selected full model's SHAP root-cause ranking. This supports the
interpretation that early detection is finding signal in the same region of the
sensor order that later drives the full-model explanation.

| Ranking source | Top N | Inside first 155 sensors | Expected by chance | Hypergeometric p-value |
| --- | ---: | ---: | ---: | ---: |
| SHAP root-cause candidates | 5 | 5 / 5 | 1.31 | 0.001193 |
| SHAP root-cause candidates | 10 | 8 / 10 | 2.63 | 0.000536 |
| SHAP root-cause candidates | 20 | 12 / 20 | 5.25 | 0.001241 |
| Selected-model impurity importance | 10 | 8 / 10 | 2.63 | 0.000536 |
| Selected-model impurity importance | 20 | 14 / 20 | 5.25 | 0.000039 |

The top SHAP candidates inside the selected early prefix include `sensor_059`,
`sensor_033`, `sensor_103`, `sensor_031`, `sensor_129`, `sensor_130`,
`sensor_021`, and `sensor_064`. The same caveat still applies: SECOM sensor IDs
are anonymized, so this is an enrichment result over anonymous column order, not
a causal process-stage claim.

## Top Trials

| Trial | Score | PR-AUC | Observation fraction | Prefix end / latest index | Features | Model | Access | Selection |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 85 | 0.197619 | 0.223890 | 0.262712 | 155 | 72.6 | random_forest | prefix | none |
| 25 | 0.196408 | 0.223188 | 0.267797 | 158 | 74.4 | random_forest | prefix | univariate |
| 65 | 0.196137 | 0.220883 | 0.247458 | 146 | 22.0 | random_forest | prefix | model_importance |
| 59 | 0.194330 | 0.219076 | 0.247458 | 146 | 87.8 | random_forest | prefix | none |
| 56 | 0.194120 | 0.211577 | 0.174576 | 103 | 62.8 | random_forest | prefix | none |

## Trial Summary

| Statistic | Value |
| --- | ---: |
| Complete trials | 100 |
| Feasible trials | 99 |
| Infeasible trials | 1 |
| Mean penalized score | 0.130715 |
| Median penalized score | 0.154394 |
| Minimum penalized score | -1.000000 |
| Mean raw PR-AUC | 0.175146 |
| Minimum raw PR-AUC among feasible trials | 0.066115 |
| Mean observation fraction | 0.326831 |
| Minimum observation fraction | 0.003390 |
| Maximum observation fraction | 1.000000 |
| Minimum prefix end / latest-index field | 2 |
| Maximum prefix end / latest-index field | 590 |

Trial distribution:

| Dimension | Counts |
| --- | --- |
| Sensor access | prefix: 87, window: 13 |
| Model family | random_forest: 64, xgboost: 20, logistic_regression: 8, lightgbm: 8 |
| Feature selection | none: 55, univariate: 23, model_importance: 12, mutual_info: 10 |

Among the top ten trials by penalized score, all used prefix access and nine
used random forests. Their mean observation fraction was 0.226441, so the best
region of the completed search favors earlier prefixes rather than full-sensor
access.

These category summaries are search diagnostics. Because Optuna samples
adaptively, they are directional evidence rather than balanced experimental
effects. A confirmatory comparison of model families or access patterns would
need repeated seeded studies or a balanced follow-up experiment.

## Interpretation

The best completed study result uses the first 155 of 590 ordered sensor
columns, about 26.3% of the available sensor sequence. The earlier reported
0.223890 PR-AUC was an inner-CV search value from the training split, not a
held-out metric.

On the held-out split, the selected early model reaches PR-AUC 0.194720. The
same non-access configuration with all sensors reaches PR-AUC 0.221676, and the
existing selected tabular baseline reaches PR-AUC 0.221624. Under the configured
5% tolerance rule, the early model is not competitive with the full-sensor
comparisons in this completed study. The expected-cost comparison at the
recorded thresholds is 111.0 for the early model, 122.0 for the full-prefix
variant, and 83.0 for the selected tabular baseline.

The result should be interpreted as useful evidence about where early signal may
exist in the ordered sensor sequence, not as a production early-exit model. The
diagnostic curve and root-cause enrichment point to the same practical
conclusion: the failure signal appears early and may saturate before the full
590-sensor sequence, but the exact early operating point needs paired
confirmation before deployment.

## Statistical Evidence And Confirmation Plan

| Question | Current evidence | Method needed |
| --- | --- | --- |
| Are top root-cause sensors enriched in the selected early prefix? | Supported by current artifacts; 8 of the top 10 SHAP candidates are inside the first 155 sensors. | Hypergeometric enrichment test. |
| Is the selected early model non-inferior to the full-sensor model? | Not supported by the current held-out metrics; PR-AUC is 0.194720 vs 0.221676. | Paired bootstrap PR-AUC difference with the configured 5% non-inferiority margin. |
| Are 64- or 256-sensor prefixes viable operating points? | Promising diagnostic curve points: PR-AUC 0.215714 at 64 sensors and 0.223055 at 256 sensors. | Paired bootstrap over saved per-wafer prefix predictions. |
| Are prefix access and random forest truly better than alternatives? | Directional only; top trials mostly use prefix access and random forest. | Repeated seeded Optuna studies or a balanced confirmatory experiment. |
| Do thresholded decisions differ materially between early and full models? | Aggregate confusion matrices are available, but paired wafer-level disagreements are not persisted. | McNemar test plus paired bootstrap for recall, false-alarm rate, and expected cost. |
| Are ROC-AUC differences meaningful? | Aggregate ROC-AUC values are available. | DeLong test on paired prediction scores. |

The current artifact set is enough to support the root-cause enrichment claim
and the descriptive prefix-curve interpretation. It is not enough to prove
non-inferiority or thresholded decision equivalence because the early/full
per-wafer score vectors are not persisted for every comparison.

## Reproducing The Study

From the repository root:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
python scripts\run_early_detection.py
```

To evaluate the already completed 100-trial study without rerunning Optuna:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
python scripts\run_early_detection.py --evaluate-existing
```
