# Early Detection Study

Generated: 2026-06-17

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
prefix curve CSV, and curve figure.

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
| Latest sensor index | 155 |
| Sensors used | 155 / 590 |
| Mean selected features | 72.6 |
| Feasible | true |

Best configuration:

| Field | Value |
| --- | --- |
| Sensor access | Prefix through sensor index 155 |
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

| Role | Model | Sensors | PR-AUC | ROC-AUC | Precision | Recall | F1 | False alarm rate | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `early_optuna_best` | random_forest | 155 / 590 | 0.194720 | 0.759091 | 0.144578 | 0.750000 | 0.242424 | 0.322727 | 0.217130 |
| `full_prefix_same_config` | random_forest | 590 / 590 | 0.221676 | 0.758807 | 0.138889 | 0.625000 | 0.227273 | 0.281818 | 0.217130 |
| `tabular_selected_model` | random_forest | 590 / 590 | 0.221624 | 0.792614 | 0.196970 | 0.812500 | n/a | n/a | 0.140000 |

The selected early model uses 26.3% of the sensor sequence. Its held-out PR-AUC
is below the best full-sensor comparison by more than the configured 5%
tolerance, so the current held-out result favors the full-sensor baseline.

Confusion matrices at the recorded thresholds:

| Role | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| `early_optuna_best` | 149 | 71 | 4 | 12 |
| `full_prefix_same_config` | 158 | 62 | 6 | 10 |

The selected tabular baseline is imported from `reports/model_comparison.json`;
that existing artifact records expected cost `83.0` at threshold `0.14`.

## Diagnostic Prefix Curve

The diagnostic curve reuses the winning non-access settings and refits prefix
models at configured prefix lengths, the selected early index, and the full
sensor count. It is diagnostic only and does not feed back into model choice.

| Latest sensor index | Sensor fraction | Held-out PR-AUC |
| ---: | ---: | ---: |
| 16 | 0.027119 | 0.074693 |
| 32 | 0.054237 | 0.090292 |
| 64 | 0.108475 | 0.215714 |
| 128 | 0.216949 | 0.205706 |
| 155 | 0.262712 | 0.194720 |
| 256 | 0.433898 | 0.223055 |
| 512 | 0.867797 | 0.216824 |
| 590 | 1.000000 | 0.221676 |

The best point on this descriptive curve is index 256 with held-out PR-AUC
0.223055, but this was not selected by the Optuna study and is not a replacement
model decision.

## Top Trials

| Trial | Score | PR-AUC | Observation fraction | Latest sensor | Features | Model | Access | Selection |
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
| Minimum latest sensor index | 2 |
| Maximum latest sensor index | 590 |

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

## Interpretation

The best completed study result uses the first 155 of 590 ordered sensor
columns, about 26.3% of the available sensor sequence. The earlier reported
0.223890 PR-AUC was an inner-CV search value from the training split, not a
held-out metric.

On the held-out split, the selected early model reaches PR-AUC 0.194720. The
same non-access configuration with all sensors reaches PR-AUC 0.221676, and the
existing selected tabular baseline reaches PR-AUC 0.221624. Under the configured
5% tolerance rule, the early model is not competitive with the full-sensor
comparisons in this completed study.

The result should be interpreted as useful evidence about where early signal may
exist in the ordered sensor sequence, not as a production early-exit model.

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
