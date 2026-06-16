# Early Detection Study

Generated: 2026-06-16

## Status

The Optuna study is finished. Both persisted artifacts agree that all configured
trials completed:

| Source | Trial count | State summary |
| --- | ---: | --- |
| `models/early_detection_study.pkl` | 100 | 100 complete |
| `reports/early_detection_trials.csv` | 100 | 100 complete |

The configured study size is `n_trials: 100` in
`configs/early_detection_config.yaml`, so the run reached the expected count.

## Scope

This document is the human-facing usage and interpretation report described by
the early-detection design docs. The current implementation corresponds to the
Optuna study slice: it writes the study, best-trial handoff record, and trials
table. The held-out test evaluation, full-feature baseline comparison, and
early-detection curve are deferred to the comparison slice and are not present in
this artifact set.

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
| Outer test split | 15%, held out and not evaluated in this slice |
| Sensor count | 590 |

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

Deferred comparison artifacts expected by the umbrella design are not available
yet: `reports/early_detection_metrics.json`,
`reports/early_detection_comparison.json`,
`reports/early_detection_comparison.csv`, and
`reports/figures/early_detection_curve.png`.

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
columns, about 26.3% of the available sensor sequence, while reaching a mean
inner-CV PR-AUC of 0.223890 after the earliness penalty. This is a search result
on the training split only, not a deployment metric.

The result should be interpreted as a candidate early-detection configuration
for the next evaluation step. A production decision requires the deferred
held-out test comparison against a full-feature baseline on the same outer split.

## Reproducing The Study

From the repository root:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
python scripts\run_early_detection.py
```

The command rewrites the three namespaced study artifacts listed above using the
configured paths in `configs/config.yaml`.
