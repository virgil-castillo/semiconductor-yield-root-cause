# Modeling Tier — Multi-Family Comparison and Selection

## Problem

The pipeline currently trains a single logistic-regression baseline
(`models/baseline_lr.joblib`). On the held-out test set it reaches ROC-AUC
0.66, PR-AUC 0.16, and recall 0.24 (catches 5 of 21 failing wafers, raises 37
false alarms). No other model family is implemented, despite
`configs/model_config.yaml` already defining hyperparameter grids for random
forest, XGBoost, and LightGBM, and `load_model_config()` already being written.

This design implements a full model-family comparison: train and tune each
family with cross-validated hyperparameter search, select a winner on a
leakage-free criterion, evaluate the winner once on the test set, and expose the
winner through a stable artifact path that downstream scripts consume.

## Scope

In scope:

- A model registry covering four distinct approaches: a stratified dummy
  classifier (chance floor), logistic regression (linear baseline), random
  forest (bagged trees), and XGBoost (gradient-boosted trees).
- `RandomizedSearchCV` hyperparameter search per tunable family (LR, RF,
  XGBoost), driven by the existing `configs/model_config.yaml`. The dummy
  classifier is fit directly with no search.
- Winner selection by mean cross-validated PR-AUC on the training set.
- Single-touch test-set evaluation of the winner, with the cost-sensitive
  threshold applied.
- A per-family comparison table written to `reports/`.
- A stable `models/selected_model.joblib` artifact and repointing of the four
  downstream scripts that currently hardcode `baseline_lr.joblib`.

Out of scope (deferred):

- SMOTE / resampling as an additional comparison axis. Class-weighting
  (`class_weight="balanced"`, and `scale_pos_weight` for XGBoost) is used
  instead. Resampling can be added later as an extra axis if it proves
  worthwhile.
- Stacking / ensembling across families.
- LightGBM. Its grid stays in `configs/model_config.yaml` as a documented
  option, but it is not wired into the registry: it is the same model class as
  XGBoost (gradient-boosted trees) and would score within noise of it on a
  dataset this small, so it adds a near-duplicate comparison row and a redundant
  tuning sweep without adding a distinct modeling approach.
- HPC / SLURM execution. A full run is minutes-scale on a workstation, so no
  cluster scaffolding is warranted.
- Changes to preprocessing, feature engineering, or the train/test split.

## Selection discipline

Model selection must not consult the test set.

- Each family is tuned with `RandomizedSearchCV` on the **training set only**.
- The winner is the family with the highest **mean cross-validated PR-AUC**.
- The **test set is touched exactly once**, to evaluate the already-chosen
  winner.
- Test metrics for the non-winning families are also reported in the comparison
  table. This is transparency only: the selection decision is made on
  cross-validation before any test metric is computed, so reporting test numbers
  does not influence the choice.

## Components

### `src/yield_risk/model.py` (extend)

The existing `build_baseline_pipeline`, `train_model`, and
`cross_validate_model` remain. Add:

- **A model registry.** A mapping from family name (`dummy`,
  `logistic_regression`, `random_forest`, `xgboost`) to a factory that builds an
  unfitted estimator plus a flag indicating whether the family requires feature
  scaling. Logistic regression requires a `StandardScaler`; the dummy and tree
  families do not. Each registry entry produces an sklearn `Pipeline` of the form
  `StandardScaler -> estimator` (scaling families) or `estimator` (the rest), so
  the explainability code's existing `_transform_pre_steps` logic continues to
  work unchanged.

- **A search runner**, e.g. `run_search(name, X, y, model_cfg, run_cfg)`. For a
  tunable family it reads the family's grid and the `search` block from the model
  config, builds a `RandomizedSearchCV` (`scoring="average_precision"`,
  `StratifiedKFold` with `run_cfg.cv_folds`, `n_iter`, `n_jobs`, `refit=True`,
  seeded by `run_cfg.random_seed`), fits it on `X`/`y`, and returns the refit
  best estimator together with the mean and standard deviation of the
  cross-validated PR-AUC. For XGBoost, `scale_pos_weight` is computed from the
  training class ratio at fit time (the config comment already specifies this).
  The dummy classifier has no grid: it is fit directly and its mean CV PR-AUC is
  obtained from a plain `cross_val_score` so it still appears in the comparison
  as the chance floor.

- **A winner-selection function**, e.g.
  `select_best(results) -> str`, returning the family name with the highest mean
  CV PR-AUC. Ties break toward the first family in a fixed order. This is a small
  function, not a separate module.

### `src/yield_risk/evaluate.py` (touch up)

Generalize `format_report` to accept a model name and use it in the report
title, replacing the hardcoded `"=== Baseline Model Evaluation ==="`. No other
behavior change.

## Scripts

### `scripts/train_models.py` (replaces `scripts/train_baseline.py`)

1. Load `config.yaml` and `model_config.yaml`; read the processed training
   split; select `sensor_` columns and the `label` column.
2. For each family in the registry, run `run_search` and record mean/std CV
   PR-AUC and the chosen hyperparameters.
3. Persist each family's refit best estimator to `models/<family>.joblib`.
4. Select the winner with `select_best`, and copy/save it to
   `models/selected_model.joblib`.
5. Write the cross-validation portion of the comparison table (per-family mean
   CV PR-AUC + chosen hyperparameters) to `reports/model_comparison.{csv,json}`.
   The test-metric columns are added later by `evaluate_model.py`; they are not
   written here, because this script must not touch the test set.

`scripts/train_baseline.py` is removed; its behavior is a strict subset of
`train_models.py`.

### `scripts/evaluate_model.py` (generalize)

1. Load `models/selected_model.joblib` instead of `baseline_lr.joblib`.
2. Compute test-set metrics for the winner via `compute_metrics`.
3. Apply the cost-sensitive threshold from `thresholding.find_optimal_threshold`
   (using `cost_config.yaml`) and report the operating point, confusion matrix,
   and expected cost at that threshold.
4. Optionally compute test metrics for each non-winning family from its saved
   `models/<family>.joblib`, to complete `reports/model_comparison.csv`.
5. Regenerate the diagnostic figures for the winner.

## Artifacts

| Path | Contents |
|------|----------|
| `models/<family>.joblib` | Refit best estimator per family |
| `models/selected_model.joblib` | The winning estimator (stable path) |
| `reports/model_comparison.csv` / `.json` | Per-family CV PR-AUC, test metrics, optimal threshold, expected cost |
| `reports/figures/*.png` | Confusion matrix, ROC, PR curve for the winner |

## Downstream wiring

Four scripts currently hardcode `models/baseline_lr.joblib`:

- `scripts/evaluate_model.py` (line 22)
- `scripts/feature_importance.py` (line 15)
- `scripts/generate_explanations.py` (line 38)
- `scripts/batch_score.py` (line 63, as the `--model` default)

All are repointed to `models/selected_model.joblib`. This is the reason for the
stable selected-model path: otherwise a winning tree model would be selected
while the logistic regression is still the artifact being scored and explained.

SHAP requires no change for the candidate winners. `compute_shap_values` already
dispatches on estimator type — `LinearExplainer` for estimators with `coef_`
(logistic regression), `TreeExplainer` for estimators with
`feature_importances_` (random forest, XGBoost). The dummy classifier has
neither attribute and would raise `NotImplementedError`, but this never occurs:
the dummy is a chance floor and cannot win selection, and `generate_explanations`
only ever runs on `selected_model.joblib`.

## Testing

- `tests/test_model.py` (extend):
  - The registry builds each family.
  - Logistic-regression pipelines include a `StandardScaler` step; tree
    pipelines do not.
  - `run_search` returns a fitted estimator carrying the chosen hyperparameters.
  - `select_best` returns the family with the highest mean CV PR-AUC.
  - All tests use small synthetic data with a minimal grid and low `n_iter`, so
    the suite stays fast.
- `tests/test_evaluate.py` (extend): `format_report` reflects the passed-in
  model name.

The real search (`n_iter: 20`, 5-fold, full feature set) runs only in
`train_models.py`, never in tests. A full training run is expected to take
several minutes.

## Risks

- **Marginal gains.** SECOM is a hard, noisy, highly imbalanced dataset. Tree
  models may improve ROC-AUC/PR-AUC modestly rather than dramatically. The
  deliverable is a rigorous, reproducible comparison and a defensible selection,
  not a guaranteed accuracy jump.
- **Roadmap drift.** `docs/ROADMAP.md` already marks Tier 2 modeling complete
  and uses stale module names (`modeling.py`, `evaluation.py`). Reconciling the
  roadmap is out of scope here but should follow.
