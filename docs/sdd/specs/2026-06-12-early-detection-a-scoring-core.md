# SDD Spec A: Early-Detection Leakage-Safe Scoring Core

> Slice 1 of 3. Umbrella design:
> [`2026-06-12-early-detection-bayesian-optimization.md`](2026-06-12-early-detection-bayesian-optimization.md).
> This spec is the testable foundation: scoring ONE configuration with
> nested-fold-safe CV. **No Optuna here** — Spec B wraps this in a study.

## Goal

Implement the pure, deterministic core that scores a single early-detection
configuration: slice an ordered SECOM sensor window, run leakage-safe inner
cross-validation (all preprocessing/selection fit on the train fold only), and
return the penalized objective score plus raw detection metrics. This is the
function the Optuna objective (Spec B) will call once per trial — and the unit
you verify on real SECOM before building anything on top of it.

## Assumption

`observation_fraction` and the prefix/window slicing treat **raw SECOM column
order as a stand-in for fabrication progress** — a declared assumption, not a
fact about the data. "Earliness" means earlier *in column order*.

## Scope

- New file `src/yield_risk/early_detection.py` (Spec B/C append to it).
- New test file `tests/test_early_detection.py`.
- **MUST NOT** modify tabular pipeline modules, `evaluate.ClassificationMetrics`,
  existing scripts, or any existing artifact.
- Dev contract: `from __future__ import annotations`; Google-style docstrings
  (Args/Returns/Raises); full annotations; `mypy --strict` clean; ruff
  `E,F,W,I,UP,ANN` @ line length 88.
- No new third-party dependency (Optuna is added in Spec B).

## Reuse

- `yield_risk.preprocess.drop_high_missing`, `drop_low_variance`,
  `drop_high_correlation` to **derive** the retained sensor columns from the
  train fold (these operate on a DataFrame whose sensor columns keep their
  `sensor_` names). Median/scaler/selector statistics are captured from the
  train fold and replayed on the val fold (do NOT re-derive on val).
- `yield_risk.evaluate.compute_metrics` for `roc_auc`/`pr_auc`.
- `yield_risk.config.CostMatrix` only for the `neg_expected_cost` metric.

## Deliverables

### 1. Sensor access
```python
@dataclass(frozen=True)
class SensorAccess:
    access_type: str          # "prefix" | "window"
    prefix_end: int | None = None      # set iff access_type == "prefix"
    window_start: int | None = None    # set iff access_type == "window"
    window_size: int | None = None     # set iff access_type == "window"

def latest_index(access: SensorAccess, n_sensors: int) -> int: ...
def observation_fraction(access: SensorAccess, n_sensors: int) -> float: ...
def select_ordered_sensors(
    x: pd.DataFrame, raw_sensor_cols: list[str], access: SensorAccess
) -> tuple[pd.DataFrame, list[str]]: ...
```
- `prefix` → window = first `prefix_end` of `raw_sensor_cols` (column order
  preserved). `latest_index = prefix_end`.
- `window` → window = `raw_sensor_cols[window_start : window_start +
  window_size]`, end clipped to `n_sensors`. `latest_index = min(window_start +
  window_size, n_sensors)`.
- `observation_fraction = latest_index / n_sensors`, in `(0, 1]`.
- `select_ordered_sensors` returns the sliced DataFrame (sensor columns only,
  original names) and the ordered window column list.

### 2. Configuration object (constructed directly in tests; Optuna fills it later)
```python
@dataclass(frozen=True)
class EarlyConfig:
    access: SensorAccess
    missing_threshold: float
    variance_threshold: float
    correlation_threshold: float
    selection_method: str            # "none"|"univariate"|"mutual_info"|"model_importance"
    max_features: int | None         # None when selection_method == "none"
    model_family: str                # "logistic_regression"|"random_forest"|"xgboost"|"lightgbm"
    model_params: dict[str, object]
    threshold_policy: str            # "tune" | "far_constraint"
    threshold: float | None          # set iff threshold_policy == "tune"
    false_alarm_rate: float | None   # set iff threshold_policy == "far_constraint"
```

### 3. Fold-local preprocessor (leakage boundary)
```python
@dataclass
class FoldPreprocessor:
    retained_cols: list[str]
    medians: dict[str, float]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    selected_idx: np.ndarray         # indices into retained_cols kept by selection

    @classmethod
    def fit(cls, x_train_window: pd.DataFrame, y_train: np.ndarray,
            cfg: EarlyConfig) -> FoldPreprocessor: ...
    def transform(self, x_window: pd.DataFrame) -> np.ndarray: ...   # (n_rows, n_selected) float64
```
- `fit` order on the **train fold only**: drop-high-missing → median impute →
  drop-low-variance → drop-high-correlation (reusing the three `preprocess`
  helpers to choose retained columns) → fit `StandardScaler` → fit feature
  selector. Capture every statistic.
- Selector: `"none"` keeps all; `"univariate"` = `SelectKBest(f_classif,
  k=min(max_features, n_retained))`; `"mutual_info"` = `SelectKBest(
  mutual_info_classif, ...)`; `"model_importance"` = `SelectFromModel` on a
  seeded estimator with `max_features` cap. `selected_idx` records the kept
  retained-column positions.
- `transform` subsets to `retained_cols`, imputes with saved `medians`, applies
  `(x - scaler_mean) / scaler_scale`, then `selected_idx` — using **only**
  fitted statistics. Zero-variance scale element → 1.0 (no div-by-zero).

### 4. Estimator + threshold + metric
```python
def build_estimator(model_family: str, model_params: dict[str, object],
                    random_seed: int) -> ClassifierMixin: ...
def resolve_threshold(y_val: np.ndarray, y_prob: np.ndarray,
                      cfg: EarlyConfig) -> float: ...
def compute_detection_metric(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
    metric_name: str, cost_matrix: CostMatrix | None = None) -> float: ...
```
- `build_estimator` returns a seeded sklearn-compatible classifier exposing
  `fit` and `predict_proba`. For `xgboost`/`lightgbm`, default
  `scale_pos_weight` to train-fold `n_neg/n_pos` unless overridden in
  `model_params`.
- `resolve_threshold`: `"tune"` returns `cfg.threshold`; `"far_constraint"`
  returns the lowest threshold whose false-alarm rate ≤
  `cfg.false_alarm_rate` (maximizing recall), or `1.0` if none qualifies.
- `compute_detection_metric`: `"pr_auc"` = average precision; `"recall_at_far"`
  = recall at the resolved threshold; `"neg_balanced_error"` =
  `balanced_accuracy - 1`; `"neg_expected_cost"` = negated
  `thresholding.expected_cost_at_threshold` (requires `cost_matrix`, else
  `ValueError`). PR-AUC/ROC-AUC are threshold-free.

### 5. Top-level scorer
```python
@dataclass
class ConfigScore:
    penalized_score: float
    detection_metric: float          # mean over valid folds; nan if none valid
    observation_fraction: float
    n_features_selected: float       # mean retained-after-selection across folds
    feasible: bool
    per_fold: list[dict[str, float]]

def evaluate_config(
    x: pd.DataFrame, y: np.ndarray, raw_sensor_cols: list[str],
    cfg: EarlyConfig, *, inner_cv_folds: int, alpha: float,
    detection_metric: str, random_seed: int,
    cost_matrix: CostMatrix | None = None) -> ConfigScore: ...
```
- `StratifiedKFold(n_splits=inner_cv_folds, shuffle=True,
  random_state=random_seed)` over the full passed `x`, `y`.
- Per fold: slice window on train+val → `FoldPreprocessor.fit` on train →
  `transform` both → `build_estimator` + fit on train → `predict_proba[:, 1]`
  on val → `resolve_threshold` → `compute_detection_metric`. Record per-fold
  metric and feature count.
- `detection_metric` = mean of fold metrics that are not `nan`.
- `penalized_score = detection_metric - alpha * observation_fraction`.
- Infeasible (see Edge cases) → `feasible=False`, `penalized_score =
  EARLY_DETECTION_FLOOR = -1.0`, `detection_metric = nan`.
- Deterministic: same inputs → identical `ConfigScore`.

## Edge cases & gotchas

- **Window clipping**: `window_start + window_size > n_sensors` → clip end to
  `n_sensors`; resulting window always ≥ 1 column.
- **`prefix_end` / `window_size` < 1**, or `window_start` out of `[0,
  n_sensors)` → `ValueError` (validated in `SensorAccess` construction or
  `select_ordered_sensors`).
- **Zero columns survive** preprocessing in a fold, OR zero features after
  selection → that fold is invalid (metric `nan`).
- **Single-class val fold** (no positives or no negatives) → `pr_auc`/`roc_auc`
  = `nan` with `warnings.warn` (do not let sklearn raise); thresholded metrics
  still computed.
- **Infeasible trial**: fewer than 2 valid folds → `feasible=False`, floor
  score `-1.0`. (Do not raise — Spec B's optimizer needs the negative signal.)
- **NaN in raw sensors** → imputed with train-fold medians. **±inf** →
  `ValueError("Sensor matrix contains infinite values")`.
- **Empty `raw_sensor_cols`** → `ValueError("No raw sensor_ columns found")`.
- **`selection_method == "none"`** → `max_features` ignored; all retained
  columns kept.
- **`max_features` > retained count** → clamp `k` to retained count (no raise).

## Acceptance

- `mypy --strict` and `ruff` pass on `early_detection.py` and the test file.
- **Leakage test**: with a fixed fold, `FoldPreprocessor` statistics
  (`retained_cols`, `medians`, `scaler_mean/scale`, `selected_idx`) computed on
  the train indices are byte-identical whether or not the val rows are present
  in the input → proves no val leakage.
- **Access math test**: `observation_fraction`/`latest_index` correct for
  prefix and window (including clipped windows); `select_ordered_sensors`
  returns the right ordered columns.
- **Infeasible-floor test**: a config whose window yields no retained features,
  and one producing < 2 valid folds, both return `feasible=False`,
  `penalized_score == -1.0`, `detection_metric` is `nan`.
- **Single-class guard test**: a fold with a single-class val set yields `nan`
  AUCs via a warning, not an exception.
- **Earliness test**: on synthetic data where a late window is only marginally
  better, raising `alpha` lowers a late config's `penalized_score` below an
  early config's.
- **Determinism test**: two `evaluate_config` calls with identical args return
  equal `ConfigScore`.
- **Real-data sanity check** (documented, run manually): `evaluate_config` with
  a full-range prefix config (`prefix_end = n_sensors`, `selection_method =
  "none"`, `random_forest`) on the SECOM train split returns a mean PR-AUC in
  the neighborhood of the existing tabular baseline (~0.19), confirming the
  pipeline is wired correctly and leakage-free before Spec B is built.
