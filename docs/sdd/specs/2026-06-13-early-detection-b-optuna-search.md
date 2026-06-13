# SDD Spec B: Early-Detection Optuna Search, CLI & Study Artifacts

> Slice 2 of 3. Umbrella design:
> [`2026-06-12-early-detection-bayesian-optimization.md`](2026-06-12-early-detection-bayesian-optimization.md).
> Builds on Spec A (the leakage-safe `evaluate_config` scoring core, already
> implemented and reviewed). This spec wraps that core in an Optuna study with a
> committed config, a CLI, and persisted study artifacts.

## Goal

Add the Bayesian-optimization layer on top of Spec A: sample one
`HyperparamConfig` per Optuna trial across the full conditional/categorical
search space, score it with `evaluate_config`, and run a `TPESampler` study on a
held-out **train** split (the test split is created but never touched here).
Persist the study, the winning configuration, and the trials table. Ship a
committed default config, a config loader, and a CLI entry point.

## Assumption

Unchanged from Spec A: raw SECOM column order is asserted as a stand-in for
fabrication progress. "Earliness" means earlier *in column order*; this is a
declared modeling choice, not a property of the data.

## Scope & B/C boundary

- **Appends to** `src/yield_risk/early_detection.py` (do not modify Spec A's
  existing symbols; add new ones below them). **Appends to**
  `tests/test_early_detection.py`.
- **New files:** `configs/early_detection_config.yaml`,
  `scripts/run_early_detection.py`.
- **This spec OWNS:** the search space + `suggest_config`, the objective, study
  orchestration, the outer train/test split (test held out, untouched), the
  config loader, the CLI, and these artifacts only:
  `early_detection_study.pkl`, `early_detection_best.json`,
  `early_detection_trials.csv`.
- **This spec DEFERS to Spec C:** refitting the winner on full train, the
  one-time test-holdout evaluation, the full-feature baseline, the comparison
  report, the early-detection curve, `early_detection_metrics.json`,
  `early_detection_comparison.{json,csv}`, `early_detection_curve.png`, and
  `docs/early_detection.md`. Spec C extends the same CLI script. `best.json`
  (incl. split provenance) is the handoff so C can reproduce the identical
  holdout.
- **MUST NOT** modify tabular pipeline modules, `evaluate.ClassificationMetrics`,
  existing scripts, or any existing artifact in `models/` or `reports/`.
- Dev contract (all new/changed code): `from __future__ import annotations`;
  Google-style docstrings (Args/Returns/Raises); full annotations; `mypy
  --strict` clean; ruff `E,F,W,I,UP,ANN` @ line length 88.

## Reuse (do not reimplement)

- Spec A's `SensorAccess`, `HyperparamConfig`, `evaluate_config`,
  `EARLY_DETECTION_FLOOR`, `observation_fraction`, `latest_index`.
- `yield_risk.data.load_secom`, `yield_risk.validation.validate_secom`.
- `yield_risk.preprocess.split_stratified` for the outer holdout.
- `yield_risk.config.load_config` (paths + `run.test_size`/`run.random_seed`),
  `load_cost_config` **only** when `detection_metric == "neg_expected_cost"`.

## Dependency

- Add `"optuna>=3.6"` to `pyproject.toml` `dependencies` (Optuna 4.8 is already
  installed in `mlops`). Add a `[[tool.mypy.overrides]]` block for
  `module = "optuna.*"` with `ignore_missing_imports = true`, matching the
  existing overrides.

## Deliverables

### 1. `EarlyDetectionConfig` + loader

```python
@dataclass(frozen=True)
class EarlyDetectionConfig:
    n_trials: int
    sampler_seed: int
    inner_cv_folds: int
    detection_metric: str
    alpha: float
    access_types: list[str]
    min_window_size: int
    max_window_size: int
    model_families: list[str]
    missing_threshold: tuple[float, float]       # (low, high) search bounds
    variance_threshold: tuple[float, float]
    correlation_threshold: tuple[float, float]
    selection_methods: list[str]
    max_features: tuple[int, int]
    threshold_policy: str                         # study-wide, NOT sampled
    threshold_range: tuple[float, float]
    false_alarm_rate: float
    performance_tolerance: float                  # consumed by Spec C
    curve_prefixes: list[int]                     # consumed by Spec C

def load_early_detection_config(
    path: Path | str = "configs/early_detection_config.yaml",
    *,
    overrides: Mapping[str, object] | None = None,
) -> EarlyDetectionConfig: ...
```

- **Missing file → all defaults** (do not raise); the shipped YAML mirrors the
  defaults so behavior is identical whether or not the file is present.
- **Unknown YAML key → `ValueError`** listing the offending key(s).
- `overrides` (the CLI passes parsed flags here) is applied last and wins;
  `None`/absent override values are ignored (do not clobber with `None`).
- **Validation at load** (raise `ValueError` with a clear message):
  `detection_metric` ∈ {`pr_auc`, `recall_at_far`, `neg_balanced_error`,
  `neg_expected_cost`}; `threshold_policy` ∈ {`tune`, `far_constraint`};
  every entry of `access_types` ∈ {`prefix`, `window`} and the list non-empty;
  `model_families` non-empty and ⊆ the four Spec-A families;
  `selection_methods` non-empty and ⊆ {`none`, `univariate`, `mutual_info`,
  `model_importance`}; `n_trials ≥ 1`; `inner_cv_folds ≥ 2`. Two-element
  `[low, high]` lists load into the tuple fields and must satisfy `low ≤ high`.

### 2. Search space — `suggest_config`

```python
def suggest_config(
    trial: optuna.Trial, ed_cfg: EarlyDetectionConfig, n_sensors: int
) -> HyperparamConfig: ...
```

Samples and returns a fully-populated Spec-A `HyperparamConfig`. Use stable
Optuna parameter names (below) so `trials_dataframe()` columns are meaningful.

- **Sensor access** — `access_type` = `suggest_categorical("access_type",
  ed_cfg.access_types)`.
  - `prefix` → `prefix_end = suggest_int("prefix_end", 1, n_sensors)`.
  - `window` → `window_start = suggest_int("window_start", 0, n_sensors - 1)`;
    `ws_high = min(ed_cfg.max_window_size, n_sensors)`,
    `ws_low = min(ed_cfg.min_window_size, ws_high)`,
    `window_size = suggest_int("window_size", ws_low, ws_high)`. (Downstream
    slicing clips the end to `n_sensors`; Spec A handles it.)
- **Preprocessing** — `missing_threshold = suggest_float("missing_threshold",
  *ed_cfg.missing_threshold)`; `variance_threshold =
  suggest_float("variance_threshold", *ed_cfg.variance_threshold, log=True)`
  (range spans orders of magnitude); `correlation_threshold =
  suggest_float("correlation_threshold", *ed_cfg.correlation_threshold)`.
- **Feature selection** — `selection_method =
  suggest_categorical("selection_method", ed_cfg.selection_methods)`. If
  `"none"` → `max_features = None` (do not sample). Else `max_features =
  suggest_int("max_features", *ed_cfg.max_features)` (Spec A clamps `k` to the
  retained count, so an over-large value is safe).
- **Model family + hyperparameters** — `model_family =
  suggest_categorical("model_family", ed_cfg.model_families)`, then per-family
  params into `model_params` using the **module-level constant ranges** below
  (constants, not config — keeps the YAML small). Prefix every param name with
  the family (e.g. `"rf_n_estimators"`) so conditional params don't collide
  across families in the trials table.
  - `logistic_regression`: `C` ∈ `suggest_float("lr_C", 1e-3, 1e2, log=True)`;
    `penalty` ∈ `suggest_categorical("lr_penalty", ["l1", "l2"])`;
    `class_weight` ∈ `suggest_categorical("lr_class_weight", [None,
    "balanced"])`. When `penalty == "l1"`, set `model_params["solver"] =
    "liblinear"` (lbfgs, the Spec-A default, supports only l2).
  - `random_forest`: `n_estimators` ∈ `suggest_int("rf_n_estimators", 100,
    800)`; `max_depth` ∈ `suggest_int("rf_max_depth", 2, 32)`;
    `min_samples_leaf` ∈ `suggest_int("rf_min_samples_leaf", 1, 20)`;
    `max_features` ∈ `suggest_categorical("rf_max_features", ["sqrt", "log2"])`;
    `class_weight` ∈ `suggest_categorical("rf_class_weight", [None, "balanced",
    "balanced_subsample"])`. (Note: this RF `max_features` model param is
    distinct from the feature-selection `max_features` field.)
  - `xgboost` / `lightgbm`: `learning_rate` ∈ `suggest_float("<fam>_learning_rate",
    1e-3, 3e-1, log=True)`; `n_estimators` ∈ `suggest_int("<fam>_n_estimators",
    100, 800)`; `max_depth` ∈ `suggest_int("<fam>_max_depth", 2, 12)`;
    `subsample` ∈ `suggest_float("<fam>_subsample", 0.5, 1.0)`. **Do NOT sample
    or set `scale_pos_weight`** — leaving it out lets `evaluate_config` default
    it to the train-fold `n_neg/n_pos` per fold (Spec A behavior).
- **Threshold** — `threshold_policy` is taken from `ed_cfg` (study-wide; this
  resolves the umbrella's apparent "sampled vs config" contradiction). If
  `"tune"` → `threshold = suggest_float("threshold", *ed_cfg.threshold_range)`,
  `false_alarm_rate = None`. If `"far_constraint"` → `threshold = None`,
  `false_alarm_rate = ed_cfg.false_alarm_rate` (no per-trial threshold sample).

### 3. Objective + study orchestration

```python
def run_study(
    x_train: pd.DataFrame, y_train: np.ndarray, raw_sensor_cols: list[str],
    ed_cfg: EarlyDetectionConfig, *, random_seed: int,
    cost_matrix: CostMatrix | None = None,
) -> optuna.Study: ...
```

- `sampler = optuna.samplers.TPESampler(seed=ed_cfg.sampler_seed)`;
  `study = optuna.create_study(direction="maximize", sampler=sampler)`.
- Objective per trial: `cfg = suggest_config(trial, ed_cfg,
  len(raw_sensor_cols))`; `score = evaluate_config(x_train, y_train,
  raw_sensor_cols, cfg, inner_cv_folds=ed_cfg.inner_cv_folds,
  alpha=ed_cfg.alpha, detection_metric=ed_cfg.detection_metric,
  random_seed=random_seed, cost_matrix=cost_matrix)`. Record via
  `trial.set_user_attr`: `detection_metric` (raw, may be `nan`),
  `observation_fraction`, `n_features_selected`, `feasible`, `access_type`,
  `latest_index`, `model_family`. Return `score.penalized_score`.
- `study.optimize(objective, n_trials=ed_cfg.n_trials, n_jobs=1)` — **`n_jobs=1`
  is required for determinism**. The objective never raises for infeasible
  trials (Spec A returns the floor `-1.0`); do not use `TrialPruned`.
- `cost_matrix` is required iff `detection_metric == "neg_expected_cost"`;
  raise `ValueError` if it is `None` in that case (fail fast before optimizing).

### 4. Best-config record (handoff artifact)

```python
def build_best_record(
    study: optuna.Study, ed_cfg: EarlyDetectionConfig, *,
    n_sensors: int, test_size: float, random_seed: int,
) -> dict[str, object]: ...
```

Returns a JSON-serializable dict with:
- `best_trial_number`, `penalized_score`, and the best trial's user attrs
  (raw `detection_metric`, `observation_fraction`, `n_features_selected`,
  `feasible`, `latest_index`).
- `config`: the winning `HyperparamConfig` fully serialized (nested
  `SensorAccess` → dict; `model_params` verbatim; `None` preserved).
- `provenance`: `n_sensors`, `n_trials`, `sampler_seed`, `inner_cv_folds`,
  `detection_metric`, `alpha`, `threshold_policy`, `test_size`, `random_seed`.
  (`test_size` + `random_seed` let Spec C reproduce the identical holdout.)
- A reusable `hyperparam_config_to_dict(cfg: HyperparamConfig) -> dict[str,
  object]` helper performs the serialization (Spec C reads it back).
- **`nan` → `null`** in all JSON output: sanitize floats before
  `json.dumps` (do not rely on `allow_nan`, which emits invalid `NaN` tokens).

### 5. CLI — `scripts/run_early_detection.py`

- `argparse` flags, all optional: `--config` (path, default
  `configs/early_detection_config.yaml`), `--n-trials` (int), `--alpha`
  (float), `--detection-metric` (str), `--seed` (int → overrides
  `sampler_seed`). Only flags the user actually passes go into `overrides`.
- Flow: `cfg = load_config()`; `ed_cfg = load_early_detection_config(args.config,
  overrides=...)`; `df = load_secom(cfg.paths.raw_dir)`; `validate_secom(df)`;
  `raw_sensor_cols = [c for c in df.columns if c.startswith("sensor_")]` (column
  order = progress axis); `train_df, _test_df = split_stratified(df,
  cfg.run.test_size, cfg.run.random_seed)` — **the test split is discarded
  here, untouched**; `x_train = train_df[raw_sensor_cols]`, `y_train =
  train_df["label"].to_numpy()`; load cost matrix only when needed; `study =
  run_study(...)`; write the three artifacts; print a short summary (best trial
  number, penalized score, raw detection metric, observation fraction, latest
  sensor index, model family).
- `models_dir` and `reports_dir` are `mkdir(parents=True, exist_ok=True)` first.
- **`n_sensors == 0`** (no `sensor_` columns) → `ValueError("No raw sensor_
  columns found")`; CLI catches, prints to stderr, exits with code 1.
- `if __name__ == "__main__": main()`.

### 6. Artifacts (namespaced; relative to `cfg.paths`)

- `models_dir / "early_detection_study.pkl"` — `joblib.dump(study, ...)`.
- `models_dir / "early_detection_best.json"` — `build_best_record(...)`,
  `json.dumps(..., indent=2)`.
- `reports_dir / "early_detection_trials.csv"` —
  `study.trials_dataframe().to_csv(index=False)`.

### 7. Config file — `configs/early_detection_config.yaml`

Ship exactly these defaults (identical to the umbrella; `[low, high]` lists map
to the tuple fields):

```yaml
n_trials: 100
sampler_seed: 42
inner_cv_folds: 5
detection_metric: pr_auc        # pr_auc | recall_at_far | neg_balanced_error | neg_expected_cost
alpha: 0.10                     # earliness penalty weight
access_types: [prefix, window]
min_window_size: 8
max_window_size: 256
model_families: [logistic_regression, random_forest, xgboost, lightgbm]
missing_threshold: [0.2, 0.6]   # [low, high] search bounds
variance_threshold: [1.0e-6, 1.0e-2]
correlation_threshold: [0.85, 0.99]
selection_methods: [none, univariate, mutual_info, model_importance]
max_features: [10, 200]
threshold_policy: tune          # tune | far_constraint
threshold_range: [0.01, 0.80]
false_alarm_rate: 0.10
performance_tolerance: 0.05      # fraction of baseline primary metric (Spec C)
curve_prefixes: [16, 32, 64, 128, 256, 512]
```

## Edge cases & gotchas

- **Missing config file** → all defaults; no raise. **Unknown YAML key** →
  `ValueError` listing the key(s). **`low > high`** in any bound pair, empty
  `access_types`/`model_families`/`selection_methods`, bad
  `detection_metric`/`threshold_policy`, `n_trials < 1`, or `inner_cv_folds < 2`
  → `ValueError` at load.
- **CLI override of `None`** must not clobber a real config value — only
  user-supplied flags enter `overrides`.
- **`min_window_size > n_sensors`** (e.g. tiny synthetic data) → clamp as in §2
  (`ws_high = min(max_window_size, n_sensors)`, `ws_low = min(min_window_size,
  ws_high)`) so `suggest_int` bounds are valid (`low ≤ high`).
- **`access_types == ["window"]` with `n_sensors == 1`** → `window_start`
  range is `[0, 0]`; still valid.
- **All trials infeasible** (every `penalized_score == -1.0`) → `study.best_trial`
  still returns one; `best.json` is written with `feasible: false` and
  `detection_metric: null`. Do not raise.
- **`detection_metric == "neg_expected_cost"` without a cost matrix** →
  `run_study` raises `ValueError` before optimizing; the CLI loads the cost
  matrix via `load_cost_config()` only for this metric.
- **`scale_pos_weight`** is never sampled (see §2) — Spec A sets it per fold.
- **`l1` penalty + lbfgs** is incompatible → `suggest_config` switches the
  solver to `liblinear` whenever `penalty == "l1"`.
- **`nan` anywhere in JSON** → serialized as `null` (sanitize, don't use
  `allow_nan`).
- **Determinism**: `TPESampler(seed=sampler_seed)` + `n_jobs=1` + Spec A's
  seeded folds/estimators ⇒ two runs with the same config produce the same best
  trial, the same `best.json`, and the same `trials.csv`.
- **No existing artifact touched**: only the three namespaced files in §6 are
  written; nothing under `models/`/`reports/` is read or overwritten besides
  them.

## Acceptance

- `pyproject.toml` declares `optuna>=3.6` and the `optuna.*` mypy override;
  `ruff` and `mypy --strict` pass on all new/changed files.
- `load_early_detection_config`: returns the documented defaults on a missing
  file; raises `ValueError` on an unknown key, on a `low > high` bound, on an
  empty `model_families`, and on a bad `detection_metric`/`threshold_policy`;
  applies CLI overrides last (tested), ignoring unset (`None`) ones.
- `suggest_config` (driven by a fixed/fake or seeded `optuna.Trial`): produces a
  valid `HyperparamConfig` for each `access_type`, each `selection_method`
  (incl. `none → max_features is None`), and each `model_family`; sets
  `solver="liblinear"` when `penalty=="l1"`; never sets `scale_pos_weight`;
  clamps window bounds when `min_window_size > n_sensors`.
- `run_study` on a small synthetic dataset returns an `optuna.Study` with
  `len(study.trials) == n_trials`, every trial carries the documented user
  attrs, and infeasible trials score exactly `EARLY_DETECTION_FLOOR` without
  raising. Running it twice with the same seed yields identical best params and
  identical per-trial values (determinism test).
- `run_study` raises `ValueError` when `detection_metric ==
  "neg_expected_cost"` and `cost_matrix is None`.
- `build_best_record` round-trips: the serialized `config` reconstructs an equal
  `HyperparamConfig` (via `hyperparam_config_to_dict` + inverse), and every
  `nan` is emitted as JSON `null` (a `json.loads` of the dumped string
  succeeds).
- End-to-end: `python scripts/run_early_detection.py --n-trials 5` on real SECOM
  data writes `early_detection_study.pkl`, `early_detection_best.json`, and
  `early_detection_trials.csv` under the configured dirs, touches no existing
  artifact, and never loads the test split (the test rows from
  `split_stratified` are discarded in this slice).
- Tests cover: config load (defaults / unknown-key raise / validation raises /
  CLI override precedence), `suggest_config` per branch, `run_study` trial
  count + user attrs + determinism + cost-matrix guard, infeasible-trial floor
  through the study, and `best.json` serialization round-trip incl. `nan → null`.
```