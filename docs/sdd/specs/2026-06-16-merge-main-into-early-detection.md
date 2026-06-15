# Merge main into dev/early-detection and reconcile early_detection.py

## Goal
Bring 58 commits of `main` into `dev/early-detection` and repair the one
resulting breakage: `early_detection.py` imports preprocess functions that
`main` made private and re-specified. Decision (made up front): migrate the
early-detection study from **variance** filtering to **coefficient-of-variation
(CV)** filtering so it explores the same feature-filter semantics production now
uses, and have `FoldPreprocessor` reuse `SecomPreprocessor` instead of
duplicating the missing→impute→filter→correlation pipeline.

## Background (verified, not assumed)
- `git merge main` into the branch produces **zero textual conflicts**
  (`pyproject.toml` and the shared spec doc auto-merge cleanly), but the merged
  tree raises `ImportError: cannot import name 'drop_high_correlation' from
  'yield_risk.preprocess'`.
- `main` renamed the three free functions to private and re-specified one:
  - `drop_high_missing` → `_drop_high_missing` (same semantics, sensor-only input)
  - `drop_high_correlation` → `_drop_high_correlation` (same semantics)
  - `drop_low_variance` (variance) → `_drop_low_cv` (**coefficient of
    variation = std/|mean|**, with explicit boundary rules: mean==0 & std==0 →
    CV 0.0 → drop; mean==0 & std>0 → CV +inf → keep)
  - all wrapped in `SecomPreprocessor(BaseEstimator, TransformerMixin)` with
    `__init__(missing_threshold, cv_threshold, correlation_threshold)`,
    `fit(X)` (learns `kept_columns_`, `medians_`; pipeline order
    missing→impute(median)→cv→corr), and `transform(X)` (selects kept columns,
    imputes with train medians; returns a DataFrame). `SecomPreprocessor` does
    **not** scale or do feature selection.
- Everything else the branch uses is unchanged on `main` and needs no edits:
  `evaluate.compute_metrics` (still exposes `.pr_auc`/`.roc_auc`),
  `thresholding.expected_cost_at_threshold` (same 4-arg signature),
  `preprocess.split_stratified`, `validation.validate_secom`,
  `config.load_config`/`load_cost_config`/`CostMatrix`, `data.load_secom`.
- `scripts/run_early_detection.py` needs **no changes** — all its imports exist
  on `main`. Production `cv_threshold` is `0.01` (`configs/config.yaml`).
- No early-detection artifacts (`*.json`/`*.csv`/`*.pkl`) are committed, so no
  stored output carries the stale `variance_threshold` key.

## Deliverables
1. **Merge `main` into `dev/early-detection`.** Standard merge commit; no
   textual conflict resolution is expected. All work below lands on top of the
   merged tree.

2. **Fix imports in `src/yield_risk/early_detection.py`.** Replace the
   now-invalid
   `from yield_risk.preprocess import (drop_high_correlation, drop_high_missing,
   drop_low_variance)` with `from yield_risk.preprocess import SecomPreprocessor`.

3. **`FoldPreprocessor` reuses `SecomPreprocessor`.** `FoldPreprocessor.fit`
   constructs `SecomPreprocessor(missing_threshold=cfg.missing_threshold,
   cv_threshold=cfg.cv_threshold, correlation_threshold=cfg.correlation_threshold)`,
   fits it on `x_train_window` to obtain the missing→impute→cv→corr stage, then
   fits a `StandardScaler` (zero-scale entries replaced with `1.0`) and the
   configured feature selector on the kept+imputed columns. `FoldPreprocessor`
   continues to own scaling and selection — those stay in `early_detection.py`.
   - The fitted state stores the fitted `SecomPreprocessor` (or its
     `kept_columns_` + `medians_`), `scaler_mean`, `scaler_scale`, and
     `selected_idx`. **Keep public attributes `retained_cols` (=
     `kept_columns_`) and `selected_idx`** because `evaluate_config` reads
     `preprocessor.retained_cols` and `preprocessor.selected_idx`.
   - `transform(x_window)` delegates column selection + median imputation to the
     fitted `SecomPreprocessor.transform`, then applies the stored scaler stats
     and `selected_idx`. Output stays a float64 `np.ndarray` of shape
     `(n_rows, n_selected)`, identical contract to today.

4. **Rename the `variance_threshold` search dimension to `cv_threshold`
   everywhere in `src/yield_risk/early_detection.py`:**
   - `HyperparamConfig.variance_threshold` → `cv_threshold` (update docstring
     from "Minimum variance…" to the CV definition).
   - `EarlyDetectionConfig.variance_threshold` → `cv_threshold` (the
     `(low, high)` bounds tuple; update docstring).
   - `_DEFAULTS["variance_threshold"]` → `"cv_threshold": [1.0e-3, 1.0]`.
   - `_TUPLE_FLOAT_FIELDS`: replace `"variance_threshold"` with `"cv_threshold"`.
   - `load_early_detection_config`: the `_to_float_tuple` call and the
     `low <= high` bound-pair check use `"cv_threshold"`.
   - `suggest_config`: `trial.suggest_float("cv_threshold", *ed_cfg.cv_threshold,
     log=True)` (keep `log=True`); assign to `HyperparamConfig(cv_threshold=…)`.
   - `hyperparam_config_to_dict`: emit key `"cv_threshold"`.

5. **Update `configs/early_detection_config.yaml`:** replace line
   `variance_threshold: [1.0e-6, 1.0e-2]` with
   `cv_threshold: [1.0e-3, 1.0]   # [low, high] coefficient-of-variation bounds`.

6. **Update `tests/test_early_detection.py`:** rename every `variance_threshold`
   reference to `cv_threshold` (kwargs, dict keys, helper params, comments/test
   names). Update the two assertions that pin the YAML default bounds
   (currently `== pytest.approx((1.0e-6, 1.0e-2))` and the
   `lo <= cfg.variance_threshold <= hi` check) to the new CV bounds
   `(1.0e-3, 1.0)`. The "drop-all" sentinel values (e.g. `1e10`) and the
   constant-sensor-survives case carry over unchanged in intent under CV.

## Edge cases & gotchas
- **CV boundary rules come from `_drop_low_cv`, not custom code.** A constant
  non-zero sensor (std=0, mean≠0) has CV=0.0; with `cv_threshold=0.0` the test
  `0.0 < 0.0` is False, so it is **kept** — matching the existing
  `test_fold_preprocessor_zero_variance_scale_is_one` expectation. Its
  `StandardScaler` scale is 0.0 → replaced with 1.0. Do not reintroduce a
  variance computation in `early_detection.py`.
- **Empty fold after filtering.** When `SecomPreprocessor` keeps zero columns
  (`kept_columns_ == []`), `FoldPreprocessor.fit` must return the same empty
  state it returns today (empty `retained_cols`, empty `selected_idx`, empty
  scaler arrays) without raising, so `evaluate_config` marks the fold invalid.
  The drop-all sentinel test (`cv_threshold` ≈ `1e10`) must still yield an
  infeasible `ConfigScore` with `penalized_score == EARLY_DETECTION_FLOOR` and
  `detection_metric == nan`, not an exception.
- **Leakage boundary preserved.** `SecomPreprocessor.fit` learns medians and
  kept columns from the training fold only; the scaler and selector are fit on
  the train fold only. The leakage-boundary test (fit on a train slice ==
  fit on the full frame restricted to those rows) must still pass.
- **`max_features` greater than retained count** must still not raise (selector
  uses `k = min(max_features, n_retained)` as today).
- **`SecomPreprocessor.transform` returns a DataFrame**; convert to a float64
  ndarray before scaling/selection so the `(n_rows, n_selected)` ndarray
  contract is preserved.
- **No stray `variance` left.** After the change, grep of `src/` and
  `configs/` and `tests/` for `variance` returns nothing related to the
  early-detection search dimension (the word may still appear in unrelated
  main code/comments, which is fine).
- **`build_best_record` round-trip.** It reconstructs the winning config via
  `suggest_config` on the frozen best trial; the Optuna param name is now
  `cv_threshold`, and `hyperparam_config_to_dict` emits `cv_threshold` — these
  must agree so the serialized record round-trips.

## Acceptance
- `git merge main` is committed on `dev/early-detection`.
- `python -c "import yield_risk.early_detection"` succeeds (no ImportError).
- `ruff check src/ tests/` passes; `mypy src/yield_risk` passes
  (`mypy --strict` clean per CLAUDE.md).
- `pytest tests/test_early_detection.py` passes; full `pytest` passes (the merged
  `main` suite plus the renamed early-detection suite).
- `grep -rn "variance_threshold" src/ tests/ configs/` returns no matches.
- `configs/early_detection_config.yaml` has `cv_threshold: [1.0e-3, 1.0]` and no
  `variance_threshold` key; `load_early_detection_config()` loads it and
  `EarlyDetectionConfig.cv_threshold == (1.0e-3, 1.0)`.
- `FoldPreprocessor` contains no variance/CV math of its own — the filter stage
  is delegated to `SecomPreprocessor`; only scaling and selection remain local.
