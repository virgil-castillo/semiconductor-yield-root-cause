# Early-Detection Spec C: Held-Out Evaluation, Comparison, and Curve

## Goal

Finish the current early-detection branch by turning the completed Optuna study
artifact into held-out evidence. Refit the selected early configuration on the
full train split, evaluate it once on the reproduced test split, compare it to
full-sensor references, write the deferred reports/figure, and update the
early-detection report.

This spec does not add multi-objective Optuna and does not run a second
full-feature Optuna search. Those are follow-on experiments. This slice answers:
"Did the scalar Optuna winner generalize to held-out wafers, and how much sensor
access did it save?"

## Deliverables

- Append to `src/yield_risk/early_detection.py`:
  - `hyperparam_config_from_dict(data: Mapping[str, object]) -> HyperparamConfig`
    that round-trips the `config` object from `models/early_detection_best.json`.
  - A fitted-model helper, e.g. `fit_early_detector(...)`, that:
    - slices train data with `select_ordered_sensors`,
    - fits `FoldPreprocessor` on full train only,
    - applies the same `scale_pos_weight` default used by `evaluate_config` for
      xgboost/lightgbm when not explicitly present,
    - fits the estimator from `build_estimator`,
    - freezes the operating threshold from training information only.
  - A prediction/evaluation helper that applies the fitted preprocessor/model to
    test data and returns JSON-safe metrics.
  - An orchestration helper, e.g. `evaluate_best_on_holdout(...)`, that returns
    all rows needed for metrics, comparison, and curve artifacts.
- Extend `scripts/run_early_detection.py`:
  - Default path remains end-to-end: run the study, write Spec B artifacts, then
    write Spec C artifacts.
  - Add `--evaluate-existing` to skip Optuna and evaluate the existing
    `models/early_detection_best.json`. This is the intended path for the
    current completed 100-trial study.
- Write artifacts under configured paths:
  - `reports/early_detection_metrics.json`
  - `reports/early_detection_comparison.json`
  - `reports/early_detection_comparison.csv`
  - `reports/early_detection_curve.csv`
  - `reports/figures/early_detection_curve.png`
  - update `docs/early_detection.md` from "deferred" to the actual held-out
    result summary.
- Comparison rows:
  - `early_optuna_best`: the best config from `early_detection_best.json`.
  - `full_prefix_same_config`: same non-access config, but
    `SensorAccess(access_type="prefix", prefix_end=n_sensors)`. This isolates
    the cost of stopping early without rerunning search.
  - `tabular_selected_model`: optional existing full-feature production
    baseline from `reports/model_comparison.json` selected row, skipped if the
    file or selected row is missing.
- Each comparison row includes at least:
  `model`, `role`, `test_pr_auc`, `test_roc_auc`, `test_precision`,
  `test_recall`, `test_f1`, `test_balanced_accuracy`, `test_false_alarm_rate`,
  `expected_cost`, `threshold`, `confusion_matrix`, `n_sensors_used`,
  `sensor_fraction`, `latest_index`, and `source`.
- `early_detection_comparison.json` also includes top-level
  `primary_metric: "test_pr_auc"`, `performance_tolerance`,
  `sensor_count`, `test_size`, `random_seed`, and `verdict`.
  Verdict is:
  - `early_model_competitive` when the early model is within
    `performance_tolerance * baseline_pr_auc` of the best available
    full-sensor reference and uses fewer sensors.
  - `baseline_preferred` otherwise.
  - `no_baseline` only when no full-sensor reference row can be produced.

## Threshold and Leakage Rules

- Never tune or select a threshold on test data.
- For `threshold_policy == "tune"`, use the threshold serialized in the winning
  `HyperparamConfig`.
- For `threshold_policy == "far_constraint"`, generate out-of-fold train
  probabilities with the fixed config and resolve the FAR threshold from those
  train-only predictions. If no valid out-of-fold predictions exist, raise
  `ValueError("Cannot resolve final threshold from training data")`.
- The train/test split must be reproduced from `early_detection_best.json`
  provenance (`test_size`, `random_seed`). Do not silently use a different
  current config value.
- If `provenance["n_sensors"]` differs from the loaded raw sensor count, raise
  `ValueError("early_detection_best.json n_sensors does not match data")`.

## Early-Detection Curve

- Build a descriptive held-out curve for prefix lengths from
  `ed_cfg.curve_prefixes`, plus the best early `latest_index`, plus
  `n_sensors`.
- Keep only unique prefix lengths in `[1, n_sensors]`, sorted ascending; raise
  if the resulting list is empty.
- For every prefix, reuse the winning non-access settings, refit on full train,
  evaluate on the same test holdout, and write one row to
  `early_detection_curve.csv`.
- Plot held-out `test_pr_auc` versus `latest_index`; mark the selected early
  model and draw horizontal references for available full-sensor comparison
  rows. The curve is diagnostic only and must not feed back into model choice.

## Edge Cases and Gotchas

- Missing `models/early_detection_best.json` with `--evaluate-existing` raises
  `FileNotFoundError` with the path in the message.
- Best record with `feasible: false` raises
  `ValueError("Cannot evaluate an infeasible early-detection best trial")`.
- No `sensor_` columns raises `ValueError("No raw sensor_ columns found")`.
- Test holdout with a single class raises
  `ValueError("Test holdout must contain both classes")` before calling sklearn
  AUC metrics.
- Zero retained or selected features after fitting full train raises
  `ValueError("Early detector retained zero features")`.
- A fitted model whose `predict_proba` output has fewer than two columns raises
  `ValueError("Estimator did not produce positive-class probabilities")`.
- JSON artifacts must not contain `NaN`, `Infinity`, or `-Infinity`; reuse the
  existing `_sanitize` behavior or equivalent.
- Existing Spec B artifacts may be read, and namespaced Spec C artifacts may be
  overwritten. Do not overwrite unrelated tabular artifacts such as
  `reports/model_comparison.json`, `models/selected_model.joblib`, or existing
  standard figures.

## Acceptance

- `python scripts/run_early_detection.py --evaluate-existing` on the completed
  100-trial study writes all Spec C artifacts without rerunning Optuna.
- `early_detection_metrics.json` contains held-out metrics for
  `early_optuna_best` and `full_prefix_same_config`, and optionally the selected
  tabular baseline when `reports/model_comparison.json` exists.
- `early_detection_comparison.{json,csv}` include the documented columns and a
  valid verdict.
- `reports/figures/early_detection_curve.png` and
  `reports/early_detection_curve.csv` are produced from the documented prefix
  sweep.
- `docs/early_detection.md` states that the earlier inner-CV values were not
  held-out metrics and reports the new held-out results.
- Tests cover config deserialization, train-only threshold freezing,
  `--evaluate-existing`, missing/mismatched best artifact errors, zero-feature
  guard, comparison schema, verdict logic, curve prefix handling, and JSON
  sanitization.
- `ruff check src/ tests/ scripts/` passes, `mypy src/yield_risk` passes, and
  `pytest tests/test_early_detection.py` passes.
