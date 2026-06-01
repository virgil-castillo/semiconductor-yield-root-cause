# Tier 4 Monitoring and Reporting Design

## Context

Tier 4 adds deterministic monitoring utilities, report generation, and written
reports for the SECOM yield-risk system. The current repository already includes
processed train/test splits, trained model artifacts, model comparison metrics,
selected-model metrics, SHAP outputs, and root-cause candidate rankings. Tier 4
should turn those artifacts into reproducible monitoring evidence and durable
written reports without re-opening the established model selection protocol.

The selected model remains the random forest chosen by training-only 5-fold
cross-validation PR-AUC. XGBoost held-out performance is reported as comparison
evidence, not as a post-test model reselection. Root-cause outputs remain
candidate triage signals, not causal claims.

## Goals

- Add lightweight, deterministic monitoring utilities for missingness drift,
  numeric feature drift, prediction distribution drift, and high-risk-rate drift.
- Add report generation utilities that render markdown reports from existing
  artifacts and processed data summaries.
- Add a script that generates Tier 4 markdown reports from configured paths.
- Add an empirical monitoring notebook that calls the monitoring utilities and
  documents static-batch drift evidence.
- Add focused tests for monitoring and reporting behavior.
- Preserve strict typing, Google-style public docstrings, and existing code
  conventions.

## Non-Goals

- Do not add live services, schedulers, online monitoring stores, dashboards, or
  external observability dependencies.
- Do not train or select a new model.
- Do not require notebook execution for report generation.
- Do not make physical process root-cause claims from anonymous SECOM sensors.

## Approach

Tier 4 will use a tested library-first implementation with a report-style
notebook as empirical confirmation. The library utilities provide deterministic
monitoring APIs, the report generator produces reproducible markdown outputs,
and the notebook demonstrates the same code on real project artifacts.

## Architecture

### Monitoring Module

Add `src/yield_risk/monitoring.py` with small dataclasses and pure functions:

- `MissingnessDriftResult`
- `FeatureDriftResult`
- `PredictionDriftResult`
- `HighRiskRateDriftResult`
- `MonitoringSummary`

Public functions:

- `compare_missingness(reference, current, feature_cols, threshold=0.05)`
- `compare_feature_distributions(reference, current, feature_cols, threshold=0.1)`
- `compare_prediction_distributions(reference_scores, current_scores, threshold=0.1)`
- `compare_high_risk_rate(reference_scores, current_scores, threshold, rate_delta_threshold=0.05)`
- `build_monitoring_summary(reference, current, feature_cols, reference_scores, current_scores, high_risk_threshold)`

The functions operate on in-memory pandas objects or numpy arrays and return
typed dataclasses. They do not read files, write files, or depend on model
artifacts.

Feature distribution drift should compute deterministic numeric summaries:

- reference/current mean
- absolute mean delta
- reference/current standard deviation
- absolute standard deviation delta
- reference/current median
- absolute median delta
- Kolmogorov-Smirnov statistic and p-value when both compared columns contain
  enough non-null values
- an `alert` boolean based on a configurable threshold

Missingness drift should compare per-feature null rates and flag features whose
absolute missing-rate delta exceeds the configured threshold.

Prediction drift should compare score distributions with mean, median, standard
deviation, selected quantiles, KS statistic, and an alert boolean.

High-risk-rate drift should compare the fraction of wafers whose score is at or
above the operating threshold and flag excessive rate changes.

### Reporting Module

Add `src/yield_risk/reporting.py` with pure rendering helpers and a small
orchestration function:

- Load model comparison, selected metrics, root-cause candidates, sensitivity
  summary/detail, processed data summaries, and monitoring summaries.
- Render markdown strings for:
  - `executive_summary.md`
  - `model_card.md`
  - `data_card.md`
  - `root_cause_report.md`
- Write those markdown strings to target paths.

The rendered reports must explicitly state:

- The selected model is random forest by training-only CV PR-AUC.
- XGBoost held-out results are comparison evidence only.
- Sensor names are anonymous, so root-cause candidates cannot be physically
  interpreted without fab metadata.
- The monitoring outputs are static-batch demonstrations built from available
  SECOM artifacts, not live production telemetry.

### Report Generation Script

Add `scripts/generate_reports.py`.

Default behavior:

- Load `configs/config.yaml`.
- Read processed train/test data from `data/processed`.
- Read model and report artifacts from configured paths.
- Score deterministic reference/current batches with `models/selected_model.joblib`.
- Build a monitoring summary from the selected model scores.
- Generate the four markdown reports under `reports/`.

The script should be safe to rerun. It should overwrite generated markdown
reports with current artifact-derived content.

### Monitoring Notebook

Add `notebooks/07_model_monitoring_drift_checks.ipynb`.

The notebook should be a report-style empirical confirmation of Tier 4:

1. Load config, processed test data, and selected model.
2. Create deterministic reference/current batches from the held-out test split.
3. Score both batches with the selected model.
4. Call `yield_risk.monitoring` utilities.
5. Display missingness drift, top feature drift rows, prediction drift, and
   high-risk-rate drift.
6. Interpret results with clear caveats:
   - static historical SECOM data
   - anonymous sensors
   - monitoring demonstration only
   - no causal or live-production claims

The notebook must not duplicate drift logic from `monitoring.py`; it should call
the library utilities and focus on evidence and interpretation.

## Data Flow

1. `scripts/generate_reports.py` loads config and existing artifacts.
2. Processed test data is split deterministically into reference/current demo
   batches.
3. The selected model scores both batches.
4. `monitoring.py` computes drift summaries from features and scores.
5. `reporting.py` renders markdown reports from metrics, root-cause artifacts,
   processed data summaries, and monitoring results.
6. The notebook independently demonstrates the same monitoring API on the same
   artifact family for human-readable inspection.

## Error Handling

- Monitoring functions raise `ValueError` for empty batches, missing feature
  columns, invalid thresholds, or score arrays with incompatible shapes.
- Reporting helpers raise `FileNotFoundError` for required missing artifacts and
  `ValueError` for malformed inputs such as missing required columns.
- The generation script should fail loudly with actionable messages rather than
  silently producing partial reports.

## Testing

Add focused tests:

- `tests/test_monitoring.py`
  - missingness drift flags expected columns
  - feature distribution drift computes stable deltas and alert booleans
  - prediction drift handles changed score distributions
  - high-risk-rate drift uses the supplied operating threshold
  - invalid inputs raise clear exceptions
- `tests/test_reporting.py`
  - report renderers include selected-model protocol language
  - root-cause report includes candidate triage caveats
  - markdown files are written to expected paths
  - missing required report columns raise `ValueError`

Before completion, run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
ruff check src/ tests/
mypy src/yield_risk
pytest
```

## Deliverables

- `src/yield_risk/monitoring.py`
- `src/yield_risk/reporting.py`
- `scripts/generate_reports.py`
- `notebooks/07_model_monitoring_drift_checks.ipynb`
- `reports/executive_summary.md`
- `reports/model_card.md`
- `reports/data_card.md`
- `reports/root_cause_report.md`
- `tests/test_monitoring.py`
- `tests/test_reporting.py`
- `tmp/tier4-pr-report.md`

## Acceptance Criteria

- Tier 4 monitoring utilities are deterministic and unit-tested.
- Markdown reports are generated from current artifacts instead of hand-maintained
  stale text.
- The empirical notebook confirms monitoring behavior using project artifacts and
  shared library APIs.
- Reports preserve the established model-selection narrative and limitations.
- No implementation text claims physical causality from anonymous sensors.
- `ruff check src/ tests/`, `mypy src/yield_risk`, and `pytest` pass in the
  `mlops` conda environment, or any verification failure is documented with the
  exact command and reason.
