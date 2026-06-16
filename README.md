# Semiconductor Yield-Risk Scoring and Root-Cause Candidate Ranking

Yield-risk scoring and root-cause candidate ranking on
the UCI SECOM semiconductor manufacturing benchmark. The system builds a
training-safe pipeline for wafer pass/fail prediction, cost-sensitive threshold
selection, sensor ranking, batch scoring, and monitoring/reporting checks.

---

## Table of contents

1. [Business problem](#business-problem)
2. [Why yield prediction matters](#why-yield-prediction-matters)
3. [Key results](#key-results)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Modeling approach](#modeling-approach)
7. [Root-cause sensor ranking](#root-cause-sensor-ranking)
8. [Cost-sensitive decision policy](#cost-sensitive-decision-policy)
9. [Dashboard](#dashboard)
10. [API](#api)
11. [Reproducing the project](#reproducing-the-project)
12. [Repository structure](#repository-structure)
13. [Limitations](#limitations)
14. [Future work](#future-work)

---

## Business problem

A semiconductor manufacturing line produces wafers through hundreds of sequential
process steps. Each wafer accumulates risk as it moves through diffusion, lithography,
etch, deposition, and metrology stages. By the time a wafer reaches final line test,
the cost of scrapping or reworking a failed unit is orders of magnitude higher than
the cost of intervening earlier.

The goal of this system is to:

1. **Predict** — score each wafer's failure risk from high-dimensional process
   measurements collected during manufacturing, before the wafer reaches the most
   expensive downstream steps.
2. **Rank** — identify the sensors most associated with yield excursions so
   process engineers know which signals to inspect first.
3. **Decide** — apply a cost-sensitive threshold policy that balances the cost of
   unnecessarily holding a passing wafer against the cost of releasing a failing one.

---

## Why yield prediction matters

Yield is the primary economic lever in semiconductor manufacturing. A single
percentage-point improvement in yield at high-volume production can translate to
tens of millions of dollars in recovered revenue per year. Conversely, a yield
excursion that goes undetected for even a few tool cycles can propagate defects
across an entire lot.

Traditional statistical process control (SPC) monitors individual process parameters
in isolation. A machine learning approach can detect subtle multivariate excursions —
combinations of process signals that individually appear in-spec but together predict
failure — that SPC charts would miss.

Early-warning systems also reduce the number of wafers that complete the full process
flow before failure is detected. Earlier detection means less wasted processing cost,
faster cycle time for corrective action, and reduced risk of releasing defective
material.

---

## Key results

Model selection is based on training-only 5-fold CV PR-AUC. The held-out test
set is used once for final comparison and cost-threshold evaluation.

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Recall (fail) | Precision | Opt. threshold | Cost |
|-------|-----------|-------------|--------------|---------------|-----------|----------------|------|
| Dummy (stratified) | 0.066 | 0.068 | 0.504 | 0.063 | 0.077 | 0.01 | 162 |
| Logistic Regression | 0.180 | 0.168 | 0.726 | 0.313 | 0.143 | 0.52 | 140 |
| Random Forest | **0.227** | **0.222** | **0.793** | **0.812** | **0.197** | 0.14 | **83** |
| XGBoost | 0.206 | 0.213 | 0.786 | 0.625 | 0.156 | 0.02 | 114 |

**Selected model:** random forest, chosen by the training-only CV protocol. It
also leads on the held-out test set (PR-AUC 0.222 vs XGBoost 0.213), so the
family fixed on cross-validation is confirmed by the single test-set look;
XGBoost is the closest challenger.

**Selected-model root-cause candidates:** `sensor_059`, `sensor_033`,
`sensor_103`, `sensor_031`, `sensor_129` from
`reports/root_cause_candidates.csv`.

`sensor_059` is the first sensor to inspect. It leads the selected model's
candidate ranking and remains first in the XGBoost sensitivity check.

**XGBoost sensitivity check:** the challenger XGBoost model also ranks
`sensor_059` first. Its top-five candidates overlap the selected random forest
on three sensors (`sensor_059`, `sensor_033`, `sensor_103`), with
a top-ten overlap of 7/10. The reporting pipeline does not persist a
cross-model sensitivity artifact; the comparison is computed for display in
`notebooks/04_2_xgboost_root_cause_sensitivity.ipynb`.

---

## Dataset

**Source:** [UCI Machine Learning Repository — SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/SECOM)

| Attribute | Value |
|-----------|-------|
| Observations | 1,567 wafer-level records |
| Features | 590 anonymous process/sensor measurements |
| Labels | Binary pass / fail (line-test outcome) |
| Fail rate | ~6.6% (highly imbalanced) |
| Missing values | Present across most features; some features >90% missing |
| Timestamps | Available; one record per production entity |

Each row represents a single production entity (wafer or lot) captured at a point
in the manufacturing flow. Features are anonymous (`sensor_000` … `sensor_589`);
physical process names are not provided in the public release.

The dataset is available for download from the UCI repository. Raw data files are
not committed to this repository. See [`data/raw/README.md`](data/raw/README.md)
for acquisition instructions.

---

## Methodology

### Pipeline stages

```
raw SECOM (data/raw)
      │  load + schema validation
      ▼
split_train_test  ──►  split_data.py  ──►  data/splits/{train,test}.csv
  (stratified                              RAW rows: NaNs intact,
   holdout)                                all sensor columns kept
      │
      ▼
sklearn Pipeline (fit on train.csv only)
  ├─ "preprocess" step (SecomPreprocessor): drop-high-missing →
  │   median impute → drop-low-CV → drop-high-correlation
  ├─ scaling (linear models only)
  └─ estimator
      │  stratified k-fold CV  →  select winner  →  freeze threshold (train OOF)
      ▼
final fit  ──►  evaluate once on test.csv
```

### 1. Data acquisition and validation

The SECOM dataset is downloaded from the UCI repository, features and labels are
joined, and the 590 anonymous columns are renamed consistently to `sensor_000`
through `sensor_589`. A schema validation step asserts expected shape, column
presence, and label cardinality before any further processing.

### 2. Exploratory data analysis

EDA covers class imbalance, per-feature and per-row missingness rates, constant
and near-constant features, sensor distributions, outlier frequency, correlation
structure, and pass/fail distribution comparisons for candidate features. Findings
are documented in notebooks 01 and 02 and inform all downstream preprocessing
decisions.

### 3. Preprocessing pipeline

The preprocessing pipeline is implemented as a scikit-learn `Pipeline` +
`ColumnTransformer` so that all fitting is done exclusively on training data.
Steps include:

- Drop features with missingness above a configurable threshold (default 10%)
- Drop low-variation features (coefficient of variation below threshold)
- Add binary missingness-indicator columns for features where missingness may
  carry signal
- Median imputation for remaining missing values
- Remove highly correlated feature pairs (Pearson |r| > 0.95)
- Standard scaling for linear models; unscaled version passed to tree models

### 4. Train/test split

A single stratified 85/15 train/test split is performed before any model fitting.
Class weights and SMOTE are evaluated during cross-validation on the training fold
only. The held-out test set is touched exactly once for final evaluation.

### 5. Modeling

See [Modeling approach](#modeling-approach) below.

### 6. Cost-sensitive thresholding

See [Cost-sensitive decision policy](#cost-sensitive-decision-policy) below.

### 7. Root-cause sensor ranking

See [Root-cause sensor ranking](#root-cause-sensor-ranking) below.

---

## Modeling approach

Four model families are trained and compared:

| Model | Purpose |
|-------|---------|
| Dummy classifier (stratified) | Baseline — establishes floor for all metrics |
| Logistic regression (L2) | Linear baseline; interpretable coefficients |
| Random forest | Non-linear ensemble; built-in feature importance |
| XGBoost | Gradient boosting; best expected performance on tabular data |

All non-dummy models use stratified 5-fold cross-validation on the training set
with class-imbalance handling (class weights or SMOTE evaluated per fold).
Hyperparameter search is performed via `RandomizedSearchCV` with configurations
defined in `configs/model_config.yaml`.

**Primary evaluation metrics** (in order of priority):

1. PR-AUC — most informative under severe class imbalance
2. Recall on failed wafers — cost of missing a failure is high
3. Balanced accuracy — accounts for imbalance without assuming a cost ratio
4. ROC-AUC — secondary; reported for completeness
5. Cost-sensitive expected loss — see next section

Accuracy is reported but not optimized, as the ~93% majority-class baseline makes
it an unreliable signal.

---

## Root-cause sensor ranking

This step ranks the sensors most associated with failure risk so engineers know
which signals to inspect first. The workflow combines model attribution,
distribution comparisons, and SPC-style excursion checks to show which signals
deserve the first process-engineering look.

Steps:

1. **SHAP global importance** — mean absolute SHAP value across the test set ranks
   features by their average contribution to model predictions.
2. **SHAP local explanation** — individual high-risk wafers are explained to show
   which sensors drove the elevated score for that specific unit.
3. **Permutation importance** — model-agnostic backup; validates SHAP rankings are
   not artefacts of the tree structure.
4. **Pass/fail distribution comparison** — for each top-ranked feature, KDE plots
   and summary statistics compare distributions between passing and failing wafers.
5. **SPC-style excursion check** — each top feature is checked for out-of-3σ values
   at the wafer level and the out-of-control frequency is compared between groups.
6. **Outlier frequency ratio** — fraction of failing wafers with extreme values
   versus passing wafers; highlights features with disproportionate excursion rates
   in the fail population.

**Dataset constraints:** SECOM uses anonymized sensor IDs and a binary pass/fail
line-test label, so the ranking names sensors to inspect rather than failure
modes or process steps. The public benchmark is a historical snapshot, not a
complete fab execution trace. In a fab follow-up, the next engineering action is
to map top IDs such as `sensor_059` to process step, tool, chamber, recipe, lot
history, and maintenance records before changing process settings.

---

## Cost-sensitive decision policy

A standard 0.5 decision threshold is rarely optimal when the costs of false
positives and false negatives are asymmetric — which is always the case in
manufacturing yield contexts.

The system parameterizes a cost matrix in `configs/cost_config.yaml`:

| Outcome | Default cost |
|---------|-------------|
| True pass (correctly released) | 0 |
| True fail (correctly flagged) | 0 |
| False fail (unnecessary hold/review) | *C_fp* — engineering review cost |
| False pass (missed failure, released) | *C_fn* — downstream processing waste + quality escape |

The threshold optimizer sweeps from 0.01 to 0.99, computes the expected cost at
each operating point, and identifies the minimum-cost threshold. The tradeoff
curves (recall, precision, false-pass rate, false-fail rate, expected cost vs.
threshold) are plotted and saved to `reports/figures/`.

The optimal threshold is reported alongside the corresponding confusion matrix
and metric profile. Changing the cost assumptions in `configs/cost_config.yaml`
automatically recomputes the recommendation without retraining.

---

## Dashboard

A multi-page Streamlit application is planned to provide an interactive
interface to the trained system.

| Page | Function |
|------|----------|
| **Batch Scoring** | Upload a CSV of wafer process measurements, score failure risk, flag high-risk units, export results |
| **Root-Cause Ranking** | Global SHAP importance chart, local explanation for a selected wafer, pass/fail distribution comparisons |
| **Cost Thresholding** | Interactive threshold slider with live confusion matrix, cost curve, and operating-point recommendation |
| **Model Monitoring** | Missingness summary, feature drift indicators, prediction distribution, high-risk rate trend |

**Screenshots**

> Screenshots will be added here after the dashboard is implemented.

---

## API

The FastAPI service is implemented and exposes the trained model for
programmatic scoring.

**Model loading:** the API loads `models/selected_model.joblib` plus
`models/model_metadata.json` at startup. `models/model_metadata.json` is
generated by `python scripts/export_model_metadata.py` and persists the
cost-optimal decision threshold, model version, cost matrix, metrics snapshot,
and expected sensors. If the metadata file is absent the API still starts with
a fallback threshold of 0.5.

**Server command:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness and model status; always HTTP 200 |
| `POST` | `/predict` | Score a single wafer |
| `POST` | `/predict/batch` | Score up to 10,000 wafers per request |

### GET /health

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "random_forest-a1b2c3d",
  "threshold": 0.14
}
```

### POST /predict

**Request:**

```json
{
  "wafer_id": "W-00123",
  "features": { "sensor_059": 1.2, "sensor_033": -0.4 }
}
```

**Response:**

```json
{
  "wafer_id": "W-00123",
  "failure_probability": 0.71,
  "risk_flag": true,
  "threshold_used": 0.14,
  "model_version": "random_forest-a1b2c3d"
}
```

**curl example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"wafer_id": "W-00123", "features": {"sensor_059": 1.2, "sensor_033": -0.4}}'
```

### POST /predict/batch

**Request:**

```json
{
  "wafers": [
    { "wafer_id": "W-1", "features": { "sensor_059": 1.2 } }
  ]
}
```

**Response:**

```json
{
  "predictions": [
    {
      "wafer_id": "W-1",
      "failure_probability": 0.12,
      "risk_flag": false,
      "threshold_used": 0.14,
      "model_version": "random_forest-a1b2c3d"
    }
  ]
}
```

### Behavior and validation

- **Partial feature maps allowed.** Unspecified sensors are NaN-filled and
  imputed by the pipeline; only sensors of interest need to be supplied.
- **Risk flag logic.** `risk_flag = failure_probability >= threshold_used`.
- **Valid feature keys.** Keys must be within the `sensor_000..sensor_589`
  namespace; any key outside that range returns 400. The namespace is the
  request contract, not a per-model check: a namespace-valid sensor that is not
  in the loaded model's expected set is accepted and then dropped at scoring
  time (inputs are aligned to the model's `expected_sensors`).

| Status | Condition |
|--------|-----------|
| 422 | Malformed request body |
| 400 | Unknown sensor keys or empty batch |
| 413 | Batch larger than 10,000 wafers |
| 503 | Model not loaded |

Request and response schemas are defined in `api/schemas.py`.

---

## Reproducing the project

### Setup

```bash
git clone https://github.com/virgil-castillo/semiconductor-yield-root-cause.git
cd semiconductor-yield-root-cause
conda env create -f environment.yml
conda activate mlops
```

### Run the pipeline

Run the scripts in order; each step consumes the previous step's artifacts:

```bash
python scripts/download_data.py          # Fetch raw SECOM files from UCI
python scripts/split_data.py             # Write raw stratified train/test splits
python scripts/train_models.py           # Train/tune model families, select winner
python scripts/evaluate_model.py         # Evaluate on the held-out test set
python scripts/feature_importance.py     # Selected-model feature importance
python scripts/generate_explanations.py  # SHAP values + root-cause tables
python scripts/generate_reports.py       # Markdown reports
```

### Tests

```bash
pytest                          # Full test suite
pytest tests/test_data.py -v    # Single module
```

### API

```bash
python scripts/export_model_metadata.py          # Persist threshold + version metadata
uvicorn api.main:app --host 0.0.0.0 --port 8000  # Serve the prediction API
```

See [API](#api) for the full endpoint contract, request/response examples, and
error codes.

---

## Repository structure

```
semiconductor-yield-root-cause/
├── src/yield_risk/   # Core importable library: loaders, preprocessing,
│                     #   models, evaluation, importance, explainability,
│                     #   thresholding, root cause, monitoring, reporting
├── scripts/          # CLI entry points; sole writers of artifacts
│                     #   (download → split → train → evaluate → explain → report)
├── notebooks/        # Read-only analyses (EDA → sensor shortlist →
│                     #   modeling → root cause → XGBoost sensitivity)
├── api/              # FastAPI scoring service (endpoints + Pydantic schemas)
├── configs/          # YAML run, model, and cost configuration
├── reports/          # Generated metrics, markdown reports, figures
├── tests/            # Test suite (pytest)
├── data/             # Dataset acquisition + split instructions (raw not committed)
└── docs/             # Roadmap, manufacturing context, assumptions & limitations
```

Planned work — the Streamlit dashboard, Docker packaging, and CI — is tracked in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Limitations

- **Anonymous features.** The SECOM dataset provides no physical process labels.
  The ranking names sensor IDs; linking those IDs to process step, tool,
  chamber, recipe, lot, or maintenance context is the first fab follow-up.

- **Historical benchmark.** The UCI SECOM release is a public historical
  dataset, not a complete fab execution trace. Monitoring modules compare
  held-out batches; real-time ingestion remains future work.

- **Single fab, single technology node.** Model performance is specific to the
  process conditions captured in this dataset. Transferring to a different fab
  or node would require retraining and threshold recalibration.

- **Label quality.** The binary pass/fail label represents a single line-test
  outcome. It does not distinguish failure modes, defect types, or yield bin
  categories. A multi-class or multi-label formulation would require richer
  labeling.

- **Class imbalance.** With ~6.6% fail rate, small changes in threshold or
  sampling strategy can produce large swings in precision and recall. All
  reported metrics account for this, but deployment decisions should be made
  with awareness of the operating cost structure.

---

## Future work

- **Multi-class failure mode modeling** — extend beyond binary pass/fail to
  predict specific yield bin categories when richer label data is available.
- **Temporal modeling** — incorporate lot genealogy and tool sequence information
  to detect drift within a production run, not just across the static dataset.
- **Online learning** — move from batch retraining to incremental model updates
  as new wafer data arrives.
- **Causal inference** — apply causal discovery methods (PC algorithm, DoWhy) to
  generate stronger hypotheses about process-to-yield relationships.
- **Tool-level integration** — link anonymous sensor features to specific process
  tools and chambers to produce actionable engineering recommendations.
- **Real-time SPC integration** — feed model risk scores into existing SPC
  dashboards so operators see early-warning flags alongside traditional control charts.
