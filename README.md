# Semiconductor Yield Excursion Early-Warning and Root-Cause Triage System

A production-style smart manufacturing ML system for semiconductor yield prediction,
process excursion detection, root-cause candidate triage, and cost-sensitive wafer
risk scoring — built on the UCI SECOM semiconductor manufacturing dataset.

---

## Table of contents

1. [Business problem](#business-problem)
2. [Why yield prediction matters](#why-yield-prediction-matters)
3. [Dataset](#dataset)
4. [Repository structure](#repository-structure)
5. [Methodology](#methodology)
6. [Modeling approach](#modeling-approach)
7. [Root-cause triage](#root-cause-triage)
8. [Cost-sensitive decision policy](#cost-sensitive-decision-policy)
9. [Dashboard](#dashboard)
10. [API](#api)
11. [Reproducing the project](#reproducing-the-project)
12. [Key results](#key-results)
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
2. **Triage** — rank the process variables most associated with yield excursions so
   process engineers have a prioritized list of candidates to investigate.
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

## Repository structure

The tree below is the target production-style structure. In the current repo,
the implemented pieces are the data pipeline, multi-family model selection,
cost-sensitive evaluation, feature importance, SHAP/root-cause ranking, batch
scoring, and notebooks through root-cause sensitivity analysis. API, dashboard,
monitoring/reporting utilities, Makefile, Docker, and CI remain planned work.

```
semiconductor-yield-root-cause/
│
├── README.md                        # This file
├── ROADMAP.md → docs/ROADMAP.md
├── pyproject.toml                   # Package metadata, deps, tool config
├── requirements.txt                 # Pinned runtime deps for reproducibility
├── Makefile                         # Top-level task runner
├── Dockerfile                       # Container for Streamlit app or API
├── .dockerignore
├── .gitignore
├── LICENSE
│
├── .github/
│   └── workflows/
│       └── ci.yml                   # Lint → test → smoke on every push
│
├── configs/
│   ├── config.yaml                  # Paths, seeds, run-level settings
│   ├── model_config.yaml            # Model hyperparameters and search grids
│   └── cost_config.yaml             # False-pass / false-fail cost assumptions
│
├── data/
│   ├── raw/README.md                # How to obtain the SECOM dataset
│   ├── interim/README.md            # Intermediate processed artifacts
│   └── processed/README.md         # Final train/val/test splits
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_eda_yield_patterns.ipynb
│   ├── 03_baseline_results.ipynb
│   ├── 04_model_training_evaluation.ipynb
│   ├── 05_root_cause_analysis.ipynb
│   ├── 05_2_xgboost_root_cause_sensitivity.ipynb
│   ├── 06_cost_sensitive_thresholding.ipynb  # Planned
│   └── 07_model_monitoring_drift_checks.ipynb # Planned
│
├── src/yield_risk/                  # Core library — importable package
│   ├── __init__.py
│   ├── config.py                    # Config loader (YAML → dataclass)
│   ├── data.py                      # SECOM loader, label join, renaming
│   ├── validation.py                # Schema and data-quality checks
│   ├── preprocess.py                # Imputation, filtering, split pipeline
│   ├── features.py                  # Aggregate features and interactions
│   ├── model.py                     # Model registry, CV search, selection
│   ├── evaluate.py                  # Metrics, confusion matrix, PR/ROC curves
│   ├── importance.py                # Family-agnostic feature importance
│   ├── thresholding.py              # Cost-sensitive threshold search
│   ├── explainability.py            # SHAP global + local explanations
│   ├── root_cause.py                # Candidate ranking, SPC checks, distributions
│   ├── monitoring.py                # Planned
│   └── reporting.py                 # Planned
│
├── scripts/
│   ├── download_data.py             # Fetch SECOM files from UCI
│   ├── preprocess_data.py           # Raw → processed train/test pipeline
│   ├── train_models.py              # Train/tune model families, select winner
│   ├── evaluate_model.py            # Load artifact, produce eval report
│   ├── generate_explanations.py     # SHAP values + root-cause tables
│   ├── feature_importance.py        # Feature importance export
│   └── batch_score.py               # Score a new batch of wafer records
│
├── app/
│   ├── streamlit_app.py             # Planned
│   └── pages/
│       ├── 1_Batch_Scoring.py       # Upload → score → export
│       ├── 2_Root_Cause_Triage.py   # SHAP plots, feature distributions
│       ├── 3_Cost_Thresholding.py   # Threshold slider, cost curve
│       └── 4_Model_Monitoring.py    # Drift summary, prediction trend
│
├── api/
│   ├── main.py                      # Planned
│   └── schemas.py                   # Planned
│
├── models/
│   └── README.md                    # Artifact naming conventions
│
├── reports/
│   ├── figures/                     # Generated plots (gitignored)
│   ├── executive_summary.md
│   ├── model_card.md
│   ├── data_card.md
│   └── root_cause_report.md
│
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_modeling.py
│   ├── test_thresholding.py
│   └── test_api.py
│
└── docs/
    ├── ROADMAP.md
    ├── manufacturing_context.md
    └── assumptions_and_limitations.md
```

---

## Methodology

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

- Drop features with missingness above a configurable threshold (default 60%)
- Drop near-constant features (variance below threshold)
- Add binary missingness-indicator columns for features where missingness may
  carry signal
- Median imputation for remaining missing values
- Remove highly correlated feature pairs (Pearson |r| > 0.95)
- Standard scaling for linear models; unscaled version passed to tree models

### 4. Train/test split

A single stratified 80/20 train/test split is performed before any model fitting.
Class weights and SMOTE are evaluated during cross-validation on the training fold
only. The held-out test set is touched exactly once for final evaluation.

### 5. Modeling

See [Modeling approach](#modeling-approach) below.

### 6. Cost-sensitive thresholding

See [Cost-sensitive decision policy](#cost-sensitive-decision-policy) below.

### 7. Root-cause triage

See [Root-cause triage](#root-cause-triage) below.

---

## Modeling approach

Four model families are trained and compared:

| Model | Purpose |
|-------|---------|
| Dummy classifier (stratified) | Baseline — establishes floor for all metrics |
| Logistic regression (L2) | Linear baseline; interpretable coefficients |
| Random forest | Non-linear ensemble; built-in feature importance |
| XGBoost / LightGBM | Gradient boosting; best expected performance on tabular data |

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

## Root-cause triage

Root-cause triage is framed as **candidate investigation prioritization**, not
causal attribution. The workflow identifies process variables statistically
associated with failure risk and ranks them for engineering review.

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

> **Important caveat:** This workflow identifies candidate variables associated with
> failure risk. It does not prove physical causality. In a production environment,
> these findings would need to be validated with tool metadata, chamber history,
> recipe information, lot genealogy, maintenance logs, metrology data, and
> process-engineer review. The anonymous feature names in SECOM prevent any direct
> physical interpretation without additional fab context.

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
| **Root-Cause Triage** | Global SHAP importance chart, local explanation for a selected wafer, pass/fail distribution comparisons |
| **Cost Thresholding** | Interactive threshold slider with live confusion matrix, cost curve, and operating-point recommendation |
| **Model Monitoring** | Missingness summary, feature drift indicators, prediction distribution, high-risk rate trend |

**Screenshots**

> Screenshots will be added here after the dashboard is implemented.

---

## API

A FastAPI service is planned to expose the trained model for programmatic
scoring.

**Planned server command:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

```
GET  /health                  Returns service status and model version
POST /predict                 Score a single wafer record
POST /predict/batch           Score a list of wafer records
```

**Single-wafer request example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"sensor_000": 3142.0, "sensor_001": -0.43, ...}}'
```

**Response:**

```json
{
  "wafer_id": "W-00123",
  "failure_probability": 0.71,
  "risk_flag": true,
  "threshold_used": 0.38,
  "model_version": "xgb-v1.0"
}
```

Full schema definitions will live in `api/schemas.py` when the API is
implemented.

---

## Reproducing the project

### Prerequisites

- Conda with an `mlops` environment (Python 3.12)
- See `pyproject.toml` for full dependency list

### Setup

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
```

### Full pipeline

```bash
python scripts/download_data.py
python scripts/preprocess_data.py
python scripts/train_models.py
python scripts/evaluate_model.py
python scripts/feature_importance.py
python scripts/generate_explanations.py
```

### Individual scripts

```bash
python scripts/download_data.py       # Fetch raw data
python scripts/preprocess_data.py     # Build processed train/test splits
python scripts/train_models.py        # Train/tune model families, select winner
python scripts/evaluate_model.py      # Evaluate on test set, save metrics + figures
python scripts/feature_importance.py  # Extract selected-model feature importance
```

### Tests

```bash
pytest           # Full test suite
pytest tests/test_data.py -v    # Single module
```

### Dashboard

Not yet implemented.

### API

Not yet implemented.

### Docker

Not yet implemented.

---

## Key results

Model selection is based on training-only 5-fold CV PR-AUC. The held-out test
set is used once for final comparison and cost-threshold evaluation.

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Recall (fail) | Precision | Opt. threshold | Cost |
|-------|-----------|-------------|--------------|---------------|-----------|----------------|------|
| Dummy (stratified) | 0.066 | 0.066 | 0.490 | 0.048 | 0.048 | 0.01 | 220 |
| Logistic Regression | 0.182 | 0.210 | 0.669 | 0.571 | 0.130 | 0.47 | 170 |
| Random Forest | **0.217** | 0.193 | 0.758 | 0.571 | 0.171 | 0.08 | 148 |
| XGBoost | 0.208 | **0.261** | **0.802** | **0.667** | **0.222** | 0.03 | **119** |

**Selected model:** random forest, chosen by the training-only CV protocol.
XGBoost performs better on the held-out test set, but that result is reported
as generalization evidence rather than used to reopen model selection.

**Selected-model root-cause candidates:** `sensor_059`, `sensor_033`,
`sensor_519`, `sensor_205`, `sensor_031` from
`reports/root_cause_candidates.csv`.

**XGBoost sensitivity check:** the near-tie XGBoost model also ranks
`sensor_059` first. Its top-five candidates overlap the selected random forest
by 4/5 sensors, and the top-ten overlap is 7/10
(`reports/root_cause_model_sensitivity_summary.csv`).

---

## Limitations

- **Anonymous features.** The SECOM dataset provides no physical process labels.
  Root-cause candidates are identified by statistical association only; physical
  interpretation requires fab-internal metadata.

- **Static dataset.** The UCI SECOM release is a historical snapshot. The system
  does not include real-time data ingestion; monitoring modules simulate drift
  detection against held-out batches.

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
