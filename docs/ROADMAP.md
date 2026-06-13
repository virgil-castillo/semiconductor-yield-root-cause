# Semiconductor Yield Excursion — Implementation Roadmap

Each phase is a self-contained deliverable. Phases within the same tier can be
worked in parallel; phases in a later tier depend on all phases above them.

---

## Status legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Complete |
| 🔄 | In progress |
| ⬜ | Not started |

Note: 🔄 is also used for partially implemented phases where the current repo
has working pieces but not the full deliverable originally listed.

---

## Tier 0 — Foundation

Must be complete before any other phase.

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0-A | `pyproject.toml`, `.gitignore`, `CLAUDE.md` updated for `yield_risk` | ✅ |
| 0-B | Directory skeleton (`src/`, `tests/`, `configs/`, `data/`, `models/`, `reports/`, `notebooks/`, `scripts/`, `app/`, `api/`, `docs/`) | 🔄 |
| 0-C | `configs/config.yaml`, `configs/model_config.yaml`, `configs/cost_config.yaml` | ✅ |
| 0-D | `src/yield_risk/__init__.py`, `src/yield_risk/config.py` (config loader) | ✅ |

---

## Tier 1 — Data Pipeline

Depends on: Tier 0

| Phase | Deliverable | Key files | Status |
|-------|-------------|-----------|--------|
| 1-A | SECOM data acquisition | `scripts/download_data.py`, `data/raw/README.md` | ✅ |
| 1-B | Data loading + label joining | `src/yield_risk/data.py` | ✅ |
| 1-C | Schema validation | `src/yield_risk/validation.py` | ✅ |
| 1-D | Preprocessing pipeline | `src/yield_risk/preprocess.py` | ✅ |
| 1-E | Feature engineering | `src/yield_risk/features.py` | ✅ |
| 1-F | Stratified split | `src/yield_risk/preprocess.py` | ✅ |
| 1-G | Dataset orchestration script | `scripts/preprocess_data.py` | ✅ |

**Tests:** `tests/test_data.py`, `tests/test_preprocess.py`, `tests/test_features.py`

---

## Tier 2 — Modeling

Depends on: Tier 1

| Phase | Deliverable | Key files | Status |
|-------|-------------|-----------|--------|
| 2-A | Model training (dummy, LR, RF, XGB/LGBM) | `src/yield_risk/model.py` | 🔄 |
| 2-B | Evaluation metrics (PR-AUC, ROC-AUC, balanced acc, false-pass/fail rates) | `src/yield_risk/evaluate.py` | 🔄 |
| 2-C | Cost-sensitive threshold optimizer | `src/yield_risk/thresholding.py` | 🔄 |
| 2-D | Train + evaluate scripts | `scripts/train_baseline.py`, `scripts/evaluate_model.py` | 🔄 |
| 2-E | Batch scoring script | `scripts/batch_score.py` | ✅ |

**Tests:** `tests/test_model.py`, `tests/test_evaluate.py`, `tests/test_thresholding.py`

---

## Tier 3 — Explainability & Root-Cause Triage

Depends on: Tier 2

| Phase | Deliverable | Key files | Status |
|-------|-------------|-----------|--------|
| 3-A | SHAP global + local explanations | `src/yield_risk/explainability.py` | ✅ |
| 3-B | Root-cause candidate ranking (SPC checks, outlier freq, pass/fail distributions) | `src/yield_risk/root_cause.py` | ✅ |
| 3-C | Explanation generation script | `scripts/generate_explanations.py` | ✅ |

**Tests:** `tests/test_explainability.py`, `tests/test_root_cause.py`

---

## Tier 4 — Monitoring & Reporting

Depends on: Tier 2 (monitoring), Tier 3 (reporting)

| Phase | Deliverable | Key files | Status |
|-------|-------------|-----------|--------|
| 4-A | Lightweight model monitoring (feature drift, missingness drift, prediction dist) | `src/yield_risk/monitoring.py` | ✅ |
| 4-B | Report generation utilities | `src/yield_risk/reporting.py`, `scripts/generate_reports.py` | ✅ |
| 4-C | Written reports | `reports/executive_summary.md`, `reports/model_card.md`, `reports/data_card.md`, `reports/root_cause_report.md` | ✅ |

---

## Tier 5 — API

Depends on: Tier 2

| Phase | Deliverable | Key files | Status |
|-------|-------------|-----------|--------|
| 5-A | Pydantic schemas | `api/schemas.py` | ✅ |
| 5-B | FastAPI app (health, single-wafer predict, batch predict) | `api/main.py` | ✅ |

Shared scoring extracted to `src/yield_risk/scoring.py` (reused by the CLI and
the API); model metadata (threshold + version) persisted via
`scripts/export_model_metadata.py`.

**Tests:** `tests/test_api.py`, `tests/test_scoring.py`, `tests/test_export_model_metadata.py`

---

## Tier 6 — Dashboard

Depends on: Tier 2, Tier 3, Tier 4

| Phase | Deliverable | Key files | Status |
|-------|-------------|-----------|--------|
| 6-A | App shell + shared state | `app/streamlit_app.py` | ⬜ |
| 6-B | Batch Scoring page | `app/pages/1_Batch_Scoring.py` | ⬜ |
| 6-C | Root-Cause Triage page | `app/pages/2_Root_Cause_Triage.py` | ⬜ |
| 6-D | Cost Thresholding page | `app/pages/3_Cost_Thresholding.py` | ⬜ |
| 6-E | Model Monitoring page | `app/pages/4_Model_Monitoring.py` | ⬜ |

---

## Tier 7 — Notebooks

Depends on: Tier 1 (notebooks 01–03), Tier 2 (04), Tier 3 (05–06), Tier 4 (07)

| Phase | Deliverable | File | Status |
|-------|-------------|------|--------|
| 7-A | EDA — data characterisation and preprocessing config justification (`missing_threshold`, `variance_threshold`, `correlation_threshold`) | `notebooks/01_eda.ipynb` | ✅ |
| 7-B | EDA — sensor-level failure signals for Tier 3 explainability (point-biserial correlation, pass/fail distributions, SHAP hypotheses) | `notebooks/02_eda_yield_patterns.ipynb` | ✅ |
| 7-C | Baseline results walkthrough | `notebooks/03_baseline_results.ipynb` | ✅ |
| 7-D | Model training + evaluation | `notebooks/04_model_training_evaluation.ipynb` | ✅ |
| 7-E | Root-cause analysis | `notebooks/05_root_cause_analysis.ipynb` | ✅ |
| 7-F | Cost-sensitive thresholding | `notebooks/06_cost_sensitive_thresholding.ipynb` | ⬜ |
| 7-G | Model monitoring + drift checks | `notebooks/07_model_monitoring_drift_checks.ipynb` | ✅ |

---

## Tier 8 — Infrastructure

Can start after Tier 0; fully useful once Tier 5 and Tier 6 are complete.

| Phase | Deliverable | Key files | Status |
|-------|-------------|-----------|--------|
| 8-A | Makefile | `Makefile` | ⬜ |
| 8-B | Dockerfile + `.dockerignore` | `Dockerfile`, `.dockerignore` | ⬜ |
| 8-C | GitHub Actions CI (lint → test → smoke) | `.github/workflows/ci.yml` | ⬜ |

---

## Tier 9 — Documentation

Depends on: all tiers

| Phase | Deliverable | Key files | Status |
|-------|-------------|-----------|--------|
| 9-A | `README.md` (full, with results placeholders filled in) | `README.md` | 🔄 |
| 9-B | Supporting docs | `docs/manufacturing_context.md`, `docs/assumptions_and_limitations.md` | ⬜ |
| 9-C | Update `CLAUDE.md` + `AGENTS.md` to reference `yield_risk` | `CLAUDE.md`, `AGENTS.md` | ✅ |

---

## Dependency graph

```
Tier 0 (Foundation)
  └── Tier 1 (Data Pipeline)
        └── Tier 2 (Modeling)
              ├── Tier 3 (Explainability)
              ├── Tier 4 (Monitoring/Reporting)  ← also needs Tier 3
              ├── Tier 5 (API)
              └── Tier 6 (Dashboard)            ← also needs Tiers 3, 4
Tier 0 ──────── Tier 7 (Notebooks, parallel with 1–4)
Tier 0 ──────── Tier 8 (Infrastructure, parallel, fully useful after 5+6)
All tiers ───── Tier 9 (Documentation)
```

---

## Suggested execution order

1. Finish Tier 0 (one session)
2. Tier 1 end-to-end — data pipeline runs and tests pass
3. Tier 2 end-to-end — model trains, evaluates, threshold is tuned
4. Tier 3 + Tier 5 in parallel (independent once Tier 2 is done)
5. Tier 4 + Tier 6 (need Tier 3 outputs)
6. Tier 7 notebooks (fill in from working code, not a research exercise)
7. Tier 8 infrastructure
8. Tier 9 documentation (README last — fill in real numbers)

---

## Backlog — Future Experiments

Not on the critical path; revisit once Tier 2 is stable.

| Idea | Rationale | Status |
|------|-----------|--------|
| Native missing-value handling for tree models (RF, XGBoost) vs. the shared median-imputed `SecomPreprocessor` | Both `RandomForestClassifier` (sklearn ≥1.4) and `XGBClassifier` learn a per-split default direction for NaNs, so missingness can act as a learned signal (e.g., a skipped sensor reading may correlate with yield) instead of being masked by median fill. Requires a separate no-impute preprocessing branch for RF/XGB only, since `logistic_regression`/`dummy` need complete data and the variance/correlation filters currently run on imputed data — keep the existing imputed pipeline as the baseline for comparison. | ⬜ |
