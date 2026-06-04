# Tier 5 — Prediction API Design

**Status:** Approved (design)
**Date:** 2026-06-04
**Roadmap reference:** [docs/ROADMAP.md](../../ROADMAP.md) Tier 5 (phases 5-A, 5-B)

## 1. Purpose

Expose the trained yield-risk model as a FastAPI service so wafer failure risk
can be scored programmatically, outside notebooks and batch CLI runs. The
service wraps the already-tested scoring logic (load pipeline → `predict_proba`
→ apply decision threshold) behind a small HTTP contract that matches the
endpoints already documented in [README.md](../../../README.md#api).

This tier depends only on Tier 2 (modeling), which is complete. It does **not**
introduce new modeling, retraining, or explainability behavior.

## 2. Scope

### In scope
- A shared, framework-agnostic scoring module reused by both the CLI and the API.
- A standalone script that persists model metadata (decision threshold + model
  version) as an artifact the API can load.
- A FastAPI application exposing `GET /health`, `POST /predict`, and
  `POST /predict/batch`.
- Pydantic v2 request/response schemas.
- An automated test suite that runs without committed model artifacts.

### Out of scope (YAGNI)
- Authentication and authorization.
- Rate limiting / quotas.
- Asynchronous or queued batch jobs.
- Model hot-reload while the server is running.
- SHAP / root-cause / explainability endpoints (these belong with the Tier 6
  dashboard).

## 3. Key decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Threshold & model version source | Persist + load `models/model_metadata.json` | The decision threshold is a frozen property of *(trained model + cost assumptions)*. Recomputing at serving time would couple the container to dataset availability and slow cold start. A small metadata artifact keeps the API stateless w.r.t. data. |
| 2 | Scoring logic location | Extract `src/yield_risk/scoring.py`, reused by `batch_score.py` and the API | Today the load + `predict_proba` + threshold logic lives only in `scripts/batch_score.py`. A shared module gives one source of truth and lets scoring be tested without HTTP. |
| 3 | Single-wafer input contract | Lenient, reject unknown keys | SECOM is heavily missing-value and the pipeline already imputes, so partial feature maps must be allowed. Rejecting keys outside the `sensor_000..sensor_589` namespace is cheap typo/schema-drift protection. |
| 4 | Endpoint scope | README contract only (`/health`, `/predict`, `/predict/batch`) | Matches the documented contract; defers UI-facing introspection endpoints to the dashboard tier. |
| 5 | Test model source | Tiny in-memory dummy pipeline fixture | Tests must run on a clean checkout / fresh CI where `models/` artifacts may be absent. Real-model coverage is the eval pipeline's responsibility, not the API's. |
| 6 | Metadata writer location | Standalone `scripts/export_model_metadata.py` | Keeps `evaluate_model.py` focused; provides one clear command to regenerate metadata when cost assumptions change in `configs/cost_config.yaml`. |

## 4. Architecture

```
                  configs/cost_config.yaml
                           │
        scripts/export_model_metadata.py
                           │
                           ▼
              models/model_metadata.json
                           │
models/selected_model.joblib              │
        │                                 │
        └────────────┬────────────────────┘
                     ▼
        src/yield_risk/scoring.py
        (load_model_bundle, score_frame)
            │                     │
            ▼                     ▼
   scripts/batch_score.py     api/main.py  ◀── api/schemas.py
        (CLI)                  (FastAPI)
```

### Components

#### `src/yield_risk/scoring.py` (new)
Framework-agnostic scoring service. No FastAPI or argparse imports.

- `ModelBundle` (dataclass): `pipeline`, `model_version: str`,
  `threshold: float`, `expected_sensors: list[str]`.
- `load_model_bundle(model_path: Path, metadata_path: Path | None = None) -> ModelBundle`
  - Loads the joblib pipeline.
  - Loads `model_metadata.json` if present; otherwise falls back to
    `threshold = 0.5` and `model_version` derived from a short hash of the
    joblib file, and logs a warning.
  - Derives `expected_sensors` (the `sensor_NNN` namespace the model was trained
    on). Source order: metadata `expected_sensors` if recorded, else the fitted
    pipeline's input feature names.
- `score_frame(bundle: ModelBundle, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame`
  - Selects/aligns sensor columns to `expected_sensors`; missing columns added
    as `NaN`.
  - Returns the input plus `score` (positive-class probability) and
    `predicted_label` (`score >= threshold`), using `bundle.threshold` when
    `threshold` is `None`.

#### `scripts/batch_score.py` (refactor)
Replace the inline load + `predict_proba` + threshold logic with calls to
`scoring.load_model_bundle` and `scoring.score_frame`. CLI surface (`--model`,
`--input`, `--output`, `--threshold`) and printed summary are unchanged.

#### `scripts/export_model_metadata.py` (new)
Generates `models/model_metadata.json`:
1. Load the selected pipeline and a held-out scored set (test predictions).
2. Read `configs/cost_config.yaml` and run
   `thresholding.find_optimal_threshold` to get the cost-optimal threshold.
3. Write JSON with:
   - `model_version`: `"<family>-<short joblib hash>"` (e.g. `random_forest-a1b2c3d`).
   - `optimal_threshold`: float.
   - `cost_matrix`: the matrix used.
   - `created_at`: ISO-8601 UTC timestamp.
   - `metrics`: snapshot from `reports/selected_model_metrics.json`.
   - `expected_sensors`: list of sensor columns the model expects.

#### `api/schemas.py` (new)
Pydantic v2 models:
- `WaferRequest`: `wafer_id: str | None = None`, `features: dict[str, float]`.
- `BatchRequest`: `wafers: list[WaferRequest]`.
- `PredictionResponse`: `wafer_id: str | None`, `failure_probability: float`,
  `risk_flag: bool`, `threshold_used: float`, `model_version: str`.
- `BatchPredictionResponse`: `predictions: list[PredictionResponse]`.
- `HealthResponse`: `status: str`, `model_loaded: bool`, `model_version: str | None`,
  `threshold: float | None`.
- `ErrorResponse`: `detail: str`.

#### `api/main.py` (new)
- FastAPI app with a `lifespan` handler that calls `load_model_bundle` once at
  startup and stores the bundle in `app.state`. A load failure is captured (not
  raised) so `/health` can report `model_loaded: false` and predict endpoints
  return 503.
- Routes per the contract below.
- Server command (already documented in the README):
  `uvicorn api.main:app --host 0.0.0.0 --port 8000`.

## 5. API contract

### `GET /health`
Liveness + model status. Always 200.
```json
{ "status": "ok", "model_loaded": true, "model_version": "random_forest-a1b2c3d", "threshold": 0.08 }
```

### `POST /predict`
Request:
```json
{ "wafer_id": "W-00123", "features": { "sensor_059": 1.2, "sensor_033": -0.4 } }
```
Response (200):
```json
{ "wafer_id": "W-00123", "failure_probability": 0.71, "risk_flag": true,
  "threshold_used": 0.08, "model_version": "random_forest-a1b2c3d" }
```
- Partial feature maps allowed; unspecified sensors imputed by the pipeline.
- `risk_flag = failure_probability >= threshold_used`.

### `POST /predict/batch`
Request:
```json
{ "wafers": [ { "wafer_id": "W-1", "features": { "sensor_059": 1.2 } }, ... ] }
```
Response (200):
```json
{ "predictions": [ { "wafer_id": "W-1", "failure_probability": 0.12, "risk_flag": false,
  "threshold_used": 0.08, "model_version": "random_forest-a1b2c3d" }, ... ] }
```
- Maximum 10,000 wafers per request.

### Error handling

| Condition | Status | Body |
|-----------|--------|------|
| Malformed body / wrong types | 422 | FastAPI/Pydantic default |
| Feature map contains keys outside `sensor_000..sensor_589` | 400 | `ErrorResponse` listing offending keys |
| Empty batch (`wafers: []`) | 400 | `ErrorResponse` |
| Batch larger than 10,000 | 413 | `ErrorResponse` |
| Model bundle failed to load at startup | 503 | `ErrorResponse` |
| Unhandled error | 500 | `ErrorResponse` |

## 6. Data flow (single predict)

1. Request validated by Pydantic → `WaferRequest`.
2. Feature keys checked against `bundle.expected_sensors`; unknown keys → 400.
3. Build a one-row DataFrame aligned to `expected_sensors`, filling provided
   values, leaving the rest `NaN`.
4. `score_frame(bundle, df)` → probability + label using `bundle.threshold`.
5. Assemble `PredictionResponse` (echo `wafer_id`, include `threshold_used`,
   `model_version`).

Batch follows the same path over N rows in a single DataFrame for vectorized
scoring.

## 7. Testing

`tests/test_api.py`, FastAPI `TestClient`. A fixture builds a tiny in-memory
sklearn `Pipeline` (e.g. a trivial classifier over a handful of `sensor_NNN`
columns) plus a synthetic `ModelBundle`, injected into `app.state` so no
committed artifact is required.

Cases:
- `GET /health` → 200, `model_loaded: true`, version + threshold present.
- `POST /predict` happy path → 200, well-formed response, `risk_flag` consistent
  with `threshold_used`.
- `POST /predict` with a partial feature map → 200 (missing sensors imputed).
- `POST /predict` with an unknown sensor key → 400.
- `POST /predict/batch` happy path → 200, one prediction per input wafer.
- `POST /predict/batch` empty list → 400.
- `POST /predict/batch` over the size limit → 413.
- Model-not-loaded state → predict endpoints return 503; `/health` reports
  `model_loaded: false`.

Separate unit coverage for `src/yield_risk/scoring.py` (`load_model_bundle`
fallback behavior; `score_frame` column alignment and threshold application) may
live in `tests/test_scoring.py`.

## 8. Acceptance criteria

- `uvicorn api.main:app` starts and `GET /health` returns 200.
- All three endpoints behave per the contract, including the error matrix.
- `scripts/export_model_metadata.py` produces a valid `models/model_metadata.json`.
- `scripts/batch_score.py` produces identical output to its current behavior
  after the refactor.
- `pytest tests/test_api.py` passes on a clean checkout with no `models/`
  artifacts present.
- `ruff check` and `mypy` pass on all new/changed files.
- README's "API … Not yet implemented" note is replaced with working usage.
