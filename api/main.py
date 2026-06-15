"""FastAPI application entry point for the yield-risk prediction API.

Server command::

    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictionResponse,
    BatchRequest,
    HealthResponse,
    PredictionResponse,
    WaferRequest,
)
from yield_risk.scoring import ModelBundle, load_model_bundle, score_frame

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/selected_model.joblib")
MAX_BATCH_SIZE = 10_000


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model bundle once at startup; store it in app.state.

    A failure during loading is captured so the app can still start and
    report ``model_loaded: false`` via GET /health, rather than crashing.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the running application.
    """
    try:
        app.state.bundle = load_model_bundle(MODEL_PATH)
    except Exception:
        logger.exception("Failed to load model bundle")
        app.state.bundle = None
    yield


app = FastAPI(
    title="Semiconductor Yield Risk Prediction API",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Return the spec's ErrorResponse JSON for any unhandled error.

    Without this, Starlette's default server-error handler responds with a
    ``text/plain`` body, which does not match the ``ErrorResponse``
    (``{"detail": str}``) shape the API contract specifies for HTTP 500.

    Args:
        request: The incoming FastAPI request.
        exc: The unhandled exception.

    Returns:
        A 500 JSONResponse with an ``ErrorResponse``-shaped body.
    """
    logger.exception("Unhandled error processing %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500, content={"detail": "Internal Server Error"}
    )


def _get_bundle(request: Request) -> ModelBundle | None:
    """Retrieve the model bundle from app state without raising AttributeError.

    Routes use this helper to safely access the bundle; when the bundle is
    ``None``, predict endpoints return 503 (handled in those routes).

    Args:
        request: The incoming FastAPI request.

    Returns:
        The loaded ModelBundle, or None if the model failed to load at startup.
    """
    bundle: ModelBundle | None = getattr(request.app.state, "bundle", None)
    return bundle


_SENSOR_KEY_RE = re.compile(r"^sensor_(\d{3})$")
_SENSOR_MAX = 589


def _invalid_feature_keys(features: dict[str, float]) -> list[str]:
    """Return feature keys that fall outside the sensor_000..sensor_589 namespace.

    A key is valid iff it matches ``sensor_NNN`` where ``NNN`` is exactly three
    digits and the integer value is in ``[0, 589]``.

    Note:
        This is a SECOM raw-input namespace check, not a per-model check. A key
        can be namespace-valid yet absent from the loaded model's
        ``expected_sensors`` (the model may have been trained on a subset). Such
        keys are accepted here and then dropped at scoring time by
        :func:`yield_risk.scoring.score_frame`, which aligns inputs to
        ``bundle.expected_sensors``. The namespace is the request contract; the
        model's ``expected_sensors`` is the scoring contract.

    Args:
        features: Mapping of feature key to numeric value from the request.

    Returns:
        Sorted list of offending keys; empty when all keys are valid.
    """
    bad: list[str] = []
    for key in features:
        match = _SENSOR_KEY_RE.match(key)
        if match is None or int(match.group(1)) > _SENSOR_MAX:
            bad.append(key)
    return sorted(bad)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness and model-status check. Always returns HTTP 200.

    Args:
        request: The incoming FastAPI request.

    Returns:
        HealthResponse with status 'ok' and model load information.
    """
    bundle = _get_bundle(request)
    return HealthResponse(
        status="ok",
        model_loaded=bundle is not None,
        model_version=bundle.model_version if bundle is not None else None,
        threshold=bundle.threshold if bundle is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: WaferRequest, request: Request) -> PredictionResponse:
    """Score a single wafer and return its failure probability and risk flag.

    Validates the incoming feature keys against the sensor_000..sensor_589
    namespace, builds a one-row DataFrame aligned to the bundle's expected
    sensors (missing sensors are NaN-filled and imputed by the pipeline), and
    returns a PredictionResponse.

    Args:
        payload: Validated WaferRequest containing an optional wafer_id and a
            feature map of sensor readings.
        request: The incoming FastAPI request, used to access app.state.bundle.

    Returns:
        PredictionResponse with wafer_id, failure_probability, risk_flag,
        threshold_used, and model_version.

    Raises:
        HTTPException: 503 if the model bundle is not loaded.
        HTTPException: 400 if any feature key falls outside the valid namespace.

    Note:
        Any unexpected error during scoring is caught by the app's
        unhandled-exception handler and returned as an HTTP 500 response with
        an ErrorResponse body ``{"detail": "Internal Server Error"}``.
    """
    bundle = _get_bundle(request)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    bad = _invalid_feature_keys(payload.features)
    if bad:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown feature keys (outside sensor_000..sensor_589): {bad}"
            ),
        )

    df = pd.DataFrame([payload.features])
    result = score_frame(bundle, df)

    failure_probability = float(result["score"].iloc[0])
    risk_flag = bool(result["predicted_label"].iloc[0])

    return PredictionResponse(
        wafer_id=payload.wafer_id,
        failure_probability=failure_probability,
        risk_flag=risk_flag,
        threshold_used=bundle.threshold,
        model_version=bundle.model_version,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchRequest, request: Request) -> BatchPredictionResponse:
    """Score a batch of wafers in a single vectorized pass.

    Validates all wafer feature keys against the sensor_000..sensor_589 namespace,
    builds one DataFrame with one row per wafer (missing sensors NaN-filled and
    imputed by the pipeline), and returns one PredictionResponse per wafer in
    input order.

    Args:
        payload: Validated BatchRequest containing a list of WaferRequest objects.
        request: The incoming FastAPI request, used to access app.state.bundle.

    Returns:
        BatchPredictionResponse with one PredictionResponse per input wafer, in
        input order.

    Raises:
        HTTPException: 503 if the model bundle is not loaded.
        HTTPException: 400 if the wafers list is empty.
        HTTPException: 413 if the batch exceeds MAX_BATCH_SIZE (10,000) wafers.
        HTTPException: 400 if any feature key across any wafer falls outside the
            valid sensor_000..sensor_589 namespace.

    Note:
        Any unexpected error during scoring is caught by the app's
        unhandled-exception handler and returned as an HTTP 500 response with
        an ErrorResponse body ``{"detail": "Internal Server Error"}``.
    """
    bundle = _get_bundle(request)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if len(payload.wafers) == 0:
        raise HTTPException(
            status_code=400,
            detail="Batch must contain at least one wafer.",
        )

    if len(payload.wafers) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Batch size {len(payload.wafers)} exceeds maximum of {MAX_BATCH_SIZE}."
            ),
        )

    all_bad: set[str] = set()
    for wafer in payload.wafers:
        all_bad.update(_invalid_feature_keys(wafer.features))
    if all_bad:
        bad_sorted = sorted(all_bad)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown feature keys (outside sensor_000..sensor_589): {bad_sorted}"
            ),
        )

    df = pd.DataFrame([w.features for w in payload.wafers])
    result = score_frame(bundle, df)

    predictions: list[PredictionResponse] = [
        PredictionResponse(
            wafer_id=payload.wafers[i].wafer_id,
            failure_probability=float(result["score"].iloc[i]),
            risk_flag=bool(result["predicted_label"].iloc[i]),
            threshold_used=bundle.threshold,
            model_version=bundle.model_version,
        )
        for i in range(len(payload.wafers))
    ]

    return BatchPredictionResponse(predictions=predictions)
