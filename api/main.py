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

from api.schemas import HealthResponse, PredictionResponse, WaferRequest
from yield_risk.scoring import ModelBundle, load_model_bundle, score_frame

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/selected_model.joblib")


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
        Any unexpected error during scoring propagates and is handled by
        FastAPI's default exception handler as an HTTP 500 response with body
        ``{"detail": "Internal Server Error"}``.
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
