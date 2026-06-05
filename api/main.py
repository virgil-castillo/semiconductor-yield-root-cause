"""FastAPI application entry point for the yield-risk prediction API.

Server command::

    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, Request

from api.schemas import HealthResponse
from yield_risk.scoring import ModelBundle, load_model_bundle

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
