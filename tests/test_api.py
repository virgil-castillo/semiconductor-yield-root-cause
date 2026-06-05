"""Tests for the FastAPI prediction API — health endpoint and reusable fixtures.

Later tasks (predict, batch) extend this file by importing the fixtures.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from api.main import app
from yield_risk.scoring import ModelBundle

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SENSOR_COLS = [f"sensor_{i:03d}" for i in range(5)]


def _make_bundle() -> ModelBundle:
    """Build a tiny in-memory ModelBundle for testing without touching disk.

    Uses a real LogisticRegression (fixed random_state=42) fitted on a small
    non-collinear synthetic dataset so that ``predict_proba`` returns a varied
    range of probabilities across different inputs (not a constant 0.5).
    threshold=0.3.  ``feature_names_in_`` is set because the model is fitted
    on a pandas DataFrame.

    Returns:
        A real ModelBundle with a fitted sklearn Pipeline, a fixed version
        string ``"test_model-abc1234"``, threshold ``0.3``, and
        ``expected_sensors`` list.
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((20, len(_SENSOR_COLS))),
        columns=_SENSOR_COLS,
    )
    y = [0] * 10 + [1] * 10

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=200)),
        ]
    )
    pipeline.fit(X, y)

    return ModelBundle(
        pipeline=pipeline,
        model_version="test_model-abc1234",
        threshold=0.3,
        expected_sensors=_SENSOR_COLS,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """TestClient with a loaded model bundle (lifespan does NOT run).

    Sets ``app.state.bundle`` before constructing the TestClient so the state
    is in place from the first request.  Tears down by resetting the attribute
    to ``None`` after the test completes.

    Yields:
        TestClient with app.state.bundle set to a real in-memory bundle.
    """
    app.state.bundle = _make_bundle()
    tc = TestClient(app, raise_server_exceptions=True)
    yield tc
    app.state.bundle = None


@pytest.fixture()
def client_no_model() -> Iterator[TestClient]:
    """TestClient with no model bundle loaded (lifespan does NOT run).

    Sets ``app.state.bundle`` to ``None`` before constructing the TestClient
    so the state is in place from the first request.  Tears down by resetting
    the attribute to ``None`` after the test completes.

    Yields:
        TestClient with app.state.bundle explicitly set to None.
    """
    app.state.bundle = None
    tc = TestClient(app, raise_server_exceptions=True)
    yield tc
    app.state.bundle = None


# ---------------------------------------------------------------------------
# /health tests
# ---------------------------------------------------------------------------


def test_health_with_loaded_bundle_returns_200_and_ok_status(
    client: TestClient,
) -> None:
    """GET /health with a loaded bundle returns 200 and status 'ok'."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_with_loaded_bundle_reports_model_loaded_true(
    client: TestClient,
) -> None:
    """GET /health with a loaded bundle reports model_loaded as True."""
    response = client.get("/health")
    assert response.json()["model_loaded"] is True


def test_health_with_loaded_bundle_reports_correct_version(
    client: TestClient,
) -> None:
    """GET /health with a loaded bundle returns the bundle's model_version."""
    response = client.get("/health")
    assert response.json()["model_version"] == "test_model-abc1234"


def test_health_with_loaded_bundle_reports_correct_threshold(
    client: TestClient,
) -> None:
    """GET /health with a loaded bundle returns the bundle's threshold."""
    response = client.get("/health")
    assert response.json()["threshold"] == pytest.approx(0.3)


def test_health_with_no_model_returns_200(client_no_model: TestClient) -> None:
    """GET /health with no model always returns 200."""
    response = client_no_model.get("/health")
    assert response.status_code == 200


def test_health_with_no_model_reports_model_loaded_false(
    client_no_model: TestClient,
) -> None:
    """GET /health with no model reports model_loaded as False."""
    response = client_no_model.get("/health")
    assert response.json()["model_loaded"] is False


def test_health_with_no_model_reports_none_version(
    client_no_model: TestClient,
) -> None:
    """GET /health with no model returns model_version as None."""
    response = client_no_model.get("/health")
    assert response.json()["model_version"] is None


def test_health_with_no_model_reports_none_threshold(
    client_no_model: TestClient,
) -> None:
    """GET /health with no model returns threshold as None."""
    response = client_no_model.get("/health")
    assert response.json()["threshold"] is None
