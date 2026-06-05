"""Tests for the FastAPI prediction API — health endpoint and reusable fixtures.

Later tasks (predict, batch) extend this file by importing the fixtures.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yield_risk.scoring import ModelBundle

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SENSOR_COLS = [f"sensor_{i:03d}" for i in range(5)]


def _make_bundle() -> ModelBundle:
    """Build a tiny in-memory ModelBundle for testing without touching disk.

    Returns:
        A real ModelBundle with a fitted sklearn Pipeline, a fixed version
        string, threshold, and expected_sensors list.
    """
    X = pd.DataFrame(
        [[0.1 * i for _ in _SENSOR_COLS] for i in range(10)],
        columns=_SENSOR_COLS,
    )
    y = [0] * 5 + [1] * 5

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="prior")),
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
def client() -> TestClient:
    """TestClient with a loaded model bundle (lifespan does NOT run).

    Returns:
        TestClient with app.state.bundle set to a real in-memory bundle.
    """
    from api.main import app  # noqa: PLC0415

    tc = TestClient(app, raise_server_exceptions=True)
    app.state.bundle = _make_bundle()
    return tc


@pytest.fixture()
def client_no_model() -> TestClient:
    """TestClient with no model bundle loaded (lifespan does NOT run).

    Returns:
        TestClient with app.state.bundle explicitly set to None.
    """
    from api.main import app  # noqa: PLC0415

    tc = TestClient(app, raise_server_exceptions=True)
    app.state.bundle = None
    return tc


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
