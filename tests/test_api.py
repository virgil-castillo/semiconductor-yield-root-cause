"""Tests for the FastAPI prediction API — health endpoint and reusable fixtures.

Later tasks (predict, batch) extend this file by importing the fixtures.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer
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
    on a pandas DataFrame.  A SimpleImputer (strategy='mean') is the first
    pipeline step to handle partial feature maps (missing sensors → NaN),
    mirroring the real production pipeline.

    Returns:
        A real ModelBundle with a fitted sklearn Pipeline (imputer + scaler +
        classifier), a fixed version string ``"test_model-abc1234"``,
        threshold ``0.3``, and ``expected_sensors`` list of sensor_000..sensor_004.
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((20, len(_SENSOR_COLS))),
        columns=_SENSOR_COLS,
    )
    y = [0] * 10 + [1] * 10

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
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


# ---------------------------------------------------------------------------
# /predict tests
# ---------------------------------------------------------------------------


def test_predict_happy_path_returns_200_and_all_fields(
    client: TestClient,
) -> None:
    """POST /predict with all sensors returns 200 and the full response shape."""
    payload = {
        "wafer_id": "W-00123",
        "features": {col: 1.0 for col in _SENSOR_COLS},
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["wafer_id"] == "W-00123"
    assert body["model_version"] == "test_model-abc1234"
    assert body["threshold_used"] == pytest.approx(0.3)
    prob = body["failure_probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    # Consistency: risk_flag must equal failure_probability >= threshold_used
    assert body["risk_flag"] == (prob >= body["threshold_used"])


def test_predict_partial_feature_map_returns_200(
    client: TestClient,
) -> None:
    """POST /predict with only one valid sensor (rest NaN-imputed) returns 200."""
    payload = {
        "wafer_id": "W-partial",
        "features": {"sensor_000": 0.5},
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["failure_probability"], float)


def test_predict_without_wafer_id_returns_200_and_null_wafer_id(
    client: TestClient,
) -> None:
    """POST /predict without wafer_id returns 200 with wafer_id null."""
    payload = {"features": {"sensor_000": 0.5}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["wafer_id"] is None


def test_predict_unknown_sensor_key_returns_400_naming_offending_keys(
    client: TestClient,
) -> None:
    """POST /predict with out-of-namespace keys returns 400 naming them."""
    payload = {
        "features": {"sensor_000": 1.0, "sensor_999": 0.5, "foo": 2.0},
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "sensor_999" in detail
    assert "foo" in detail


def test_predict_valid_namespace_key_not_in_expected_sensors_returns_200(
    client: TestClient,
) -> None:
    """POST /predict with sensor_100 (valid namespace, outside expected) returns 200."""
    # sensor_100 is within sensor_000..sensor_589 but NOT in _SENSOR_COLS (000..004)
    payload = {
        "wafer_id": "W-ns",
        "features": {"sensor_100": 0.7, "sensor_000": 1.2},
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_no_model_returns_503_with_error_response(
    client_no_model: TestClient,
) -> None:
    """POST /predict with no loaded model returns 503 with detail."""
    payload = {"features": {"sensor_000": 0.5}}
    response = client_no_model.post("/predict", json=payload)
    assert response.status_code == 503
    assert "detail" in response.json()


def test_predict_malformed_body_missing_features_returns_422(
    client: TestClient,
) -> None:
    """POST /predict with missing features field returns 422."""
    payload = {"wafer_id": "W-bad"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_malformed_body_non_numeric_feature_value_returns_422(
    client: TestClient,
) -> None:
    """POST /predict with non-numeric feature value returns 422."""
    payload = {"features": {"sensor_000": "not-a-number"}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_malformed_body_features_not_object_returns_422(
    client: TestClient,
) -> None:
    """POST /predict when features is not a dict returns 422."""
    payload = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
