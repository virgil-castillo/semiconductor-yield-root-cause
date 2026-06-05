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


# ---------------------------------------------------------------------------
# Namespace boundary tests (sensor_000..sensor_589)
# ---------------------------------------------------------------------------


def test_predict_sensor_590_first_invalid_returns_400_naming_key(
    client: TestClient,
) -> None:
    """POST /predict with sensor_590 (first INVALID key) returns 400 naming it."""
    payload = {"features": {"sensor_590": 1.0}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "sensor_590" in response.json()["detail"]


def test_predict_sensor_589_last_valid_returns_200(
    client: TestClient,
) -> None:
    """POST /predict with sensor_589 (last VALID key) returns 200.

    sensor_589 is within the sensor_000..sensor_589 namespace so it passes
    validation; it is not in the fixture's expected_sensors (000..004) and is
    simply ignored by score_frame (NaN-filled for missing expected sensors).
    """
    payload = {"features": {"sensor_589": 0.5}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Malformed-format key tests
# ---------------------------------------------------------------------------


def test_predict_two_digit_sensor_key_returns_400_naming_key(
    client: TestClient,
) -> None:
    """POST /predict with sensor_59 (two digits) returns 400 naming the key."""
    payload = {"features": {"sensor_59": 1.0}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "sensor_59" in response.json()["detail"]


def test_predict_four_digit_sensor_key_returns_400_naming_key(
    client: TestClient,
) -> None:
    """POST /predict with sensor_0590 (four digits) returns 400 naming the key."""
    payload = {"features": {"sensor_0590": 1.0}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "sensor_0590" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /predict/batch tests
# ---------------------------------------------------------------------------


def test_batch_happy_path_returns_200_and_all_fields(
    client: TestClient,
) -> None:
    """POST /predict/batch with 3 valid wafers returns 200 with all fields."""
    payload = {
        "wafers": [
            {"wafer_id": "W-1", "features": {"sensor_000": 1.0, "sensor_001": 0.5}},
            {"wafer_id": "W-2", "features": {"sensor_000": -1.0, "sensor_002": 2.0}},
            {"wafer_id": "W-3", "features": {"sensor_003": 0.0, "sensor_004": -0.5}},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    predictions = body["predictions"]
    assert len(predictions) == 3
    # Check wafer_ids are echoed in input order
    assert predictions[0]["wafer_id"] == "W-1"
    assert predictions[1]["wafer_id"] == "W-2"
    assert predictions[2]["wafer_id"] == "W-3"
    # Check each prediction has all five required fields and correct constants
    for pred in predictions:
        assert "wafer_id" in pred
        assert "failure_probability" in pred
        assert "risk_flag" in pred
        assert "threshold_used" in pred
        assert "model_version" in pred
        assert pred["threshold_used"] == pytest.approx(0.3)
        assert pred["model_version"] == "test_model-abc1234"
        prob = pred["failure_probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
        assert pred["risk_flag"] == (prob >= pred["threshold_used"])


def test_batch_per_wafer_independence_and_order(
    client: TestClient,
) -> None:
    """Batch predictions correspond positionally, are distinct, and vary monotonically.

    Uses inputs whose sensor_000 values span a wide range so that a vectorized
    implementation that collapses all rows to the same value is falsified by the
    monotonicity check.  With the fixture model and random_state=42:
      - W-a (sensor_000=10.0)  → lowest failure probability
      - W-b (empty/imputed)    → middle failure probability
      - W-c (sensor_000=-10.0) → highest failure probability
    """
    payload = {
        "wafers": [
            {"wafer_id": "W-a", "features": {"sensor_000": 10.0}},
            {"wafer_id": "W-b", "features": {}},
            {"wafer_id": "W-c", "features": {"sensor_000": -10.0}},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    # Positional / wafer_id order correctness
    assert predictions[0]["wafer_id"] == "W-a"
    assert predictions[1]["wafer_id"] == "W-b"
    assert predictions[2]["wafer_id"] == "W-c"
    # Empty features wafer still gets a prediction (imputed to training mean)
    assert isinstance(predictions[1]["failure_probability"], float)
    # Per-wafer values are DISTINCT — falsifies any vectorized-collapse bug
    probs = [p["failure_probability"] for p in predictions]
    assert len(set(probs)) == 3, f"Expected 3 distinct probabilities, got {probs}"
    # Monotonicity: high sensor_000 → low failure probability for this model
    prob_a = predictions[0]["failure_probability"]
    prob_b = predictions[1]["failure_probability"]
    prob_c = predictions[2]["failure_probability"]
    assert prob_a < prob_b, (
        f"W-a prob {prob_a} should be < W-b prob {prob_b}"
    )
    assert prob_b < prob_c, (
        f"W-b prob {prob_b} should be < W-c prob {prob_c}"
    )


def test_batch_empty_wafers_list_returns_400(
    client: TestClient,
) -> None:
    """POST /predict/batch with empty wafers list returns 400 with detail."""
    payload: dict[str, list[object]] = {"wafers": []}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 400
    assert "detail" in response.json()


def test_batch_over_size_limit_returns_413(
    client: TestClient,
) -> None:
    """POST /predict/batch with 10001 wafers returns 413 with detail."""
    wafers = [
        {"wafer_id": f"W-{i}", "features": {}} for i in range(10001)
    ]
    payload = {"wafers": wafers}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 413
    assert "detail" in response.json()


def test_batch_exactly_max_size_returns_200(
    client: TestClient,
) -> None:
    """POST /predict/batch with exactly 10000 wafers returns 200 (not 413)."""
    wafers = [
        {"wafer_id": f"W-{i}", "features": {}} for i in range(10000)
    ]
    payload = {"wafers": wafers}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 10000


def test_batch_unknown_key_in_one_wafer_returns_400_naming_key(
    client: TestClient,
) -> None:
    """POST /predict/batch with an invalid key in any wafer returns 400 naming it."""
    payload = {
        "wafers": [
            {"wafer_id": "W-good", "features": {"sensor_000": 1.0}},
            {"wafer_id": "W-bad", "features": {"sensor_000": 0.5, "sensor_999": 2.0}},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "sensor_999" in detail


def test_batch_multi_wafer_bad_key_aggregation_returns_400_naming_all_keys(
    client: TestClient,
) -> None:
    """POST /predict/batch with distinct bad keys across wafers returns 400 naming both.

    Verifies that the set-union aggregation of invalid keys across the whole
    batch reports every offending key, not just the first encountered.
    """
    payload = {
        "wafers": [
            {"wafer_id": "W-1", "features": {"sensor_000": 1.0, "foo": 9.9}},
            {"wafer_id": "W-2", "features": {"sensor_001": 0.5, "sensor_999": 2.0}},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "foo" in detail
    assert "sensor_999" in detail


def test_batch_no_model_returns_503(
    client_no_model: TestClient,
) -> None:
    """POST /predict/batch with no loaded model returns 503 with detail."""
    payload = {
        "wafers": [{"wafer_id": "W-1", "features": {"sensor_000": 1.0}}]
    }
    response = client_no_model.post("/predict/batch", json=payload)
    assert response.status_code == 503
    assert "detail" in response.json()


def test_batch_malformed_body_wafers_not_list_returns_422(
    client: TestClient,
) -> None:
    """POST /predict/batch with wafers not a list returns 422."""
    payload = {"wafers": "not-a-list"}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_batch_malformed_body_wafer_missing_features_returns_422(
    client: TestClient,
) -> None:
    """POST /predict/batch with a wafer missing the features field returns 422."""
    payload = {"wafers": [{"wafer_id": "W-1"}]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422
