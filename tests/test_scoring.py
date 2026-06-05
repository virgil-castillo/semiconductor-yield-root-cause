"""Tests for yield_risk.scoring — framework-agnostic scoring service."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yield_risk.scoring import ModelBundle, load_model_bundle, score_frame

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SENSOR_COLS = ["sensor_001", "sensor_002", "sensor_003"]


def _make_pipeline() -> Pipeline:
    """Return a fitted Pipeline over three sensor columns."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((50, 3)), columns=SENSOR_COLS)
    y = pd.Series((X["sensor_001"] > 0.5).astype(int))
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", DummyClassifier(strategy="stratified", random_state=0)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture()
def tmp_model(tmp_path: Path) -> Path:
    """Dump a fitted pipeline to tmp_path/model.joblib; return the path."""
    pipeline = _make_pipeline()
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    return model_path


@pytest.fixture()
def tmp_model_with_metadata(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Dump model + metadata JSON beside it; return (model_path, metadata)."""
    pipeline = _make_pipeline()
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    metadata: dict[str, Any] = {
        "model_version": "v1.2.3",
        "optimal_threshold": 0.35,
        "expected_sensors": SENSOR_COLS,
    }
    (tmp_path / "model_metadata.json").write_text(json.dumps(metadata))
    return model_path, metadata


@pytest.fixture()
def bundle(tmp_model_with_metadata: tuple[Path, dict[str, Any]]) -> ModelBundle:
    """Return a loaded ModelBundle from the metadata fixture."""
    model_path, _ = tmp_model_with_metadata
    return load_model_bundle(model_path)


# ---------------------------------------------------------------------------
# load_model_bundle — happy path (metadata present)
# ---------------------------------------------------------------------------


def test_load_model_bundle_reads_optimal_threshold(
    tmp_model_with_metadata: tuple[Path, dict[str, Any]],
) -> None:
    """Threshold comes from metadata's optimal_threshold key."""
    model_path, metadata = tmp_model_with_metadata
    result = load_model_bundle(model_path)
    assert result.threshold == metadata["optimal_threshold"]


def test_load_model_bundle_reads_model_version(
    tmp_model_with_metadata: tuple[Path, dict[str, Any]],
) -> None:
    """model_version comes from metadata's model_version key."""
    model_path, metadata = tmp_model_with_metadata
    result = load_model_bundle(model_path)
    assert result.model_version == metadata["model_version"]


def test_load_model_bundle_reads_expected_sensors(
    tmp_model_with_metadata: tuple[Path, dict[str, Any]],
) -> None:
    """expected_sensors comes from metadata's expected_sensors key."""
    model_path, metadata = tmp_model_with_metadata
    result = load_model_bundle(model_path)
    assert result.expected_sensors == metadata["expected_sensors"]


def test_load_model_bundle_pipeline_is_loaded(
    tmp_model_with_metadata: tuple[Path, dict[str, Any]],
) -> None:
    """The loaded bundle contains a sklearn Pipeline."""
    model_path, _ = tmp_model_with_metadata
    result = load_model_bundle(model_path)
    assert isinstance(result.pipeline, Pipeline)


def test_load_model_bundle_raises_value_error_on_missing_metadata_key(
    tmp_path: Path,
) -> None:
    """ValueError is raised when metadata JSON exists but is missing a required key."""
    pipeline = _make_pipeline()
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    # Write metadata that is missing 'optimal_threshold'
    incomplete_metadata = {
        "model_version": "v1.2.3",
        "expected_sensors": SENSOR_COLS,
        # 'optimal_threshold' deliberately omitted
    }
    metadata_path = tmp_path / "model_metadata.json"
    metadata_path.write_text(json.dumps(incomplete_metadata))

    with pytest.raises(ValueError, match="optimal_threshold"):
        load_model_bundle(model_path)


# ---------------------------------------------------------------------------
# load_model_bundle — fallback (no metadata file)
# ---------------------------------------------------------------------------


def test_load_model_bundle_fallback_threshold(tmp_model: Path) -> None:
    """Without metadata, threshold defaults to 0.5."""
    result = load_model_bundle(tmp_model)
    assert result.threshold == 0.5


def test_load_model_bundle_fallback_model_version(tmp_model: Path) -> None:
    """Without metadata, model_version is 'model-<7hex>' derived from sha256."""
    result = load_model_bundle(tmp_model)
    raw = tmp_model.read_bytes()
    short_hash = hashlib.sha256(raw).hexdigest()[:7]
    assert result.model_version == f"model-{short_hash}"


def test_load_model_bundle_fallback_expected_sensors_from_pipeline(
    tmp_model: Path,
) -> None:
    """Without metadata, expected_sensors comes from pipeline's feature_names_in_."""
    result = load_model_bundle(tmp_model)
    assert result.expected_sensors == SENSOR_COLS


def test_load_model_bundle_fallback_warns(
    tmp_model: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Without metadata a WARNING is logged."""
    with caplog.at_level(logging.WARNING, logger="yield_risk.scoring"):
        load_model_bundle(tmp_model)
    assert any("warning" in r.levelname.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# score_frame — column alignment
# ---------------------------------------------------------------------------


def test_score_frame_output_has_score_and_predicted_label(bundle: ModelBundle) -> None:
    """Output DataFrame contains 'score' and 'predicted_label' columns."""
    df = pd.DataFrame({"sensor_001": [0.1], "sensor_002": [0.5], "sensor_003": [0.9]})
    out = score_frame(bundle, df)
    assert "score" in out.columns
    assert "predicted_label" in out.columns


def test_score_frame_preserves_original_columns(bundle: ModelBundle) -> None:
    """All input columns appear in the output."""
    df = pd.DataFrame(
        {
            "sensor_001": [0.1],
            "sensor_002": [0.5],
            "sensor_003": [0.9],
            "wafer_id": ["W001"],
        }
    )
    out = score_frame(bundle, df)
    for col in df.columns:
        assert col in out.columns


def test_score_frame_missing_sensors_filled_with_nan(bundle: ModelBundle) -> None:
    """Sensors missing from input are passed to the pipeline as NaN, in order.

    Verifies that:
    - The X matrix passed to predict_proba has columns equal to
      bundle.expected_sensors in that exact order.
    - Missing sensor columns contain NaN.
    - The provided sensor column's value is preserved.
    """
    # Only provide sensor_001; sensor_002 and sensor_003 are absent
    df = pd.DataFrame({"sensor_001": [0.3]})

    captured: list[pd.DataFrame] = []

    original_predict_proba = bundle.pipeline.predict_proba

    def _capturing_predict_proba(X: pd.DataFrame) -> np.ndarray:
        captured.append(X.copy())
        return original_predict_proba(X)

    with patch.object(
        bundle.pipeline, "predict_proba", side_effect=_capturing_predict_proba
    ):
        out = score_frame(bundle, df)

    assert "score" in out.columns
    assert len(captured) == 1
    X_received = captured[0]

    # Columns must exactly match expected_sensors in order
    assert list(X_received.columns) == bundle.expected_sensors

    # The provided sensor value is preserved
    assert float(X_received["sensor_001"].iloc[0]) == pytest.approx(0.3)

    # Missing sensors are NaN
    for col in ["sensor_002", "sensor_003"]:
        actual = X_received[col].iloc[0]
        assert pd.isna(actual), f"{col} should be NaN but got {actual}"


def test_score_frame_does_not_mutate_input(bundle: ModelBundle) -> None:
    """score_frame must not modify the caller's DataFrame."""
    df = pd.DataFrame({"sensor_001": [0.1], "sensor_002": [0.5], "sensor_003": [0.9]})
    original_cols = list(df.columns)
    original_shape = df.shape
    score_frame(bundle, df)
    assert list(df.columns) == original_cols
    assert df.shape == original_shape
    assert "score" not in df.columns
    assert "predicted_label" not in df.columns


def test_score_frame_extra_columns_ignored(bundle: ModelBundle) -> None:
    """Columns not in expected_sensors are ignored for scoring but kept in output."""
    df = pd.DataFrame(
        {
            "sensor_001": [0.1],
            "sensor_002": [0.5],
            "sensor_003": [0.9],
            "sensor_999": [99.0],  # not in expected_sensors
        }
    )
    out = score_frame(bundle, df)
    assert "score" in out.columns
    assert "sensor_999" in out.columns  # preserved in output


# ---------------------------------------------------------------------------
# score_frame — threshold application
# ---------------------------------------------------------------------------


def test_score_frame_predicted_label_matches_threshold(bundle: ModelBundle) -> None:
    """predicted_label equals int(score >= threshold) for every row."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        rng.random((20, 3)), columns=["sensor_001", "sensor_002", "sensor_003"]
    )
    out = score_frame(bundle, df)
    expected = (out["score"] >= bundle.threshold).astype(int)
    pd.testing.assert_series_equal(
        out["predicted_label"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_score_frame_override_threshold(bundle: ModelBundle) -> None:
    """Passing threshold arg overrides bundle.threshold."""
    df = pd.DataFrame({"sensor_001": [0.1], "sensor_002": [0.5], "sensor_003": [0.9]})
    out_low = score_frame(bundle, df, threshold=0.0)
    out_high = score_frame(bundle, df, threshold=1.0)
    # threshold=0.0 → always predict 1; threshold=1.0 → always predict 0
    assert int(out_low["predicted_label"].iloc[0]) == 1
    assert int(out_high["predicted_label"].iloc[0]) == 0


def test_score_frame_uses_bundle_threshold_when_arg_none(bundle: ModelBundle) -> None:
    """When threshold arg is None, bundle.threshold is used."""
    df = pd.DataFrame({"sensor_001": [0.1], "sensor_002": [0.5], "sensor_003": [0.9]})
    out_default = score_frame(bundle, df, threshold=None)
    out_explicit = score_frame(bundle, df, threshold=bundle.threshold)
    pd.testing.assert_frame_equal(out_default, out_explicit)


def test_score_frame_predicted_label_is_int(bundle: ModelBundle) -> None:
    """predicted_label dtype is int (0 or 1), not bool."""
    df = pd.DataFrame({"sensor_001": [0.1], "sensor_002": [0.5], "sensor_003": [0.9]})
    out = score_frame(bundle, df)
    # Values must be 0 or 1 and dtype should be integer-like
    assert set(out["predicted_label"].unique()).issubset({0, 1})
    assert out["predicted_label"].dtype in (
        np.dtype("int64"),
        np.dtype("int32"),
        np.dtype("int8"),
    )
