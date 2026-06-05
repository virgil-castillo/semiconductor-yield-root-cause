"""Tests for scripts/export_model_metadata.py.

All fixtures are self-contained in tmp_path — no real artifacts required.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from export_model_metadata import export_model_metadata
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SENSOR_COLS = ["sensor_001", "sensor_002", "sensor_003"]
FAMILY = "random_forest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline() -> Pipeline:
    """Return a tiny fitted Pipeline over SENSOR_COLS."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((40, len(SENSOR_COLS))), columns=SENSOR_COLS)
    y = pd.Series((X["sensor_001"] > 0.5).astype(int))
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="stratified", random_state=0)),
        ]
    )
    pipe.fit(X, y)
    return pipe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def model_path(tmp_path: Path) -> Path:
    """Dump a fitted pipeline; return the path."""
    path = tmp_path / "selected_model.joblib"
    joblib.dump(_make_pipeline(), path)
    return path


@pytest.fixture()
def test_csv_path(tmp_path: Path) -> Path:
    """Write a test.csv with sensor cols + label; return path."""
    rng = np.random.default_rng(1)
    n = 30
    df = pd.DataFrame(rng.random((n, len(SENSOR_COLS))), columns=SENSOR_COLS)
    df["label"] = (rng.random(n) > 0.5).astype(int)
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def cv_results_path(tmp_path: Path) -> Path:
    """Write a cv_results.json with one selected entry; return path."""
    records = [
        {"model": "logistic_regression", "cv_pr_auc_mean": 0.72, "selected": False},
        {"model": FAMILY, "cv_pr_auc_mean": 0.85, "selected": True},
    ]
    path = tmp_path / "cv_results.json"
    path.write_text(json.dumps(records))
    return path


@pytest.fixture()
def metrics_path(tmp_path: Path) -> Path:
    """Write a selected_model_metrics.json; return path."""
    metrics: dict[str, Any] = {
        "roc_auc": 0.91,
        "pr_auc": 0.85,
        "precision": 0.80,
        "recall": 0.75,
        "f1": 0.77,
        "confusion_matrix": [[10, 2], [3, 15]],
    }
    path = tmp_path / "selected_model_metrics.json"
    path.write_text(json.dumps(metrics))
    return path


@pytest.fixture()
def cost_config_path(tmp_path: Path) -> Path:
    """Write a minimal cost_config.yaml; return path."""
    config = {
        "cost_matrix": {
            "true_pass": 0.0,
            "true_fail": 0.0,
            "false_fail": 1.0,
            "false_pass": 10.0,
        },
        "threshold_search": {
            "low": 0.1,
            "high": 0.9,
            "steps": 9,
        },
    }
    path = tmp_path / "cost_config.yaml"
    path.write_text(yaml.dump(config))
    return path


@pytest.fixture()
def output_path(tmp_path: Path) -> Path:
    """Return path for model_metadata.json (not yet created)."""
    return tmp_path / "models" / "model_metadata.json"


@pytest.fixture()
def metadata(
    model_path: Path,
    test_csv_path: Path,
    cv_results_path: Path,
    metrics_path: Path,
    cost_config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run export_model_metadata and return the result dict."""
    return export_model_metadata(
        model_path=model_path,
        test_path=test_csv_path,
        cv_results_path=cv_results_path,
        metrics_path=metrics_path,
        cost_config_path=cost_config_path,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Tests — output file
# ---------------------------------------------------------------------------


def test_output_file_is_created(
    metadata: dict[str, Any], output_path: Path
) -> None:
    """output_path exists after export_model_metadata runs."""
    assert output_path.exists()


def test_output_file_is_valid_json(
    metadata: dict[str, Any], output_path: Path
) -> None:
    """The written file is valid JSON and matches the returned dict."""
    loaded = json.loads(output_path.read_text())
    assert loaded == metadata


# ---------------------------------------------------------------------------
# Tests — model_version
# ---------------------------------------------------------------------------


def test_model_version_family_prefix(
    metadata: dict[str, Any], model_path: Path
) -> None:
    """model_version starts with the selected family name."""
    assert metadata["model_version"].startswith(f"{FAMILY}-")


def test_model_version_short_hash(
    metadata: dict[str, Any], model_path: Path
) -> None:
    """model_version suffix equals first 7 chars of sha256 of the model file."""
    expected_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:7]
    assert metadata["model_version"] == f"{FAMILY}-{expected_hash}"


# ---------------------------------------------------------------------------
# Tests — optimal_threshold
# ---------------------------------------------------------------------------


def test_optimal_threshold_is_float(metadata: dict[str, Any]) -> None:
    """optimal_threshold is a Python float."""
    assert isinstance(metadata["optimal_threshold"], float)


def test_optimal_threshold_within_search_range(metadata: dict[str, Any]) -> None:
    """optimal_threshold is within [0.1, 0.9] (the fixture search range)."""
    assert 0.1 <= metadata["optimal_threshold"] <= 0.9


# ---------------------------------------------------------------------------
# Tests — cost_matrix
# ---------------------------------------------------------------------------


def test_cost_matrix_equals_config(
    metadata: dict[str, Any], cost_config_path: Path
) -> None:
    """cost_matrix in output equals dataclasses.asdict of the loaded CostMatrix."""
    raw = yaml.safe_load(cost_config_path.read_text())
    expected = raw["cost_matrix"]
    assert metadata["cost_matrix"] == expected


# ---------------------------------------------------------------------------
# Tests — created_at
# ---------------------------------------------------------------------------


def test_created_at_is_iso8601_utc(metadata: dict[str, Any]) -> None:
    """created_at parses as an ISO-8601 UTC-aware datetime."""
    dt = datetime.fromisoformat(metadata["created_at"])
    assert dt.tzinfo is not None
    assert dt.utcoffset() is not None
    # UTC offset must be zero
    assert dt.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Tests — metrics
# ---------------------------------------------------------------------------


def test_metrics_equals_metrics_file(
    metadata: dict[str, Any], metrics_path: Path
) -> None:
    """metrics in output equals the parsed contents of selected_model_metrics.json."""
    expected = json.loads(metrics_path.read_text())
    assert metadata["metrics"] == expected


# ---------------------------------------------------------------------------
# Tests — expected_sensors
# ---------------------------------------------------------------------------


def test_expected_sensors_equals_sensor_columns(
    metadata: dict[str, Any], test_csv_path: Path
) -> None:
    """expected_sensors matches the sensor_ columns in the test CSV."""
    df = pd.read_csv(test_csv_path)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    assert metadata["expected_sensors"] == sensor_cols
