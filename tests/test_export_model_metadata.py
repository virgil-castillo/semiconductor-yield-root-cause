"""Tests for scripts/export_model_metadata.py.

All fixtures are self-contained in tmp_path — no real artifacts required.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from export_model_metadata import export_model_metadata
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yield_risk.evaluate import compute_metrics
from yield_risk.model import model_feature_names
from yield_risk.preprocess import SecomPreprocessor

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SENSOR_COLS = ["sensor_001", "sensor_002", "sensor_003"]
FAMILY = "random_forest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline() -> Pipeline:
    """Return a tiny fitted Pipeline over SENSOR_COLS.

    Includes a SecomPreprocessor as the first step so that
    model_feature_names() can be called on the resulting pipeline.
    Thresholds are chosen to keep all three sensor columns.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((40, len(SENSOR_COLS))), columns=SENSOR_COLS)
    y = pd.Series((X["sensor_001"] > 0.5).astype(int))
    pipe = Pipeline(
        [
            (
                "preprocess",
                SecomPreprocessor(
                    missing_threshold=0.9,
                    variance_threshold=0.0,
                    correlation_threshold=0.999,
                ),
            ),
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
    """Write a cv_results.json with one selected entry; return path.

    Each entry includes a frozen ``threshold`` field produced by train_models.py
    after the second-leak fix.
    """
    records = [
        {
            "model": "logistic_regression",
            "cv_pr_auc_mean": 0.72,
            "selected": False,
            "threshold": 0.4,
        },
        {
            "model": FAMILY,
            "cv_pr_auc_mean": 0.85,
            "selected": True,
            "threshold": 0.3,
        },
    ]
    path = tmp_path / "cv_results.json"
    path.write_text(json.dumps(records))
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
    cost_config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run export_model_metadata and return the result dict."""
    return export_model_metadata(
        model_path=model_path,
        test_path=test_csv_path,
        cv_results_path=cv_results_path,
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
# Tests — frozen_threshold
# ---------------------------------------------------------------------------


def test_frozen_threshold_is_float(metadata: dict[str, Any]) -> None:
    """frozen_threshold is a Python float."""
    assert isinstance(metadata["frozen_threshold"], float)


def test_frozen_threshold_is_frozen_from_cv_results(
    metadata: dict[str, Any],
    cv_results_path: Path,
) -> None:
    """frozen_threshold equals the selected entry's frozen threshold from cv_results.

    After the second-leak fix, export_model_metadata reads the threshold from
    the cv_results.json artifact (produced by train_models.py using OOF train
    predictions) rather than computing it from test labels.
    """
    cv_data = json.loads(cv_results_path.read_text())
    selected_entry = next(r for r in cv_data if r["selected"])
    assert metadata["frozen_threshold"] == float(selected_entry["threshold"])


def test_export_never_calls_find_optimal_threshold_on_test(
    model_path: Path,
    test_csv_path: Path,
    cv_results_path: Path,
    cost_config_path: Path,
    output_path: Path,
) -> None:
    """export_model_metadata never calls find_optimal_threshold on test data.

    Patches find_optimal_threshold at its definition site and asserts it is
    never invoked during export_model_metadata execution.
    """
    spy = Mock()
    with patch("yield_risk.thresholding.find_optimal_threshold", spy):
        export_model_metadata(
            model_path=model_path,
            test_path=test_csv_path,
            cv_results_path=cv_results_path,
            cost_config_path=cost_config_path,
            output_path=output_path,
        )
    spy.assert_not_called()


def test_missing_threshold_key_raises_value_error(
    model_path: Path,
    test_csv_path: Path,
    cost_config_path: Path,
    output_path: Path,
    tmp_path: Path,
) -> None:
    """export_model_metadata raises ValueError when selected entry lacks 'threshold'.

    Simulates an artifact produced by an older train_models.py that did not
    persist the frozen threshold.
    """
    old_cv_results = [
        {"model": "logistic_regression", "cv_pr_auc_mean": 0.72, "selected": False},
        {"model": FAMILY, "cv_pr_auc_mean": 0.85, "selected": True},
        # Note: no "threshold" key in the selected entry
    ]
    bad_cv_path = tmp_path / "cv_old_format.json"
    bad_cv_path.write_text(json.dumps(old_cv_results))

    with pytest.raises(ValueError, match="threshold"):
        export_model_metadata(
            model_path=model_path,
            test_path=test_csv_path,
            cv_results_path=bad_cv_path,
            cost_config_path=cost_config_path,
            output_path=output_path,
        )


# ---------------------------------------------------------------------------
# Tests — error handling
# ---------------------------------------------------------------------------


def test_no_selected_model_raises_value_error(
    model_path: Path,
    test_csv_path: Path,
    cost_config_path: Path,
    output_path: Path,
    tmp_path: Path,
) -> None:
    """export_model_metadata raises ValueError when no entry has selected=True."""
    no_selected = [
        {"model": "logistic_regression", "cv_pr_auc_mean": 0.72, "selected": False},
        {"model": "random_forest", "cv_pr_auc_mean": 0.85, "selected": False},
    ]
    bad_cv_path = tmp_path / "cv_no_selected.json"
    bad_cv_path.write_text(json.dumps(no_selected))

    with pytest.raises(ValueError, match="No selected model"):
        export_model_metadata(
            model_path=model_path,
            test_path=test_csv_path,
            cv_results_path=bad_cv_path,
            cost_config_path=cost_config_path,
            output_path=output_path,
        )


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


def test_metrics_computed_directly_from_model_and_test_data(
    metadata: dict[str, Any], model_path: Path, test_csv_path: Path
) -> None:
    """metrics are recomputed from the pipeline + test data at the opt threshold.

    This guards against the stale-file bug: metrics must reflect THIS model and
    test set, not a side-effect file from another script. We independently
    recompute the same metrics at the exported threshold and require a match.
    """
    pipeline = joblib.load(model_path)
    df = pd.read_csv(test_csv_path)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    y_true = df["label"].to_numpy()
    y_prob = pipeline.predict_proba(df[sensor_cols])[:, 1]
    expected = dataclasses.asdict(
        compute_metrics(y_true, y_prob, threshold=metadata["frozen_threshold"])
    )
    expected["threshold"] = metadata["frozen_threshold"]
    assert metadata["metrics"] == expected


def test_metrics_threshold_matches_frozen_threshold(
    metadata: dict[str, Any],
) -> None:
    """The metrics block's threshold is the exported cost-optimal threshold.

    Ensures internal consistency between the metrics snapshot and the served
    operating point.
    """
    assert metadata["metrics"]["threshold"] == metadata["frozen_threshold"]


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


# ---------------------------------------------------------------------------
# Tests — selected_features
# ---------------------------------------------------------------------------


def test_selected_features_equals_model_feature_names(
    metadata: dict[str, Any], model_path: Path
) -> None:
    """selected_features equals model_feature_names of the loaded pipeline."""
    pipeline = joblib.load(model_path)
    assert metadata["selected_features"] == model_feature_names(pipeline)


def test_selected_features_is_subset_of_expected_sensors(
    metadata: dict[str, Any],
) -> None:
    """Every entry in selected_features is present in expected_sensors."""
    expected_set = set(metadata["expected_sensors"])
    for feature in metadata["selected_features"]:
        assert feature in expected_set
