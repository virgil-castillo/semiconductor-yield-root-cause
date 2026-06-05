"""Tests for scripts/batch_score.py — regression net for the scoring refactor."""
from __future__ import annotations

from pathlib import Path

import batch_score  # imported via conftest.py sys.path injection
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(sensor_cols: list[str]) -> Pipeline:
    """Return a trivial fitted sklearn Pipeline over *sensor_cols*."""
    X = pd.DataFrame(
        np.random.default_rng(0).random((10, len(sensor_cols))),
        columns=sensor_cols,
    )
    y = [0, 1] * 5
    pipe = Pipeline([("clf", LogisticRegression(random_state=0))])
    pipe.fit(X, y)
    return pipe


def _write_model(tmp_path: Path, sensor_cols: list[str]) -> Path:
    """Fit and dump a tiny pipeline; return its path."""
    pipe = _make_pipeline(sensor_cols)
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipe, model_path)
    return model_path


def _write_input_csv(
    tmp_path: Path,
    sensor_cols: list[str],
    n_rows: int = 4,
) -> Path:
    """Write a small input CSV with sensor columns plus non-sensor columns."""
    rng = np.random.default_rng(42)
    data: dict[str, list[object]] = {
        "wafer_id": [f"W{i}" for i in range(n_rows)],
        "label": list(rng.integers(0, 2, n_rows)),
    }
    for col in sensor_cols:
        data[col] = list(rng.random(n_rows))
    df = pd.DataFrame(data)
    csv_path = tmp_path / "input.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SENSOR_COLS = ["sensor_001", "sensor_002", "sensor_003"]


class TestScoreBatch:
    """Regression tests for score_batch after the yield_risk.scoring refactor."""

    def test_returned_frame_has_original_columns_then_score_and_predicted_label(
        self, tmp_path: Path
    ) -> None:
        """score_batch appends score and predicted_label after all original cols."""
        model_path = _write_model(tmp_path, SENSOR_COLS)
        input_path = _write_input_csv(tmp_path, SENSOR_COLS)
        output_path = tmp_path / "out" / "scores.csv"

        result = batch_score.score_batch(model_path, input_path, output_path)

        original_df = pd.read_csv(input_path)
        expected_cols = list(original_df.columns) + ["score", "predicted_label"]
        assert list(result.columns) == expected_cols

    def test_predicted_label_equals_score_ge_threshold(self, tmp_path: Path) -> None:
        """predicted_label is 1 iff score >= threshold (default 0.5)."""
        model_path = _write_model(tmp_path, SENSOR_COLS)
        input_path = _write_input_csv(tmp_path, SENSOR_COLS)
        output_path = tmp_path / "scores.csv"

        result = batch_score.score_batch(model_path, input_path, output_path)

        expected_labels = (result["score"] >= 0.5).astype(int)
        pd.testing.assert_series_equal(
            result["predicted_label"],
            expected_labels,
            check_names=False,
        )

    def test_output_csv_written_and_round_trips(self, tmp_path: Path) -> None:
        """The output CSV exists and reads back to the same DataFrame."""
        model_path = _write_model(tmp_path, SENSOR_COLS)
        input_path = _write_input_csv(tmp_path, SENSOR_COLS)
        output_path = tmp_path / "scores.csv"

        result = batch_score.score_batch(model_path, input_path, output_path)

        assert output_path.exists()
        loaded = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), loaded)

    def test_non_default_threshold_changes_predicted_label(
        self, tmp_path: Path
    ) -> None:
        """A threshold of 0.9 labels only rows with score >= 0.9 as failures."""
        model_path = _write_model(tmp_path, SENSOR_COLS)
        input_path = _write_input_csv(tmp_path, SENSOR_COLS, n_rows=20)
        output_path = tmp_path / "scores.csv"

        result = batch_score.score_batch(
            model_path, input_path, output_path, threshold=0.9
        )

        expected_labels = (result["score"] >= 0.9).astype(int)
        pd.testing.assert_series_equal(
            result["predicted_label"],
            expected_labels,
            check_names=False,
        )

    def test_no_sensor_columns_raises_value_error(self, tmp_path: Path) -> None:
        """score_batch raises ValueError when the input CSV has no sensor_ cols."""
        model_path = _write_model(tmp_path, SENSOR_COLS)

        # CSV with no sensor_ columns
        no_sensor_csv = tmp_path / "no_sensors.csv"
        pd.DataFrame({"wafer_id": ["W0", "W1"], "label": [0, 1]}).to_csv(
            no_sensor_csv, index=False
        )
        output_path = tmp_path / "scores.csv"

        with pytest.raises(ValueError, match="sensor_"):
            batch_score.score_batch(model_path, no_sensor_csv, output_path)
