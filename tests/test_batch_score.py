"""Tests for scripts/batch_score.py — regression net for the scoring refactor."""
from __future__ import annotations

import json
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
        """A threshold of 0.9 labels only rows with score >= 0.9 as failures.

        Uses a fixed RNG seed (hard-coded as 42 inside _write_input_csv, plus
        rng=0 inside _make_pipeline) that produces scores in [0.39, 0.57] for
        n_rows=20, so every row flips from 0.5-threshold to 0.9-threshold,
        guaranteeing the discriminating assertion is deterministically true.
        """
        model_path = _write_model(tmp_path, SENSOR_COLS)
        input_path = _write_input_csv(tmp_path, SENSOR_COLS, n_rows=20)
        output_path_05 = tmp_path / "scores_05.csv"
        output_path_09 = tmp_path / "scores_09.csv"

        result_at_05 = batch_score.score_batch(
            model_path, input_path, output_path_05, threshold=0.5
        )
        result_at_09 = batch_score.score_batch(
            model_path, input_path, output_path_09, threshold=0.9
        )

        # Labels must exactly follow the threshold rule.
        expected_labels = (result_at_09["score"] >= 0.9).astype(int)
        pd.testing.assert_series_equal(
            result_at_09["predicted_label"],
            expected_labels,
            check_names=False,
        )

        # Discriminating: at least one label actually changed between the two
        # thresholds (guaranteed by the fixed seeds — scores span [0.39, 0.57]).
        labels_05 = result_at_05["predicted_label"]
        labels_09 = result_at_09["predicted_label"]
        assert (labels_05 != labels_09).any(), (
            "No label differed between threshold=0.5 and threshold=0.9; "
            "the threshold argument may not be taking effect."
        )

    def test_cli_threshold_overrides_bundle_metadata_threshold(
        self, tmp_path: Path
    ) -> None:
        """The caller-supplied threshold wins over the bundle's metadata threshold.

        Writes a model_metadata.json with optimal_threshold=0.01, then calls
        score_batch with threshold=0.9.  The fixed seeds produce scores in
        [0.43, 0.55], so every score is >= 0.01 but < 0.9.  If the metadata
        threshold leaked through, all labels would be 1; with threshold=0.9 they
        must all be 0.
        """
        model_path = _write_model(tmp_path, SENSOR_COLS)
        input_path = _write_input_csv(tmp_path, SENSOR_COLS)  # n_rows=4, seed=42
        output_path = tmp_path / "scores.csv"

        # Write metadata with a very low threshold (0.01) beside the model.
        metadata = {
            "optimal_threshold": 0.01,
            "model_version": "test-v1",
            "expected_sensors": SENSOR_COLS,
        }
        (tmp_path / "model_metadata.json").write_text(json.dumps(metadata))

        result = batch_score.score_batch(
            model_path, input_path, output_path, threshold=0.9
        )

        # Labels must reflect the caller's 0.9, not the bundle's 0.01.
        expected_at_09 = (result["score"] >= 0.9).astype(int)
        pd.testing.assert_series_equal(
            result["predicted_label"],
            expected_at_09,
            check_names=False,
        )

        # Discriminating: the labels would be different if 0.01 had been used
        # (all scores are in [0.01, 0.9), so labels_at_0.01 would all be 1).
        labels_at_001 = (result["score"] >= 0.01).astype(int)
        assert (result["predicted_label"] != labels_at_001).any(), (
            "Labels match threshold=0.01 for all rows; the bundle metadata "
            "threshold may have overridden the caller-supplied threshold=0.9."
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
