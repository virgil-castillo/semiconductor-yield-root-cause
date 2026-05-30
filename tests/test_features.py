"""Tests for feature engineering (features.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yield_risk.features import (
    add_aggregate_features,
    add_top_interactions,
    engineer_features,
    get_sensor_cols,
)


def _make_df(n_sensors: int = 5, n_rows: int = 20) -> pd.DataFrame:
    """Build a minimal SECOM-schema DataFrame with *n_sensors* sensor columns.

    Each sensor column has a distinct variance so top-k selection is
    deterministic.  The DataFrame also contains ``label`` and ``timestamp``
    metadata columns.
    """
    rng = np.random.default_rng(0)
    data: dict[str, object] = {}
    for i in range(n_sensors):
        # Scale each sensor so variances are clearly ordered
        data[f"sensor_{i:03d}"] = rng.normal(0, float(i + 1), n_rows).tolist()
    data["label"] = ([0, 1] * n_rows)[:n_rows]
    data["timestamp"] = ["t"] * n_rows
    return pd.DataFrame(data)


class TestGetSensorCols:
    def test_excludes_label_and_timestamp(self) -> None:
        df = _make_df(n_sensors=3)
        cols = get_sensor_cols(df)
        assert "label" not in cols
        assert "timestamp" not in cols

    def test_includes_all_sensor_columns(self) -> None:
        df = _make_df(n_sensors=4)
        cols = get_sensor_cols(df)
        assert sorted(cols) == ["sensor_000", "sensor_001", "sensor_002", "sensor_003"]

    def test_preserves_column_order(self) -> None:
        df = _make_df(n_sensors=3)
        cols = get_sensor_cols(df)
        sensor_positions = [df.columns.get_loc(c) for c in cols]
        assert sensor_positions == sorted(sensor_positions)


class TestAddAggregateFeatures:
    def test_adds_four_columns(self) -> None:
        df = _make_df()
        result = add_aggregate_features(df)
        agg_cols = (
            "sensor_row_mean",
            "sensor_row_std",
            "sensor_row_min",
            "sensor_row_max",
        )
        for col in agg_cols:
            assert col in result.columns

    def test_original_columns_preserved(self) -> None:
        df = _make_df()
        result = add_aggregate_features(df)
        for col in df.columns:
            assert col in result.columns

    def test_row_mean_is_correct(self) -> None:
        df = pd.DataFrame(
            {
                "s0": [1.0, 2.0],
                "s1": [3.0, 4.0],
                "label": [0, 1],
                "timestamp": ["t", "t"],
            }
        )
        result = add_aggregate_features(df)
        assert result["sensor_row_mean"].tolist() == pytest.approx([2.0, 3.0])

    def test_row_min_is_correct(self) -> None:
        df = pd.DataFrame(
            {
                "s0": [1.0, 5.0],
                "s1": [3.0, 2.0],
                "label": [0, 1],
                "timestamp": ["t", "t"],
            }
        )
        result = add_aggregate_features(df)
        assert result["sensor_row_min"].tolist() == pytest.approx([1.0, 2.0])

    def test_row_max_is_correct(self) -> None:
        df = pd.DataFrame(
            {
                "s0": [1.0, 5.0],
                "s1": [3.0, 2.0],
                "label": [0, 1],
                "timestamp": ["t", "t"],
            }
        )
        result = add_aggregate_features(df)
        assert result["sensor_row_max"].tolist() == pytest.approx([3.0, 5.0])

    def test_does_not_mutate_input(self) -> None:
        df = _make_df()
        original_cols = list(df.columns)
        add_aggregate_features(df)
        assert list(df.columns) == original_cols


class TestAddTopInteractions:
    def test_interaction_columns_use_x_separator(self) -> None:
        df = _make_df(n_sensors=4)
        result = add_top_interactions(df, top_k=2)
        interaction_cols = [c for c in result.columns if "__x__" in c]
        assert len(interaction_cols) == 1

    def test_number_of_interaction_cols_is_k_choose_2(self) -> None:
        df = _make_df(n_sensors=5)
        result = add_top_interactions(df, top_k=4)
        interaction_cols = [c for c in result.columns if "__x__" in c]
        # C(4, 2) = 6
        assert len(interaction_cols) == 6

    def test_interaction_values_are_products(self) -> None:
        df = pd.DataFrame(
            {
                "s0": [2.0, 3.0],
                "s1": [4.0, 5.0],
                "label": [0, 1],
                "timestamp": ["t", "t"],
            }
        )
        result = add_top_interactions(df, top_k=2)
        col = "s0__x__s1"
        assert col in result.columns
        assert result[col].tolist() == pytest.approx([8.0, 15.0])

    def test_selects_highest_variance_sensors(self) -> None:
        # sensor_001 has clearly higher variance than sensor_000
        df = pd.DataFrame(
            {
                "sensor_000": [1.0, 1.0, 1.0, 1.0],
                "sensor_001": [1.0, 10.0, -10.0, 5.0],
                "sensor_002": [0.5, 0.5, 0.5, 0.5],
                "label": [0, 1, 0, 1],
                "timestamp": ["t"] * 4,
            }
        )
        # top_k=1 → no pairs; top_k=2 → the highest-variance pair is used
        add_top_interactions(df, top_k=1)
        result2 = add_top_interactions(df, top_k=2)
        interaction_cols = [c for c in result2.columns if "__x__" in c]
        assert len(interaction_cols) == 1
        assert "sensor_001" in interaction_cols[0]

    def test_original_columns_preserved(self) -> None:
        df = _make_df()
        result = add_top_interactions(df, top_k=3)
        for col in df.columns:
            assert col in result.columns

    def test_does_not_mutate_input(self) -> None:
        df = _make_df()
        original_cols = list(df.columns)
        add_top_interactions(df, top_k=3)
        assert list(df.columns) == original_cols


class TestEngineerFeatures:
    def test_contains_aggregate_columns(self) -> None:
        df = _make_df()
        result = engineer_features(df, top_k_interactions=3)
        agg_cols = (
            "sensor_row_mean",
            "sensor_row_std",
            "sensor_row_min",
            "sensor_row_max",
        )
        for col in agg_cols:
            assert col in result.columns

    def test_contains_interaction_columns(self) -> None:
        df = _make_df(n_sensors=4)
        result = engineer_features(df, top_k_interactions=3)
        interaction_cols = [c for c in result.columns if "__x__" in c]
        assert len(interaction_cols) > 0

    def test_original_columns_preserved(self) -> None:
        df = _make_df()
        result = engineer_features(df, top_k_interactions=3)
        for col in df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self) -> None:
        df = _make_df()
        result = engineer_features(df, top_k_interactions=3)
        assert len(result) == len(df)

    def test_interactions_use_raw_sensor_cols_not_aggregates(self) -> None:
        df = _make_df(n_sensors=5)
        result = engineer_features(df, top_k_interactions=3)
        interaction_cols = [c for c in result.columns if "__x__" in c]
        for col in interaction_cols:
            parts = col.split("__x__")
            for part in parts:
                assert not part.startswith("sensor_row_"), (
                    f"Aggregate column '{part}' appeared in interaction '{col}'"
                )
