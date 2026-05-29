"""Tests for the preprocessing pipeline (preprocess.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yield_risk.preprocess import (
    drop_high_correlation,
    drop_high_missing,
    drop_low_variance,
    impute_median,
    split_stratified,
)


def _df(**sensor_vals: list[float]) -> pd.DataFrame:
    """Build a minimal SECOM-schema DataFrame from keyword sensor columns."""
    n = len(next(iter(sensor_vals.values())))
    return pd.DataFrame(
        {**sensor_vals, "label": ([0, 1] * n)[:n], "timestamp": ["t"] * n}
    )


class TestDropHighMissing:
    def test_drops_column_above_threshold(self) -> None:
        df = _df(
            sensor_000=[1.0, 2.0, 3.0, 4.0, 5.0],
            sensor_001=[1.0, float("nan"), float("nan"), float("nan"), float("nan")],
        )
        result = drop_high_missing(df, threshold=0.5)
        assert "sensor_001" not in result.columns

    def test_keeps_column_at_threshold(self) -> None:
        # exactly 50 % missing — not strictly above threshold, so kept
        df = _df(
            sensor_000=[1.0, 2.0, 3.0, 4.0],
            sensor_001=[1.0, float("nan"), 3.0, float("nan")],
        )
        result = drop_high_missing(df, threshold=0.5)
        assert "sensor_001" in result.columns

    def test_keeps_column_below_threshold(self) -> None:
        df = _df(
            sensor_000=[1.0, 2.0, 3.0, 4.0, 5.0],
            sensor_001=[1.0, float("nan"), 3.0, 4.0, 5.0],
        )
        result = drop_high_missing(df, threshold=0.5)
        assert "sensor_001" in result.columns

    def test_preserves_label_and_timestamp(self) -> None:
        df = _df(sensor_000=[float("nan")] * 5)
        result = drop_high_missing(df, threshold=0.5)
        assert "label" in result.columns
        assert "timestamp" in result.columns


class TestImputeMedian:
    def test_no_nans_in_sensor_columns_after_imputation(self) -> None:
        df = _df(sensor_000=[1.0, float("nan"), 3.0], sensor_001=[2.0, 4.0, 6.0])
        result = impute_median(df)
        assert not result[["sensor_000", "sensor_001"]].isnull().any().any()

    def test_fills_with_column_median(self) -> None:
        # column values [1.0, nan, 3.0] → median of [1, 3] = 2.0
        df = _df(sensor_000=[1.0, float("nan"), 3.0])
        result = impute_median(df)
        assert result["sensor_000"].iloc[1] == pytest.approx(2.0)

    def test_does_not_alter_label_or_timestamp(self) -> None:
        df = _df(sensor_000=[1.0, float("nan"), 3.0])
        result = impute_median(df)
        pd.testing.assert_series_equal(result["label"], df["label"])
        pd.testing.assert_series_equal(result["timestamp"], df["timestamp"])


class TestDropLowVariance:
    def test_drops_constant_column(self) -> None:
        df = _df(
            sensor_000=[1.0, 2.0, 3.0, 4.0, 5.0],
            sensor_001=[5.0, 5.0, 5.0, 5.0, 5.0],
        )
        result = drop_low_variance(df, threshold=0.01)
        assert "sensor_001" not in result.columns

    def test_keeps_high_variance_column(self) -> None:
        df = _df(
            sensor_000=[1.0, 2.0, 3.0, 4.0, 5.0],
            sensor_001=[5.0, 5.0, 5.0, 5.0, 5.0],
        )
        result = drop_low_variance(df, threshold=0.01)
        assert "sensor_000" in result.columns

    def test_preserves_label_and_timestamp(self) -> None:
        df = _df(sensor_000=[1.0, 2.0, 3.0])
        result = drop_low_variance(df, threshold=0.0)
        assert "label" in result.columns
        assert "timestamp" in result.columns


class TestDropHighCorrelation:
    def test_drops_duplicate_column(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = _df(sensor_000=vals, sensor_001=vals)
        result = drop_high_correlation(df, threshold=0.9)
        assert "sensor_000" in result.columns
        assert "sensor_001" not in result.columns

    def test_keeps_uncorrelated_columns(self) -> None:
        df = _df(
            sensor_000=[1.0, 2.0, 3.0, 4.0, 5.0],
            sensor_001=[5.0, 1.0, 4.0, 2.0, 3.0],
        )
        result = drop_high_correlation(df, threshold=0.9)
        assert "sensor_000" in result.columns
        assert "sensor_001" in result.columns

    def test_preserves_label_and_timestamp(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = _df(sensor_000=vals, sensor_001=vals)
        result = drop_high_correlation(df, threshold=0.9)
        assert "label" in result.columns
        assert "timestamp" in result.columns


class TestSplitStratified:
    @pytest.fixture()
    def balanced_df(self) -> pd.DataFrame:
        """20 rows, 10 per class — clean 20 % test split = 4 rows."""
        return pd.DataFrame(
            {
                "sensor_000": np.arange(20, dtype=float),
                "label": [0] * 10 + [1] * 10,
                "timestamp": ["t"] * 20,
            }
        )

    def test_test_size_matches_fraction(self, balanced_df: pd.DataFrame) -> None:
        _, test = split_stratified(balanced_df, test_size=0.2, random_seed=0)
        assert len(test) == 4

    def test_train_size_is_remainder(self, balanced_df: pd.DataFrame) -> None:
        train, _ = split_stratified(balanced_df, test_size=0.2, random_seed=0)
        assert len(train) == 16

    def test_all_rows_present(self, balanced_df: pd.DataFrame) -> None:
        train, test = split_stratified(balanced_df, test_size=0.2, random_seed=0)
        assert len(train) + len(test) == len(balanced_df)

    def test_splits_are_disjoint(self, balanced_df: pd.DataFrame) -> None:
        train, test = split_stratified(balanced_df, test_size=0.2, random_seed=0)
        assert set(train.index).isdisjoint(set(test.index))

    def test_both_classes_in_test(self, balanced_df: pd.DataFrame) -> None:
        _, test = split_stratified(balanced_df, test_size=0.2, random_seed=0)
        assert set(test["label"].unique()) == {0, 1}
