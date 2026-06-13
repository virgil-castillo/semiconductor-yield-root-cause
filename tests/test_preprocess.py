"""Tests for the preprocessing pipeline (preprocess.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from yield_risk.config import RunConfig
from yield_risk.preprocess import (
    SecomPreprocessor,
    run_preprocessing,
    split_stratified,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sensor_df(**sensor_vals: list[float]) -> pd.DataFrame:
    """Build a minimal SECOM-schema sensor-only DataFrame (no label/timestamp)."""
    return pd.DataFrame(sensor_vals)


def _full_df(n: int = 10, n_sensors: int = 3) -> pd.DataFrame:
    """Return a SECOM-schema DataFrame with sensor cols, label, and timestamp."""
    rng = np.random.default_rng(0)
    data: dict[str, object] = {
        f"sensor_{i:03d}": rng.normal(0, 1, n).tolist()
        for i in range(n_sensors)
    }
    data["label"] = ([0, 1] * n)[:n]
    data["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="h").tolist()
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# SecomPreprocessor — missing filter
# ---------------------------------------------------------------------------


class TestSecomPreprocessorMissingFilter:
    def test_drops_column_above_missing_threshold(self) -> None:
        X = _sensor_df(
            s0=[1.0, 2.0, 3.0, 4.0, 5.0],
            s1=[float("nan")] * 4 + [1.0],  # 80 % missing > 0.5
        )
        pp = SecomPreprocessor(
            missing_threshold=0.5,
            variance_threshold=0.0,
            correlation_threshold=1.0,
        )
        pp.fit(X)
        out = pp.transform(X)
        assert "s1" not in out.columns

    def test_keeps_column_at_missing_threshold(self) -> None:
        # exactly 50 % missing — not strictly above, so kept
        X = _sensor_df(
            s0=[1.0, 2.0, 3.0, 4.0],
            s1=[1.0, float("nan"), 3.0, float("nan")],
        )
        pp = SecomPreprocessor(
            missing_threshold=0.5,
            variance_threshold=0.0,
            correlation_threshold=1.0,
        )
        pp.fit(X)
        out = pp.transform(X)
        assert "s1" in out.columns

    def test_all_nan_column_removed_and_no_nan_in_output(self) -> None:
        X = _sensor_df(
            s0=[1.0, 2.0, 3.0],
            s1=[float("nan"), float("nan"), float("nan")],
        )
        pp = SecomPreprocessor(
            missing_threshold=0.5,
            variance_threshold=0.0,
            correlation_threshold=1.0,
        )
        pp.fit(X)
        out = pp.transform(X)
        assert "s1" not in out.columns
        assert not out.isnull().any().any()


# ---------------------------------------------------------------------------
# SecomPreprocessor — variance filter
# ---------------------------------------------------------------------------


class TestSecomPreprocessorVarianceFilter:
    def test_drops_constant_column(self) -> None:
        X = _sensor_df(
            s0=[1.0, 2.0, 3.0, 4.0, 5.0],
            s1=[5.0, 5.0, 5.0, 5.0, 5.0],
        )
        pp = SecomPreprocessor(
            missing_threshold=1.0,
            variance_threshold=0.01,
            correlation_threshold=1.0,
        )
        pp.fit(X)
        out = pp.transform(X)
        assert "s1" not in out.columns

    def test_keeps_high_variance_column(self) -> None:
        X = _sensor_df(
            s0=[1.0, 2.0, 3.0, 4.0, 5.0],
            s1=[5.0, 5.0, 5.0, 5.0, 5.0],
        )
        pp = SecomPreprocessor(
            missing_threshold=1.0,
            variance_threshold=0.01,
            correlation_threshold=1.0,
        )
        pp.fit(X)
        out = pp.transform(X)
        assert "s0" in out.columns


# ---------------------------------------------------------------------------
# SecomPreprocessor — correlation filter
# ---------------------------------------------------------------------------


class TestSecomPreprocessorCorrelationFilter:
    def test_drops_duplicate_later_column(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        X = _sensor_df(s0=vals, s1=vals)
        pp = SecomPreprocessor(
            missing_threshold=1.0,
            variance_threshold=0.0,
            correlation_threshold=0.9,
        )
        pp.fit(X)
        out = pp.transform(X)
        assert "s0" in out.columns
        assert "s1" not in out.columns

    def test_keeps_uncorrelated_columns(self) -> None:
        X = _sensor_df(
            s0=[1.0, 2.0, 3.0, 4.0, 5.0],
            s1=[5.0, 1.0, 4.0, 2.0, 3.0],
        )
        pp = SecomPreprocessor(
            missing_threshold=1.0,
            variance_threshold=0.0,
            correlation_threshold=0.9,
        )
        pp.fit(X)
        out = pp.transform(X)
        assert "s0" in out.columns
        assert "s1" in out.columns


# ---------------------------------------------------------------------------
# SecomPreprocessor — filter order (missing → variance → correlation)
# ---------------------------------------------------------------------------


class TestSecomPreprocessorFilterOrder:
    def test_missing_then_variance_then_correlation_order(self) -> None:
        """Missing filter runs before variance so a mostly-NaN col is dropped.

        s0: 80 % missing — dropped by missing filter first.
        s1: constant — would be dropped by variance filter.
        s2: high variance, kept.
        """
        X = pd.DataFrame(
            {
                "s0": [float("nan")] * 8 + [1.0, 2.0],  # 80 % missing
                "s1": [5.0] * 10,  # constant (zero variance)
                "s2": list(range(10)),  # kept
            }
        )
        pp = SecomPreprocessor(
            missing_threshold=0.5,
            variance_threshold=0.01,
            correlation_threshold=1.0,
        )
        pp.fit(X)
        out = pp.transform(X)
        assert "s0" not in out.columns
        assert "s1" not in out.columns
        assert "s2" in out.columns


# ---------------------------------------------------------------------------
# SecomPreprocessor — median imputation (train medians used on transform)
# ---------------------------------------------------------------------------


class TestSecomPreprocessorImputation:
    def test_fills_nan_with_train_median(self) -> None:
        X_train = _sensor_df(s0=[1.0, float("nan"), 3.0])
        pp = SecomPreprocessor(
            missing_threshold=1.0,
            variance_threshold=0.0,
            correlation_threshold=1.0,
        )
        pp.fit(X_train)
        # train median of s0: median([1, 3]) = 2.0
        X_new = pd.DataFrame({"s0": [float("nan")]})
        out = pp.transform(X_new)
        assert out["s0"].iloc[0] == pytest.approx(2.0)

    def test_no_nans_after_transform(self) -> None:
        X = _sensor_df(s0=[1.0, float("nan"), 3.0], s1=[2.0, 4.0, 6.0])
        pp = SecomPreprocessor(
            missing_threshold=1.0,
            variance_threshold=0.0,
            correlation_threshold=1.0,
        )
        pp.fit(X)
        out = pp.transform(X)
        assert not out.isnull().any().any()


# ---------------------------------------------------------------------------
# SecomPreprocessor — no leakage (held-out stats must not influence output)
# ---------------------------------------------------------------------------


class TestSecomPreprocessorNoLeakage:
    def test_transform_uses_train_medians_not_holdout_medians(self) -> None:
        """A held-out value's imputed fill equals the TRAIN median.

        Train s0 = [10, 20, 30] → train median = 20.0
        Holdout s0 = [100, nan, 200] → holdout median would be 150.0.
        The NaN in holdout must be filled with 20.0, not 150.0.
        """
        X_train = pd.DataFrame({"s0": [10.0, 20.0, 30.0]})
        X_holdout = pd.DataFrame({"s0": [100.0, float("nan"), 200.0]})

        pp = SecomPreprocessor(
            missing_threshold=1.0,
            variance_threshold=0.0,
            correlation_threshold=1.0,
        )
        pp.fit(X_train)
        out = pp.transform(X_holdout)
        assert out["s0"].iloc[1] == pytest.approx(20.0)

    def test_transform_uses_train_selected_columns_only(self) -> None:
        """Columns kept come from fit; a column absent after fit is excluded."""
        # s1 is constant in train → dropped by variance filter
        X_train = pd.DataFrame(
            {
                "s0": [1.0, 2.0, 3.0, 4.0, 5.0],
                "s1": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        # In holdout s1 has real variance, but it was dropped at fit time
        X_holdout = pd.DataFrame(
            {
                "s0": [1.0, 2.0, 3.0],
                "s1": [1.0, 5.0, 9.0],
            }
        )
        pp = SecomPreprocessor(
            missing_threshold=1.0,
            variance_threshold=0.01,
            correlation_threshold=1.0,
        )
        pp.fit(X_train)
        out = pp.transform(X_holdout)
        assert "s1" not in out.columns


# ---------------------------------------------------------------------------
# SecomPreprocessor — get_feature_names_out
# ---------------------------------------------------------------------------


class TestSecomPreprocessorGetFeatureNamesOut:
    def test_returns_kept_column_names(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        X = _sensor_df(s0=vals, s1=vals)  # s1 duplicate → dropped
        pp = SecomPreprocessor(
            missing_threshold=1.0,
            variance_threshold=0.0,
            correlation_threshold=0.9,
        )
        pp.fit(X)
        names = pp.get_feature_names_out()
        assert list(names) == list(pp.transform(X).columns)

    def test_raises_not_fitted_error_before_fit(self) -> None:
        pp = SecomPreprocessor(
            missing_threshold=0.5,
            variance_threshold=0.0,
            correlation_threshold=1.0,
        )
        with pytest.raises(NotFittedError):
            pp.get_feature_names_out()

    def test_transform_raises_not_fitted_error_before_fit(self) -> None:
        pp = SecomPreprocessor(
            missing_threshold=0.5,
            variance_threshold=0.0,
            correlation_threshold=1.0,
        )
        X = _sensor_df(s0=[1.0, 2.0, 3.0])
        with pytest.raises(NotFittedError):
            pp.transform(X)

    def test_transform_returns_dataframe(self) -> None:
        X = _sensor_df(s0=[1.0, 2.0, 3.0], s1=[4.0, 5.0, 6.0])
        pp = SecomPreprocessor(
            missing_threshold=1.0,
            variance_threshold=0.0,
            correlation_threshold=1.0,
        )
        pp.fit(X)
        out = pp.transform(X)
        assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# split_stratified
# ---------------------------------------------------------------------------


def _stratified_df(n: int = 200, pos_rate: float = 0.13) -> pd.DataFrame:
    """Build a synthetic SECOM-schema DataFrame for stratification tests.

    Args:
        n: Total number of rows.
        pos_rate: Fraction of rows with label == 1.

    Returns:
        DataFrame with sensor_000, label, and timestamp columns.
    """
    rng = np.random.default_rng(99)
    n_pos = round(n * pos_rate)
    labels = [1] * n_pos + [0] * (n - n_pos)
    return pd.DataFrame(
        {
            "sensor_000": rng.normal(0, 1, n).tolist(),
            "label": labels,
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h").tolist(),
        }
    )


class TestSplitStratified:
    """Tests for split_stratified."""

    @pytest.fixture()
    def strat_df(self) -> pd.DataFrame:
        """200-row DataFrame with ~13 % positive rate."""
        return _stratified_df(n=200, pos_rate=0.13)

    def test_both_splits_contain_both_classes(
        self, strat_df: pd.DataFrame
    ) -> None:
        """Each split must have at least one positive and one negative."""
        train, test = split_stratified(strat_df, test_size=0.15, random_seed=0)
        assert set(train["label"].unique()) == {0, 1}
        assert set(test["label"].unique()) == {0, 1}

    def test_test_split_size_is_approximately_correct(
        self, strat_df: pd.DataFrame
    ) -> None:
        """With 200 rows and test_size=0.15, test should be 30 rows."""
        _, test = split_stratified(strat_df, test_size=0.15, random_seed=0)
        assert len(test) == 30

    def test_stratification_balance_train(self, strat_df: pd.DataFrame) -> None:
        """Train positive prevalence within ±0.03 of overall prevalence."""
        overall = strat_df["label"].mean()
        train, _ = split_stratified(strat_df, test_size=0.15, random_seed=0)
        assert abs(train["label"].mean() - overall) <= 0.03

    def test_stratification_balance_test(self, strat_df: pd.DataFrame) -> None:
        """Test positive prevalence within ±0.03 of overall prevalence."""
        overall = strat_df["label"].mean()
        _, test = split_stratified(strat_df, test_size=0.15, random_seed=0)
        assert abs(test["label"].mean() - overall) <= 0.03

    def test_all_rows_present(self, strat_df: pd.DataFrame) -> None:
        """Train + test must account for every row."""
        train, test = split_stratified(strat_df, test_size=0.15, random_seed=0)
        assert len(train) + len(test) == len(strat_df)

    def test_splits_are_disjoint(self, strat_df: pd.DataFrame) -> None:
        """Train and test index sets must not overlap."""
        train, test = split_stratified(strat_df, test_size=0.15, random_seed=0)
        assert set(train.index).isdisjoint(set(test.index))

    def test_same_seed_yields_identical_splits(
        self, strat_df: pd.DataFrame
    ) -> None:
        """Calling twice with the same seed must return the same partition."""
        train_a, test_a = split_stratified(strat_df, test_size=0.15, random_seed=7)
        train_b, test_b = split_stratified(strat_df, test_size=0.15, random_seed=7)
        assert list(train_a.index) == list(train_b.index)
        assert list(test_a.index) == list(test_b.index)

    def test_all_columns_preserved_in_splits(
        self, strat_df: pd.DataFrame
    ) -> None:
        """All columns of the input DataFrame are present in both splits."""
        train, test = split_stratified(strat_df, test_size=0.15, random_seed=0)
        assert list(train.columns) == list(strat_df.columns)
        assert list(test.columns) == list(strat_df.columns)


# ---------------------------------------------------------------------------
# run_preprocessing — raw splits contract
# ---------------------------------------------------------------------------


class TestRunPreprocessing:
    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 40
        vals = list(rng.normal(0, 1, n).tolist())
        # Interleave labels so both time-based splits have both classes.
        # Positives at positions 4, 12, 20, 28, 36 (every 8th row starting at 4).
        labels = [0] * n
        for i in range(4, n, 8):
            labels[i] = 1
        return pd.DataFrame(
            {
                "sensor_000": [5.0] * n,
                "sensor_001": [float("nan")] * 25
                + list(rng.normal(0, 1, 15).tolist()),
                "sensor_002": vals,
                "sensor_003": vals,
                "sensor_004": list(rng.normal(0, 1, n).tolist()),
                "label": labels,
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            }
        )

    @pytest.fixture()
    def run_cfg(self) -> RunConfig:
        return RunConfig(
            random_seed=42,
            test_size=0.20,
            cv_folds=5,
            missing_threshold=0.60,
            variance_threshold=0.01,
            correlation_threshold=0.95,
        )

    def test_returns_two_dataframes(
        self, sample_df: pd.DataFrame, run_cfg: RunConfig
    ) -> None:
        result = run_preprocessing(sample_df, run_cfg)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DataFrame)

    def test_output_has_label_column(
        self, sample_df: pd.DataFrame, run_cfg: RunConfig
    ) -> None:
        train, test = run_preprocessing(sample_df, run_cfg)
        assert "label" in train.columns
        assert "label" in test.columns
        assert set(train["label"].unique()).issubset({0, 1})
        assert set(test["label"].unique()).issubset({0, 1})

    def test_raw_splits_preserve_nans_in_sensor_columns(
        self, sample_df: pd.DataFrame, run_cfg: RunConfig
    ) -> None:
        """Raw splits must keep NaNs intact (no imputation in run_preprocessing)."""
        train, _ = run_preprocessing(sample_df, run_cfg)
        # sensor_001 has NaNs in the early rows; train is those early rows
        sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
        total_nans = train[sensor_cols].isnull().sum().sum()
        assert total_nans > 0, "Expected NaNs in raw train split"

    def test_raw_splits_retain_all_sensor_columns(
        self, sample_df: pd.DataFrame, run_cfg: RunConfig
    ) -> None:
        """Raw splits keep ALL sensor columns (no selection in run_preprocessing)."""
        train, test = run_preprocessing(sample_df, run_cfg)
        original_sensors = [c for c in sample_df.columns if c.startswith("sensor_")]
        for col in original_sensors:
            assert col in train.columns
            assert col in test.columns

    def test_splits_are_disjoint(
        self, sample_df: pd.DataFrame, run_cfg: RunConfig
    ) -> None:
        train, test = run_preprocessing(sample_df, run_cfg)
        assert set(train.index).isdisjoint(set(test.index))

    def test_fitted_preprocessor_output_has_no_nans(
        self, sample_df: pd.DataFrame, run_cfg: RunConfig
    ) -> None:
        """After fitting SecomPreprocessor on train split, transform has no NaNs."""
        train, _ = run_preprocessing(sample_df, run_cfg)
        sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
        X_train = train[sensor_cols]
        pp = SecomPreprocessor(
            missing_threshold=run_cfg.missing_threshold,
            variance_threshold=run_cfg.variance_threshold,
            correlation_threshold=run_cfg.correlation_threshold,
        )
        pp.fit(X_train)
        out = pp.transform(X_train)
        assert not out.isnull().any().any()

    def test_fitted_preprocessor_output_has_reduced_columns(
        self, sample_df: pd.DataFrame, run_cfg: RunConfig
    ) -> None:
        """After fitting SecomPreprocessor, only selected columns remain."""
        train, _ = run_preprocessing(sample_df, run_cfg)
        sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
        X_train = train[sensor_cols]
        pp = SecomPreprocessor(
            missing_threshold=run_cfg.missing_threshold,
            variance_threshold=run_cfg.variance_threshold,
            correlation_threshold=run_cfg.correlation_threshold,
        )
        pp.fit(X_train)
        out = pp.transform(X_train)
        assert len(out.columns) < len(sensor_cols)
