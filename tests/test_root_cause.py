"""Tests for root-cause candidate ranking module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yield_risk.explainability import ShapExplanations
from yield_risk.root_cause import (
    fail_shap_lift,
    rank_root_cause_candidates,
    spc_flag_rate,
)


def _make_sensor_df(n_rows: int = 20, n_sensors: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        f"sensor_{i:03d}": rng.normal(i, 1, n_rows).tolist()
        for i in range(n_sensors)
    }
    data["label"] = ([0] * (n_rows - 4) + [1] * 4)
    data["timestamp"] = ["t"] * n_rows
    return pd.DataFrame(data)


def _make_explanations(n_samples: int = 20, n_features: int = 4) -> ShapExplanations:
    rng = np.random.default_rng(7)
    sv = rng.normal(size=(n_samples, n_features))
    features = [f"sensor_{i:03d}" for i in range(n_features)]
    return ShapExplanations(shap_values=sv, feature_names=features, base_value=-2.0)


class TestSpcFlagRate:
    def test_returns_series(self) -> None:
        df = _make_sensor_df()
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        result = spc_flag_rate(df, sensor_cols)
        assert isinstance(result, pd.Series)

    def test_index_is_sensor_names(self) -> None:
        df = _make_sensor_df()
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        result = spc_flag_rate(df, sensor_cols)
        assert set(result.index) == set(sensor_cols)

    def test_values_between_zero_and_one(self) -> None:
        df = _make_sensor_df()
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        result = spc_flag_rate(df, sensor_cols)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_constant_sensor_has_zero_flag_rate(self) -> None:
        df = pd.DataFrame({
            "sensor_000": [5.0] * 20,
            "label": [0] * 20,
            "timestamp": ["t"] * 20,
        })
        result = spc_flag_rate(df, ["sensor_000"])
        assert result["sensor_000"] == pytest.approx(0.0)

    def test_outlier_sensor_has_nonzero_flag_rate(self) -> None:
        # First value is 100σ away from the rest
        vals = [0.0] * 19 + [1000.0]
        df = pd.DataFrame({
            "sensor_000": vals,
            "label": [0] * 20,
            "timestamp": ["t"] * 20,
        })
        result = spc_flag_rate(df, ["sensor_000"])
        assert result["sensor_000"] > 0.0

    def test_reference_population_sets_control_limits(self) -> None:
        """Limits come from *reference*, not the batch being scored.

        A batch shifted far from a tight reference is flagged against the
        reference's narrow limits, whereas using the shifted batch as its own
        reference (the default) self-normalises and flags nothing.
        """
        reference = pd.DataFrame({"sensor_000": [0.0, 1.0, -1.0, 0.5, -0.5]})
        # Batch sits ~100 units away from the reference mean of 0.
        batch = pd.DataFrame({"sensor_000": [100.0, 101.0, 99.0, 100.5, 99.5]})

        against_reference = spc_flag_rate(
            batch, ["sensor_000"], reference=reference
        )
        self_normalised = spc_flag_rate(batch, ["sensor_000"])

        assert against_reference["sensor_000"] == pytest.approx(1.0)
        assert self_normalised["sensor_000"] == pytest.approx(0.0)


class TestFailShapLift:
    def test_returns_dataframe(self) -> None:
        exp = _make_explanations()
        y = np.array([0] * 16 + [1] * 4)
        result = fail_shap_lift(exp, y)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        exp = _make_explanations()
        y = np.array([0] * 16 + [1] * 4)
        result = fail_shap_lift(exp, y)
        for col in ("feature", "fail_mean_shap", "pass_mean_shap", "shap_lift"):
            assert col in result.columns

    def test_row_count_matches_features(self) -> None:
        exp = _make_explanations(n_features=4)
        y = np.array([0] * 16 + [1] * 4)
        result = fail_shap_lift(exp, y)
        assert len(result) == 4

    def test_shap_lift_is_difference(self) -> None:
        exp = _make_explanations()
        y = np.array([0] * 16 + [1] * 4)
        result = fail_shap_lift(exp, y)
        expected_lift = result["fail_mean_shap"] - result["pass_mean_shap"]
        pd.testing.assert_series_equal(
            result["shap_lift"].reset_index(drop=True),
            expected_lift.reset_index(drop=True),
            check_names=False,
        )

    def test_sorted_by_abs_lift_descending(self) -> None:
        exp = _make_explanations()
        y = np.array([0] * 16 + [1] * 4)
        result = fail_shap_lift(exp, y)
        abs_lifts = result["shap_lift"].abs().tolist()
        assert abs_lifts == sorted(abs_lifts, reverse=True)

    def test_all_fail_raises(self) -> None:
        exp = _make_explanations()
        y = np.ones(20, dtype=int)
        with pytest.raises(ValueError, match="no passing samples"):
            fail_shap_lift(exp, y)

    def test_all_pass_raises(self) -> None:
        exp = _make_explanations()
        y = np.zeros(20, dtype=int)
        with pytest.raises(ValueError, match="no failing samples"):
            fail_shap_lift(exp, y)


class TestRankRootCauseCandidates:
    @pytest.fixture()
    def inputs(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        features = [f"sensor_{i:03d}" for i in range(4)]
        global_imp = pd.DataFrame({
            "feature": features,
            "mean_abs_shap": [0.4, 0.3, 0.2, 0.1],
        })
        lift_df = pd.DataFrame({
            "feature": features,
            "fail_mean_shap": [0.5, -0.1, 0.2, 0.0],
            "pass_mean_shap": [0.1, 0.0, 0.1, 0.0],
            "shap_lift": [0.4, -0.1, 0.1, 0.0],
        })
        flag_rates = pd.Series(
            [0.1, 0.2, 0.05, 0.0],
            index=features,
        )
        return global_imp, lift_df, flag_rates

    def test_returns_dataframe(
        self, inputs: tuple[pd.DataFrame, pd.DataFrame, pd.Series]
    ) -> None:
        global_imp, lift_df, flag_rates = inputs
        result = rank_root_cause_candidates(global_imp, lift_df, flag_rates)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(
        self, inputs: tuple[pd.DataFrame, pd.DataFrame, pd.Series]
    ) -> None:
        global_imp, lift_df, flag_rates = inputs
        result = rank_root_cause_candidates(global_imp, lift_df, flag_rates)
        for col in (
            "sensor",
            "mean_abs_shap",
            "shap_lift",
            "spc_flag_rate",
            "composite_score",
        ):
            assert col in result.columns

    def test_sorted_by_composite_score_descending(
        self, inputs: tuple[pd.DataFrame, pd.DataFrame, pd.Series]
    ) -> None:
        global_imp, lift_df, flag_rates = inputs
        result = rank_root_cause_candidates(global_imp, lift_df, flag_rates)
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_composite_score_in_zero_one(
        self, inputs: tuple[pd.DataFrame, pd.DataFrame, pd.Series]
    ) -> None:
        global_imp, lift_df, flag_rates = inputs
        result = rank_root_cause_candidates(global_imp, lift_df, flag_rates)
        assert (result["composite_score"] >= 0).all()
        assert (result["composite_score"] <= 1).all()

    def test_row_count_matches_features(
        self, inputs: tuple[pd.DataFrame, pd.DataFrame, pd.Series]
    ) -> None:
        global_imp, lift_df, flag_rates = inputs
        result = rank_root_cause_candidates(global_imp, lift_df, flag_rates)
        assert len(result) == 4

    def test_sensor_missing_from_lift_df_gets_zero_lift(
        self, inputs: tuple[pd.DataFrame, pd.DataFrame, pd.Series]
    ) -> None:
        global_imp, lift_df, flag_rates = inputs
        # Remove one sensor from lift_df
        partial_lift = lift_df[lift_df["feature"] != "sensor_003"].copy()
        result = rank_root_cause_candidates(global_imp, partial_lift, flag_rates)
        # sensor_003 should have shap_lift=0.0, not NaN
        row = result[result["sensor"] == "sensor_003"]
        assert len(row) == 1
        assert not pd.isna(row["shap_lift"].iloc[0])
        assert row["shap_lift"].iloc[0] == pytest.approx(0.0)
        # composite score must also be non-NaN
        assert not pd.isna(row["composite_score"].iloc[0])
