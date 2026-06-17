"""Tests for early_detection sensor access primitives."""

from __future__ import annotations

import importlib.util
import json
import math
import types
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from yield_risk.early_detection import (
    EarlyDetectionConfig,
    HyperparamConfig,
    SensorAccess,
    latest_index,
    observation_fraction,
    select_ordered_sensors,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RAW_COLS = [f"sensor_{i}" for i in range(10)]  # 10 sensors


def _make_df(cols: list[str]) -> pd.DataFrame:
    """Return a single-row DataFrame with the given columns."""
    return pd.DataFrame({c: [float(i)] for i, c in enumerate(cols)})


# ---------------------------------------------------------------------------
# latest_index
# ---------------------------------------------------------------------------


def test_latest_index_prefix() -> None:
    """latest_index for a prefix access equals prefix_end."""
    access = SensorAccess(access_type="prefix", prefix_end=4)
    assert latest_index(access, n_sensors=10) == 4


def test_latest_index_window() -> None:
    """latest_index for a window access equals window_start + window_size."""
    access = SensorAccess(access_type="window", window_start=2, window_size=3)
    assert latest_index(access, n_sensors=10) == 5


def test_latest_index_window_clipped() -> None:
    """latest_index for a clipped window is capped at n_sensors."""
    access = SensorAccess(access_type="window", window_start=8, window_size=5)
    assert latest_index(access, n_sensors=10) == 10


# ---------------------------------------------------------------------------
# observation_fraction
# ---------------------------------------------------------------------------


def test_observation_fraction_prefix() -> None:
    """observation_fraction for prefix is prefix_end / n_sensors."""
    access = SensorAccess(access_type="prefix", prefix_end=5)
    assert observation_fraction(access, n_sensors=10) == pytest.approx(0.5)


def test_observation_fraction_window() -> None:
    """observation_fraction for window is (window_start + window_size) / n_sensors."""
    access = SensorAccess(access_type="window", window_start=3, window_size=4)
    assert observation_fraction(access, n_sensors=10) == pytest.approx(0.7)


def test_observation_fraction_window_clipped() -> None:
    """observation_fraction for clipped window clips to 1.0."""
    access = SensorAccess(access_type="window", window_start=8, window_size=5)
    assert observation_fraction(access, n_sensors=10) == pytest.approx(1.0)


def test_observation_fraction_full_prefix() -> None:
    """observation_fraction of 1.0 is valid (boundary of (0,1] range)."""
    access = SensorAccess(access_type="prefix", prefix_end=10)
    assert observation_fraction(access, n_sensors=10) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# select_ordered_sensors — correct columns returned
# ---------------------------------------------------------------------------


def test_select_ordered_sensors_prefix_columns() -> None:
    """select_ordered_sensors with prefix returns first prefix_end columns."""
    df = _make_df(RAW_COLS)
    access = SensorAccess(access_type="prefix", prefix_end=3)
    result_df, window_cols = select_ordered_sensors(df, RAW_COLS, access)

    assert window_cols == RAW_COLS[:3]
    assert list(result_df.columns) == RAW_COLS[:3]


def test_select_ordered_sensors_window_columns() -> None:
    """select_ordered_sensors with window returns the correct slice."""
    df = _make_df(RAW_COLS)
    access = SensorAccess(access_type="window", window_start=2, window_size=4)
    result_df, window_cols = select_ordered_sensors(df, RAW_COLS, access)

    expected = RAW_COLS[2:6]
    assert window_cols == expected
    assert list(result_df.columns) == expected


def test_select_ordered_sensors_window_clipped_columns() -> None:
    """select_ordered_sensors clips the window end to n_sensors."""
    df = _make_df(RAW_COLS)
    access = SensorAccess(access_type="window", window_start=7, window_size=6)
    result_df, window_cols = select_ordered_sensors(df, RAW_COLS, access)

    expected = RAW_COLS[7:]  # sensors 7, 8, 9
    assert window_cols == expected
    assert list(result_df.columns) == expected


def test_select_ordered_sensors_preserves_names() -> None:
    """Column names in returned DataFrame match original sensor names exactly."""
    df = _make_df(RAW_COLS)
    access = SensorAccess(access_type="prefix", prefix_end=5)
    result_df, window_cols = select_ordered_sensors(df, RAW_COLS, access)

    assert all(col in RAW_COLS for col in result_df.columns)
    assert all(col in RAW_COLS for col in window_cols)


def test_select_ordered_sensors_returns_dataframe() -> None:
    """select_ordered_sensors returns a DataFrame, not a Series or other type."""
    df = _make_df(RAW_COLS)
    access = SensorAccess(access_type="prefix", prefix_end=2)
    result_df, _ = select_ordered_sensors(df, RAW_COLS, access)

    assert isinstance(result_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Validation: SensorAccess construction errors
# ---------------------------------------------------------------------------


def test_sensor_access_invalid_access_type() -> None:
    """SensorAccess raises ValueError for unknown access_type."""
    with pytest.raises(ValueError, match="access_type"):
        SensorAccess(access_type="sliding", prefix_end=3)


def test_sensor_access_prefix_end_less_than_1() -> None:
    """SensorAccess raises ValueError when prefix_end < 1."""
    with pytest.raises(ValueError, match="prefix_end"):
        SensorAccess(access_type="prefix", prefix_end=0)


def test_sensor_access_window_size_less_than_1() -> None:
    """SensorAccess raises ValueError when window_size < 1."""
    with pytest.raises(ValueError, match="window_size"):
        SensorAccess(access_type="window", window_start=0, window_size=0)


def test_sensor_access_window_start_negative() -> None:
    """SensorAccess raises ValueError when window_start < 0."""
    with pytest.raises(ValueError, match="window_start"):
        SensorAccess(access_type="window", window_start=-1, window_size=2)


def test_sensor_access_prefix_missing_prefix_end() -> None:
    """SensorAccess raises ValueError when prefix access_type has no prefix_end."""
    with pytest.raises(ValueError, match="prefix_end"):
        SensorAccess(access_type="prefix")


def test_sensor_access_window_missing_fields() -> None:
    """SensorAccess raises ValueError when window access_type lacks window fields."""
    with pytest.raises(ValueError, match="window_start"):
        SensorAccess(access_type="window", window_size=3)


def test_sensor_access_prefix_with_window_fields() -> None:
    """SensorAccess raises ValueError when prefix access has window fields set."""
    with pytest.raises(ValueError, match="window_start"):
        SensorAccess(
            access_type="prefix", prefix_end=3, window_start=0, window_size=3
        )


def test_sensor_access_window_with_prefix_field() -> None:
    """SensorAccess raises ValueError when window access has prefix_end set."""
    with pytest.raises(ValueError, match="prefix_end"):
        SensorAccess(
            access_type="window", prefix_end=3, window_start=0, window_size=3
        )


# ---------------------------------------------------------------------------
# Validation: select_ordered_sensors runtime errors
# ---------------------------------------------------------------------------


def test_select_ordered_sensors_empty_raw_cols() -> None:
    """select_ordered_sensors raises ValueError when raw_sensor_cols is empty."""
    df = _make_df(RAW_COLS)
    access = SensorAccess(access_type="prefix", prefix_end=3)
    with pytest.raises(ValueError, match="No raw sensor_ columns found"):
        select_ordered_sensors(df, [], access)


def test_select_ordered_sensors_window_start_out_of_range() -> None:
    """select_ordered_sensors raises ValueError when window_start >= n_sensors."""
    df = _make_df(RAW_COLS)
    access = SensorAccess(access_type="window", window_start=10, window_size=2)
    with pytest.raises(ValueError, match="window_start"):
        select_ordered_sensors(df, RAW_COLS, access)


# ---------------------------------------------------------------------------
# HyperparamConfig dataclass
# ---------------------------------------------------------------------------


def test_hyperparam_config_is_frozen_dataclass() -> None:
    """HyperparamConfig is a frozen dataclass — fields are immutable."""
    from yield_risk.early_detection import HyperparamConfig

    access = SensorAccess(access_type="prefix", prefix_end=5)
    cfg = HyperparamConfig(
        access=access,
        missing_threshold=0.5,
        cv_threshold=0.01,
        correlation_threshold=0.95,
        selection_method="none",
        max_features=None,
        model_family="random_forest",
        model_params={},
        threshold_policy="tune",
        threshold=0.5,
        false_alarm_rate=None,
    )
    with pytest.raises(Exception):
        cfg.model_family = "xgboost"  # type: ignore[misc]


def test_hyperparam_config_stores_all_fields() -> None:
    """HyperparamConfig stores all provided field values correctly."""
    from yield_risk.early_detection import HyperparamConfig

    access = SensorAccess(access_type="window", window_start=2, window_size=4)
    cfg = HyperparamConfig(
        access=access,
        missing_threshold=0.3,
        cv_threshold=0.005,
        correlation_threshold=0.9,
        selection_method="univariate",
        max_features=10,
        model_family="xgboost",
        model_params={"scale_pos_weight": 5.0},
        threshold_policy="far_constraint",
        threshold=None,
        false_alarm_rate=0.05,
    )
    assert cfg.access is access
    assert cfg.missing_threshold == 0.3
    assert cfg.cv_threshold == 0.005
    assert cfg.correlation_threshold == 0.9
    assert cfg.selection_method == "univariate"
    assert cfg.max_features == 10
    assert cfg.model_family == "xgboost"
    assert cfg.model_params == {"scale_pos_weight": 5.0}
    assert cfg.threshold_policy == "far_constraint"
    assert cfg.threshold is None
    assert cfg.false_alarm_rate == 0.05


# ---------------------------------------------------------------------------
# build_estimator — helper toy data
# ---------------------------------------------------------------------------

def _toy_xy(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return a small 2-class toy dataset that is clearly separable."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=-3.0, scale=0.5, size=(30, 4))
    X1 = rng.normal(loc=3.0, scale=0.5, size=(30, 4))
    X = np.vstack([X0, X1])
    y = np.array([0] * 30 + [1] * 30)
    return X, y


# ---------------------------------------------------------------------------
# build_estimator — fit + predict_proba shape for each family
# ---------------------------------------------------------------------------


def test_build_estimator_logistic_regression_shape() -> None:
    """build_estimator logistic_regression: predict_proba yields shape (n, 2)."""
    from yield_risk.early_detection import build_estimator

    X, y = _toy_xy()
    est = build_estimator("logistic_regression", {}, random_seed=0)
    assert callable(est.fit)
    assert callable(est.predict_proba)
    est.fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (len(y), 2)


def test_build_estimator_random_forest_shape() -> None:
    """build_estimator random_forest: predict_proba yields shape (n, 2)."""
    from yield_risk.early_detection import build_estimator

    X, y = _toy_xy()
    est = build_estimator("random_forest", {}, random_seed=0)
    assert callable(est.fit)
    assert callable(est.predict_proba)
    est.fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (len(y), 2)


def test_build_estimator_xgboost_shape() -> None:
    """build_estimator xgboost: predict_proba yields shape (n, 2)."""
    from yield_risk.early_detection import build_estimator

    X, y = _toy_xy()
    est = build_estimator("xgboost", {}, random_seed=0)
    assert callable(est.fit)
    assert callable(est.predict_proba)
    est.fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (len(y), 2)


def test_build_estimator_lightgbm_shape() -> None:
    """build_estimator lightgbm: predict_proba yields shape (n, 2)."""
    from yield_risk.early_detection import build_estimator

    X, y = _toy_xy()
    est = build_estimator("lightgbm", {}, random_seed=0)
    assert callable(est.fit)
    assert callable(est.predict_proba)
    est.fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (len(y), 2)


# ---------------------------------------------------------------------------
# build_estimator — seeding determinism
# ---------------------------------------------------------------------------


def test_build_estimator_logistic_regression_determinism() -> None:
    """Two logistic_regression estimators with the same seed produce identical proba."""
    from yield_risk.early_detection import build_estimator

    X, y = _toy_xy()
    est1 = build_estimator("logistic_regression", {}, random_seed=42)
    est2 = build_estimator("logistic_regression", {}, random_seed=42)
    est1.fit(X, y)
    est2.fit(X, y)
    np.testing.assert_array_equal(est1.predict_proba(X), est2.predict_proba(X))


def test_build_estimator_random_forest_determinism() -> None:
    """Two random_forest estimators with the same seed produce identical proba."""
    from yield_risk.early_detection import build_estimator

    X, y = _toy_xy()
    est1 = build_estimator("random_forest", {}, random_seed=42)
    est2 = build_estimator("random_forest", {}, random_seed=42)
    est1.fit(X, y)
    est2.fit(X, y)
    np.testing.assert_array_equal(est1.predict_proba(X), est2.predict_proba(X))


def test_build_estimator_xgboost_determinism() -> None:
    """Two xgboost estimators with the same seed produce identical proba."""
    from yield_risk.early_detection import build_estimator

    X, y = _toy_xy()
    est1 = build_estimator("xgboost", {}, random_seed=42)
    est2 = build_estimator("xgboost", {}, random_seed=42)
    est1.fit(X, y)
    est2.fit(X, y)
    np.testing.assert_array_equal(est1.predict_proba(X), est2.predict_proba(X))


def test_build_estimator_lightgbm_determinism() -> None:
    """Two lightgbm estimators with the same seed produce identical proba."""
    from yield_risk.early_detection import build_estimator

    X, y = _toy_xy()
    est1 = build_estimator("lightgbm", {}, random_seed=42)
    est2 = build_estimator("lightgbm", {}, random_seed=42)
    est1.fit(X, y)
    est2.fit(X, y)
    np.testing.assert_array_equal(est1.predict_proba(X), est2.predict_proba(X))


# ---------------------------------------------------------------------------
# build_estimator — model_params overlay
# ---------------------------------------------------------------------------


def test_build_estimator_model_params_overlay_random_forest() -> None:
    """model_params overlay: random_forest respects n_estimators from params."""
    from yield_risk.early_detection import build_estimator

    est = build_estimator("random_forest", {"n_estimators": 7}, random_seed=0)
    assert est.get_params()["n_estimators"] == 7


# ---------------------------------------------------------------------------
# build_estimator — scale_pos_weight passthrough for xgboost
# ---------------------------------------------------------------------------


def test_build_estimator_xgboost_scale_pos_weight_passthrough() -> None:
    """scale_pos_weight in model_params is honored by xgboost estimator."""
    from yield_risk.early_detection import build_estimator

    est = build_estimator("xgboost", {"scale_pos_weight": 3.0}, random_seed=0)
    assert est.get_params()["scale_pos_weight"] == 3.0


# ---------------------------------------------------------------------------
# build_estimator — unknown family raises ValueError
# ---------------------------------------------------------------------------


def test_build_estimator_unknown_family_raises_value_error() -> None:
    """build_estimator raises ValueError for an unknown model_family."""
    from yield_risk.early_detection import build_estimator

    with pytest.raises(ValueError, match="model_family"):
        build_estimator("neural_net", {}, random_seed=0)


# ---------------------------------------------------------------------------
# FoldPreprocessor helpers
# ---------------------------------------------------------------------------


def _make_fold_cfg(
    selection_method: str = "none",
    max_features: int | None = None,
    missing_threshold: float = 0.5,
    cv_threshold: float = 0.0,
    correlation_threshold: float = 1.0,
    model_family: str = "random_forest",
) -> HyperparamConfig:
    """Return a HyperparamConfig for FoldPreprocessor tests."""
    return HyperparamConfig(
        access=SensorAccess(access_type="prefix", prefix_end=1),
        missing_threshold=missing_threshold,
        cv_threshold=cv_threshold,
        correlation_threshold=correlation_threshold,
        selection_method=selection_method,
        max_features=max_features,
        model_family=model_family,
        model_params={},
        threshold_policy="tune",
        threshold=0.5,
        false_alarm_rate=None,
    )


def _make_sensor_df(
    n_rows: int = 40,
    n_cols: int = 6,
    seed: int = 7,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a small synthetic sensor DataFrame and binary labels.

    Columns are named sensor_0 .. sensor_{n_cols-1}.
    """
    rng = np.random.default_rng(seed)
    data = {f"sensor_{i}": rng.standard_normal(n_rows) for i in range(n_cols)}
    df = pd.DataFrame(data)
    y = (rng.random(n_rows) > 0.5).astype(int)
    return df, y


# ---------------------------------------------------------------------------
# FoldPreprocessor — test 1: leakage boundary
# ---------------------------------------------------------------------------


def test_fold_preprocessor_fit_uses_only_train_rows() -> None:
    """Fit statistics are identical whether val rows are present in the DataFrame.

    This is the headline leakage test: fitting on x_full.iloc[train_idx] must
    yield byte-identical statistics to fitting on x_train_only (the same rows
    extracted to their own DataFrame).
    """
    from yield_risk.early_detection import FoldPreprocessor

    df_full, y_full = _make_sensor_df(n_rows=40, n_cols=6, seed=7)
    train_idx = list(range(30))
    y_train = y_full[train_idx]

    cfg = _make_fold_cfg(selection_method="univariate", max_features=4)

    # Fit on a slice of the full df (val rows still exist in df_full)
    fp_a = FoldPreprocessor.fit(df_full.iloc[train_idx], y_train, cfg, random_seed=0)

    # Fit on an independent DataFrame holding exactly the same rows
    x_train_only = df_full.iloc[train_idx].reset_index(drop=True)
    fp_b = FoldPreprocessor.fit(x_train_only, y_train, cfg, random_seed=0)

    assert fp_a.retained_cols == fp_b.retained_cols
    assert fp_a.medians == fp_b.medians
    np.testing.assert_array_equal(fp_a.scaler_mean, fp_b.scaler_mean)
    np.testing.assert_array_equal(fp_a.scaler_scale, fp_b.scaler_scale)
    np.testing.assert_array_equal(fp_a.selected_idx, fp_b.selected_idx)


# ---------------------------------------------------------------------------
# FoldPreprocessor — test 2: constant-column scale → 1.0
# ---------------------------------------------------------------------------


def test_fold_preprocessor_constant_sensor_scale_is_one() -> None:
    """A constant column on the train fold gets scaler_scale entry of 1.0.

    Transforming that column must produce finite values (no inf/nan).
    """
    from yield_risk.early_detection import FoldPreprocessor

    rng = np.random.default_rng(99)
    n = 20
    data = {
        "sensor_0": rng.standard_normal(n),
        "sensor_1": np.full(n, 3.14),  # constant non-zero signal
        "sensor_2": rng.standard_normal(n),
    }
    df = pd.DataFrame(data)
    y = (rng.random(n) > 0.5).astype(int)

    cfg = _make_fold_cfg(selection_method="none", cv_threshold=0.0)
    fp = FoldPreprocessor.fit(df, y, cfg, random_seed=0)

    # sensor_1 should survive (CV == 0.0 is NOT strictly below 0.0)
    assert "sensor_1" in fp.retained_cols
    idx_const = fp.retained_cols.index("sensor_1")
    assert fp.scaler_scale[idx_const] == pytest.approx(1.0)

    out = fp.transform(df)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# FoldPreprocessor — test 3: selector methods
# ---------------------------------------------------------------------------


def test_fold_preprocessor_selector_none_keeps_all() -> None:
    """selection_method='none' → selected_idx == arange(n_retained)."""
    from yield_risk.early_detection import FoldPreprocessor

    df, y = _make_sensor_df(n_rows=40, n_cols=6)
    cfg = _make_fold_cfg(selection_method="none")
    fp = FoldPreprocessor.fit(df, y, cfg, random_seed=0)

    np.testing.assert_array_equal(
        fp.selected_idx, np.arange(len(fp.retained_cols))
    )


def test_fold_preprocessor_selector_univariate_respects_max_features() -> None:
    """selection_method='univariate' with max_features=k keeps min(k, n_retained)."""
    from yield_risk.early_detection import FoldPreprocessor

    df, y = _make_sensor_df(n_rows=40, n_cols=6)
    k = 3
    cfg = _make_fold_cfg(selection_method="univariate", max_features=k)
    fp = FoldPreprocessor.fit(df, y, cfg, random_seed=0)

    expected = min(k, len(fp.retained_cols))
    assert len(fp.selected_idx) == expected


def test_fold_preprocessor_selector_mutual_info_respects_max_features() -> None:
    """selection_method='mutual_info' with max_features=k keeps min(k, n_retained)."""
    from yield_risk.early_detection import FoldPreprocessor

    df, y = _make_sensor_df(n_rows=40, n_cols=6)
    k = 2
    cfg = _make_fold_cfg(selection_method="mutual_info", max_features=k)
    fp = FoldPreprocessor.fit(df, y, cfg, random_seed=0)

    expected = min(k, len(fp.retained_cols))
    assert len(fp.selected_idx) == expected


def test_fold_preprocessor_selector_model_importance_bounded() -> None:
    """selection_method='model_importance' keeps ≤ max_features and ≤ n_retained."""
    from yield_risk.early_detection import FoldPreprocessor

    df, y = _make_sensor_df(n_rows=40, n_cols=6)
    max_f = 4
    cfg = _make_fold_cfg(
        selection_method="model_importance",
        max_features=max_f,
        model_family="random_forest",
    )
    fp = FoldPreprocessor.fit(df, y, cfg, random_seed=0)

    assert len(fp.selected_idx) <= max_f
    assert len(fp.selected_idx) <= len(fp.retained_cols)


# ---------------------------------------------------------------------------
# FoldPreprocessor — test 4: max_features > retained count → no raise
# ---------------------------------------------------------------------------


def test_fold_preprocessor_max_features_exceeds_retained_no_raise() -> None:
    """max_features larger than n_retained does not raise; k is clamped."""
    from yield_risk.early_detection import FoldPreprocessor

    df, y = _make_sensor_df(n_rows=40, n_cols=4)
    # max_features deliberately larger than n_cols
    cfg = _make_fold_cfg(selection_method="univariate", max_features=999)
    fp = FoldPreprocessor.fit(df, y, cfg, random_seed=0)  # must not raise

    assert len(fp.selected_idx) == len(fp.retained_cols)


# ---------------------------------------------------------------------------
# FoldPreprocessor — test 5: transform uses saved stats
# ---------------------------------------------------------------------------


def test_fold_preprocessor_transform_shape_and_dtype() -> None:
    """transform returns float64 array with shape (n_val, n_selected)."""
    from yield_risk.early_detection import FoldPreprocessor

    df, y = _make_sensor_df(n_rows=40, n_cols=6)
    train_idx = list(range(30))
    val_idx = list(range(30, 40))

    cfg = _make_fold_cfg(selection_method="univariate", max_features=3)
    fp = FoldPreprocessor.fit(df.iloc[train_idx], y[train_idx], cfg, random_seed=0)

    out = fp.transform(df.iloc[val_idx])
    assert out.shape == (len(val_idx), len(fp.selected_idx))
    assert out.dtype == np.float64


def test_fold_preprocessor_transform_uses_train_median_for_nan() -> None:
    """A NaN in a val row retained column is filled with the TRAIN-fold median."""
    from yield_risk.early_detection import FoldPreprocessor

    rng = np.random.default_rng(42)
    n = 30
    sensor_vals = rng.standard_normal(n)
    df = pd.DataFrame({"sensor_0": sensor_vals})
    y = (rng.random(n) > 0.5).astype(int)

    train_idx = list(range(20))
    val_idx = [20]

    cfg = _make_fold_cfg(selection_method="none")
    fp = FoldPreprocessor.fit(df.iloc[train_idx], y[train_idx], cfg, random_seed=0)

    train_median = float(np.median(sensor_vals[:20]))
    assert fp.medians["sensor_0"] == pytest.approx(train_median)

    # Build a val slice where sensor_0 is NaN
    val_df = df.iloc[val_idx].copy()
    val_df.loc[val_df.index[0], "sensor_0"] = np.nan

    out = fp.transform(val_df)
    # The value should have been filled with the train median, then scaled
    expected_raw = (train_median - fp.scaler_mean[0]) / fp.scaler_scale[0]
    assert out[0, 0] == pytest.approx(expected_raw)


# ---------------------------------------------------------------------------
# FoldPreprocessor — test 6: zero columns survive → no crash
# ---------------------------------------------------------------------------


def test_fold_preprocessor_all_columns_dropped_no_crash() -> None:
    """When all columns are dropped (missing_threshold=0), fit does not crash."""
    from yield_risk.early_detection import FoldPreprocessor

    rng = np.random.default_rng(11)
    n = 10
    # All columns have 100% missing → dropped by missing_threshold=0 (strictly > 0)
    df = pd.DataFrame(
        {
            "sensor_0": np.full(n, np.nan),
            "sensor_1": np.full(n, np.nan),
        }
    )
    y = (rng.random(n) > 0.5).astype(int)

    cfg = _make_fold_cfg(selection_method="none", missing_threshold=0.0)
    fp = FoldPreprocessor.fit(df, y, cfg, random_seed=0)

    assert fp.retained_cols == []
    assert fp.medians == {}

    out = fp.transform(df)
    assert out.shape == (n, 0)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# resolve_threshold + compute_detection_metric
# ---------------------------------------------------------------------------


def _make_tune_cfg(threshold: float) -> HyperparamConfig:
    """Return a HyperparamConfig with threshold_policy='tune'."""
    return HyperparamConfig(
        access=SensorAccess(access_type="prefix", prefix_end=1),
        missing_threshold=0.5,
        cv_threshold=0.0,
        correlation_threshold=1.0,
        selection_method="none",
        max_features=None,
        model_family="random_forest",
        model_params={},
        threshold_policy="tune",
        threshold=threshold,
        false_alarm_rate=None,
    )


def _make_far_cfg(false_alarm_rate: float) -> HyperparamConfig:
    """Return a HyperparamConfig with threshold_policy='far_constraint'."""
    return HyperparamConfig(
        access=SensorAccess(access_type="prefix", prefix_end=1),
        missing_threshold=0.5,
        cv_threshold=0.0,
        correlation_threshold=1.0,
        selection_method="none",
        max_features=None,
        model_family="random_forest",
        model_params={},
        threshold_policy="far_constraint",
        threshold=None,
        false_alarm_rate=false_alarm_rate,
    )


# ---------------------------------------------------------------------------
# resolve_threshold — "tune" policy
# ---------------------------------------------------------------------------


def test_resolve_threshold_tune_returns_cfg_threshold() -> None:
    """resolve_threshold with 'tune' policy returns cfg.threshold exactly."""
    from yield_risk.early_detection import resolve_threshold

    cfg = _make_tune_cfg(threshold=0.37)
    y_val = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.2, 0.9])
    result = resolve_threshold(y_val, y_prob, cfg)
    assert result == 0.37


# ---------------------------------------------------------------------------
# resolve_threshold — "far_constraint" policy
# ---------------------------------------------------------------------------


def test_resolve_threshold_far_constraint_selects_lowest_valid_threshold() -> None:
    """resolve_threshold 'far_constraint' picks lowest threshold with FAR <= target.

    Setup:
      y_true: [0, 0, 0, 0, 1, 1]  → 4 true negatives, 2 positives
      y_prob: [0.1, 0.2, 0.4, 0.6, 0.7, 0.9]

    Candidate thresholds (unique y_prob values): 0.1, 0.2, 0.4, 0.6, 0.7, 0.9

    At threshold t, positives = samples with y_prob >= t.
    FAR = FP / (FP + TN) = (negatives predicted positive) / 4

    threshold=0.1: all 4 negatives predicted positive → FAR = 4/4 = 1.0
    threshold=0.2: negatives [0.2, 0.4, 0.6] predicted → FAR = 3/4 = 0.75
    threshold=0.4: negatives [0.4, 0.6] predicted    → FAR = 2/4 = 0.50
    threshold=0.6: negatives [0.6] predicted         → FAR = 1/4 = 0.25
    threshold=0.7: no negatives predicted            → FAR = 0/4 = 0.0
    threshold=0.9: no negatives predicted            → FAR = 0/4 = 0.0

    With target FAR=0.25: thresholds satisfying FAR<=0.25 are 0.6, 0.7, 0.9.
    Lowest (most recall) = 0.6.
    """
    from yield_risk.early_detection import resolve_threshold

    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.7, 0.9])
    cfg = _make_far_cfg(false_alarm_rate=0.25)
    result = resolve_threshold(y_true, y_prob, cfg)
    assert result == pytest.approx(0.6)

    # Verify: threshold=0.6 satisfies FAR<=0.25
    neg_mask = y_true == 0
    far_at_result = float(np.sum((y_prob >= result) & neg_mask)) / float(
        np.sum(neg_mask)
    )
    assert far_at_result <= 0.25

    # Verify: lowering to next candidate (0.4) violates the cap
    far_at_lower = float(np.sum((y_prob >= 0.4) & neg_mask)) / float(
        np.sum(neg_mask)
    )
    assert far_at_lower > 0.25


def test_resolve_threshold_far_constraint_no_qualifying_threshold_returns_one() -> None:
    """resolve_threshold 'far_constraint' returns 1.0 when no threshold qualifies.

    With target FAR=0.0 but every threshold predicts at least one FP:
      y_true: [0, 0, 1]
      y_prob: [0.3, 0.5, 0.9]

    threshold=0.3: FAR = 2/2 = 1.0 (both negatives predicted)
    threshold=0.5: FAR = 1/2 = 0.5 (one negative predicted)
    threshold=0.9: FAR = 0/2 = 0.0 (no negative predicted) → qualifies!

    Use target FAR=-0.01 so that no threshold can satisfy FAR <= -0.01.
    """
    from yield_risk.early_detection import resolve_threshold

    y_true = np.array([0, 0, 1])
    y_prob = np.array([0.3, 0.5, 0.9])
    # FAR is always >= 0, so target=-0.01 guarantees no threshold qualifies
    cfg = _make_far_cfg(false_alarm_rate=-0.01)
    result = resolve_threshold(y_true, y_prob, cfg)
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_detection_metric — pr_auc
# ---------------------------------------------------------------------------


def test_compute_detection_metric_pr_auc_matches_average_precision() -> None:
    """compute_detection_metric 'pr_auc' equals average_precision_score."""
    from sklearn.metrics import average_precision_score

    from yield_risk.early_detection import compute_detection_metric

    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.4, 0.9])
    expected = average_precision_score(y_true, y_prob)
    result = compute_detection_metric(
        y_true, y_prob, threshold=0.5, metric_name="pr_auc"
    )
    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_detection_metric — roc_auc
# ---------------------------------------------------------------------------


def test_compute_detection_metric_roc_auc_matches_roc_auc_score() -> None:
    """compute_detection_metric 'roc_auc' equals roc_auc_score."""
    from sklearn.metrics import roc_auc_score

    from yield_risk.early_detection import compute_detection_metric

    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.4, 0.9])
    expected = roc_auc_score(y_true, y_prob)
    result = compute_detection_metric(
        y_true, y_prob, threshold=0.5, metric_name="roc_auc"
    )
    assert result == pytest.approx(expected)


def test_compute_detection_metric_roc_auc_delegates_to_compute_metrics() -> None:
    """roc_auc branch value equals compute_metrics(y_true, y_prob).roc_auc."""
    from yield_risk.early_detection import compute_detection_metric
    from yield_risk.evaluate import compute_metrics

    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.4, 0.9])
    expected = compute_metrics(y_true, y_prob).roc_auc
    result = compute_detection_metric(
        y_true, y_prob, threshold=0.5, metric_name="roc_auc"
    )
    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_detection_metric — recall_at_far
# ---------------------------------------------------------------------------


def test_compute_detection_metric_recall_at_far_matches_manual() -> None:
    """compute_detection_metric 'recall_at_far' equals manual recall calculation."""
    from yield_risk.early_detection import compute_detection_metric

    # y_true: 3 positives at indices 2,3,5 with y_prob 0.6, 0.8, 0.9
    # threshold=0.7 → predict positive when y_prob >= 0.7 → indices 3 and 5
    # TP=2, FN=1 → recall = 2/3
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.4, 0.9])
    result = compute_detection_metric(
        y_true, y_prob, threshold=0.7, metric_name="recall_at_far"
    )
    assert result == pytest.approx(2.0 / 3.0)


def test_compute_detection_metric_recall_at_far_no_positives_returns_nan() -> None:
    """recall_at_far returns nan when no positives in y_true."""
    from yield_risk.early_detection import compute_detection_metric

    y_true = np.array([0, 0, 0])
    y_prob = np.array([0.1, 0.5, 0.9])
    result = compute_detection_metric(
        y_true, y_prob, threshold=0.5, metric_name="recall_at_far"
    )
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# compute_detection_metric — neg_balanced_error
# ---------------------------------------------------------------------------


def test_compute_detection_metric_neg_balanced_error_matches_manual() -> None:
    """compute_detection_metric 'neg_balanced_error' equals balanced_accuracy-1."""
    from sklearn.metrics import balanced_accuracy_score

    from yield_risk.early_detection import compute_detection_metric

    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.4, 0.9])
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)
    expected = balanced_accuracy_score(y_true, y_pred) - 1.0
    result = compute_detection_metric(
        y_true, y_prob, threshold=threshold, metric_name="neg_balanced_error"
    )
    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_detection_metric — neg_expected_cost
# ---------------------------------------------------------------------------


def test_compute_detection_metric_neg_expected_cost_matches_thresholding() -> None:
    """neg_expected_cost equals -expected_cost_at_threshold."""
    from yield_risk.config import CostMatrix
    from yield_risk.early_detection import compute_detection_metric
    from yield_risk.thresholding import expected_cost_at_threshold

    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.4, 0.9])
    threshold = 0.5
    cm = CostMatrix(true_pass=0.0, true_fail=0.0, false_fail=1.0, false_pass=5.0)
    expected = -expected_cost_at_threshold(y_true, y_prob, threshold, cm)
    result = compute_detection_metric(
        y_true, y_prob, threshold=threshold, metric_name="neg_expected_cost",
        cost_matrix=cm,
    )
    assert result == pytest.approx(expected)


def test_compute_detection_metric_neg_expected_cost_none_matrix_raises() -> None:
    """neg_expected_cost raises ValueError when cost_matrix is None."""
    from yield_risk.early_detection import compute_detection_metric

    y_true = np.array([0, 1])
    y_prob = np.array([0.3, 0.8])
    with pytest.raises(ValueError, match="cost_matrix"):
        compute_detection_metric(
            y_true, y_prob, threshold=0.5, metric_name="neg_expected_cost",
            cost_matrix=None,
        )


# ---------------------------------------------------------------------------
# compute_detection_metric — unknown metric_name
# ---------------------------------------------------------------------------


def test_compute_detection_metric_unknown_metric_raises() -> None:
    """compute_detection_metric raises ValueError for unknown metric_name."""
    from yield_risk.early_detection import compute_detection_metric

    y_true = np.array([0, 1])
    y_prob = np.array([0.3, 0.8])
    with pytest.raises(ValueError, match="metric_name"):
        compute_detection_metric(
            y_true, y_prob, threshold=0.5, metric_name="bogus_metric"
        )


# ---------------------------------------------------------------------------
# Single-class guard — pr_auc and roc_auc return nan + emit warning
# ---------------------------------------------------------------------------


def test_compute_detection_metric_pr_auc_single_class_nan_warning() -> None:
    """pr_auc on single-class y_true returns nan and emits a UserWarning."""
    from yield_risk.early_detection import compute_detection_metric

    y_true = np.array([0, 0, 0])
    y_prob = np.array([0.1, 0.4, 0.7])
    with pytest.warns(UserWarning):
        result = compute_detection_metric(
            y_true, y_prob, threshold=0.5, metric_name="pr_auc"
        )
    assert math.isnan(result)


def test_compute_detection_metric_roc_auc_single_class_nan_warning() -> None:
    """roc_auc on single-class y_true returns nan and emits a UserWarning."""
    from yield_risk.early_detection import compute_detection_metric

    y_true = np.array([1, 1, 1])
    y_prob = np.array([0.6, 0.8, 0.9])
    with pytest.warns(UserWarning):
        result = compute_detection_metric(
            y_true, y_prob, threshold=0.5, metric_name="roc_auc"
        )
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# Helpers for evaluate_config tests
# ---------------------------------------------------------------------------

_SECOM_RAW_DIR = Path("C:/Users/Virgil/Code/semiconductor-yield-root-cause/data/raw")
_SECOM_FILES_PRESENT = (
    (_SECOM_RAW_DIR / "secom.data").exists()
    and (_SECOM_RAW_DIR / "secom_labels.data").exists()
)


def _make_eval_df(
    n_rows: int = 120,
    n_sensors: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return synthetic DataFrame, y (30% positives), and sensor col names.

    Args:
        n_rows: Number of samples.
        n_sensors: Number of sensor columns.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (DataFrame, y, raw_sensor_cols).
    """
    rng = np.random.default_rng(seed)
    raw_sensor_cols = [f"sensor_{i:03d}" for i in range(n_sensors)]
    data = {col: rng.standard_normal(n_rows) for col in raw_sensor_cols}
    x = pd.DataFrame(data)
    # ~30% positives, fixed seed
    y = (rng.random(n_rows) < 0.30).astype(int)
    return x, y, raw_sensor_cols


def _make_eval_cfg(
    access: SensorAccess,
    model_family: str = "random_forest",
    selection_method: str = "none",
    max_features: int | None = None,
    missing_threshold: float = 0.9,
    cv_threshold: float = 0.0,
    correlation_threshold: float = 1.0,
    model_params: dict[str, object] | None = None,
) -> HyperparamConfig:
    """Return a HyperparamConfig for evaluate_config tests."""
    return HyperparamConfig(
        access=access,
        missing_threshold=missing_threshold,
        cv_threshold=cv_threshold,
        correlation_threshold=correlation_threshold,
        selection_method=selection_method,
        max_features=max_features,
        model_family=model_family,
        model_params=model_params if model_params is not None else {},
        threshold_policy="tune",
        threshold=0.5,
        false_alarm_rate=None,
    )


# ---------------------------------------------------------------------------
# evaluate_config — EARLY_DETECTION_FLOOR constant exists
# ---------------------------------------------------------------------------


def test_early_detection_floor_value() -> None:
    """EARLY_DETECTION_FLOOR is -1.0."""
    from yield_risk.early_detection import EARLY_DETECTION_FLOOR

    assert EARLY_DETECTION_FLOOR == -1.0


# ---------------------------------------------------------------------------
# evaluate_config — ConfigScore dataclass
# ---------------------------------------------------------------------------


def test_config_score_fields() -> None:
    """ConfigScore is a dataclass with the expected fields."""
    from yield_risk.early_detection import ConfigScore

    cs = ConfigScore(
        penalized_score=0.5,
        detection_metric=0.6,
        observation_fraction=0.4,
        n_features_selected=5.0,
        feasible=True,
        per_fold=[{"metric": 0.6, "n_features": 5.0, "threshold": 0.5}],
    )
    assert cs.penalized_score == 0.5
    assert cs.detection_metric == 0.6
    assert cs.observation_fraction == 0.4
    assert cs.n_features_selected == 5.0
    assert cs.feasible is True
    assert len(cs.per_fold) == 1


# ---------------------------------------------------------------------------
# evaluate_config — up-front validation
# ---------------------------------------------------------------------------


def test_evaluate_config_empty_raw_sensor_cols_raises() -> None:
    """evaluate_config raises ValueError when raw_sensor_cols is empty."""
    from yield_risk.early_detection import evaluate_config

    x, y, _ = _make_eval_df()
    access = SensorAccess(access_type="prefix", prefix_end=5)
    cfg = _make_eval_cfg(access)
    with pytest.raises(ValueError, match="No raw sensor_ columns found"):
        evaluate_config(
            x, y, [],
            cfg,
            inner_cv_folds=3,
            alpha=0.0,
            detection_metric="pr_auc",
            random_seed=0,
        )


def test_evaluate_config_inf_in_sensors_raises() -> None:
    """evaluate_config raises ValueError when sensor matrix contains +/-inf."""
    from yield_risk.early_detection import evaluate_config

    x, y, raw_sensor_cols = _make_eval_df()
    # Inject +inf into the sensor matrix
    x = x.copy()
    x.iloc[0, 0] = float("inf")
    access = SensorAccess(access_type="prefix", prefix_end=5)
    cfg = _make_eval_cfg(access)
    with pytest.raises(ValueError, match="Sensor matrix contains infinite values"):
        evaluate_config(
            x, y, raw_sensor_cols,
            cfg,
            inner_cv_folds=3,
            alpha=0.0,
            detection_metric="pr_auc",
            random_seed=0,
        )


def test_evaluate_config_neg_inf_in_sensors_raises() -> None:
    """evaluate_config raises ValueError when sensor matrix contains -inf."""
    from yield_risk.early_detection import evaluate_config

    x, y, raw_sensor_cols = _make_eval_df()
    x = x.copy()
    x.iloc[5, 3] = float("-inf")
    access = SensorAccess(access_type="prefix", prefix_end=5)
    cfg = _make_eval_cfg(access)
    with pytest.raises(ValueError, match="Sensor matrix contains infinite values"):
        evaluate_config(
            x, y, raw_sensor_cols,
            cfg,
            inner_cv_folds=3,
            alpha=0.0,
            detection_metric="pr_auc",
            random_seed=0,
        )


# ---------------------------------------------------------------------------
# evaluate_config — infeasibility floor (spec acceptance test 1)
# ---------------------------------------------------------------------------


def test_evaluate_config_infeasible_no_retained_features() -> None:
    """Config yielding no retained features → feasible=False, score=-1.0, metric=nan.

    Uses a cv_threshold so large that all sensor columns are dropped
    in every fold, making the preprocessor empty and every fold invalid.
    """
    from yield_risk.early_detection import EARLY_DETECTION_FLOOR, evaluate_config

    x, y, raw_sensor_cols = _make_eval_df(n_rows=120, n_sensors=20, seed=42)
    access = SensorAccess(access_type="prefix", prefix_end=20)
    # cv_threshold=1e10 will drop all columns.
    cfg = _make_eval_cfg(access, cv_threshold=1e10)

    result = evaluate_config(
        x, y, raw_sensor_cols,
        cfg,
        inner_cv_folds=3,
        alpha=0.1,
        detection_metric="pr_auc",
        random_seed=42,
    )

    assert result.feasible is False
    assert result.penalized_score == EARLY_DETECTION_FLOOR
    assert np.isnan(result.detection_metric)


def test_evaluate_config_infeasible_fewer_than_two_valid_folds() -> None:
    """Config with < 2 valid folds → feasible=False, score=-1.0, metric=nan.

    We craft data where all val folds are single-class (only positives) so
    pr_auc is undefined (returns nan) for every fold → < 2 valid → infeasible.
    """
    from yield_risk.early_detection import EARLY_DETECTION_FLOOR, evaluate_config

    # Build data: all labels are 1 → every val fold is single-class
    rng = np.random.default_rng(7)
    n_rows = 60
    n_sensors = 10
    raw_sensor_cols = [f"sensor_{i:03d}" for i in range(n_sensors)]
    x = pd.DataFrame(
        {col: rng.standard_normal(n_rows) for col in raw_sensor_cols}
    )
    y = np.ones(n_rows, dtype=int)  # all positives → single-class every fold

    access = SensorAccess(access_type="prefix", prefix_end=n_sensors)
    cfg = _make_eval_cfg(access, model_family="random_forest")

    result = evaluate_config(
        x, y, raw_sensor_cols,
        cfg,
        inner_cv_folds=3,
        alpha=0.1,
        detection_metric="pr_auc",
        random_seed=42,
    )

    assert result.feasible is False
    assert result.penalized_score == EARLY_DETECTION_FLOOR
    assert np.isnan(result.detection_metric)


# ---------------------------------------------------------------------------
# evaluate_config — single-class guard (spec acceptance test 2)
# ---------------------------------------------------------------------------


def test_evaluate_config_single_class_val_fold_no_raise() -> None:
    """evaluate_config does NOT raise when a val fold is single-class.

    Returns a ConfigScore. We construct data where some folds will have a
    single-class val split. With very few positives (2 out of 60) and 5 folds,
    at least one fold's val set will have no positives.
    """
    from yield_risk.early_detection import ConfigScore, evaluate_config

    rng = np.random.default_rng(11)
    n_rows = 60
    n_sensors = 10
    raw_sensor_cols = [f"sensor_{i:03d}" for i in range(n_sensors)]
    x = pd.DataFrame(
        {col: rng.standard_normal(n_rows) for col in raw_sensor_cols}
    )
    # Only 2 positives → some val folds will be single-class
    y = np.zeros(n_rows, dtype=int)
    y[0] = 1
    y[1] = 1

    access = SensorAccess(access_type="prefix", prefix_end=n_sensors)
    cfg = _make_eval_cfg(access, model_family="random_forest")

    # Must not raise
    result = evaluate_config(
        x, y, raw_sensor_cols,
        cfg,
        inner_cv_folds=5,
        alpha=0.0,
        detection_metric="pr_auc",
        random_seed=0,
    )

    assert isinstance(result, ConfigScore)
    # With only 2 positives across 5 folds, many folds will have nan metric
    # But the call itself must not raise


# ---------------------------------------------------------------------------
# evaluate_config — earliness test (spec acceptance test 3)
# ---------------------------------------------------------------------------


def test_evaluate_config_alpha_penalizes_late_window() -> None:
    """Raising alpha penalizes a late config (high obs_fraction) below an early one.

    At low alpha: late.penalized_score >= early.penalized_score (late sees
    all sensors including the informative ones → higher raw detection metric).
    At high alpha: late.penalized_score < early.penalized_score (the penalty
    alpha * obs_fraction dominates; late has obs_fraction=1.0 vs early=0.1).

    Data construction: the last 10 of 20 sensors are strongly predictive of y;
    the first 2 sensors are pure noise.  The "early" config sees only 2 sensors
    (noise), the "late" config sees all 20 (signal + noise).
    """
    from yield_risk.early_detection import evaluate_config

    rng = np.random.default_rng(2024)
    n_rows = 120
    n_sensors = 20
    raw_sensor_cols = [f"sensor_{i:03d}" for i in range(n_sensors)]

    # First 2 sensors: pure noise; sensors 10-19: carry signal
    noise = rng.standard_normal((n_rows, n_sensors))
    # Build y from the last 10 sensors: strong linear signal
    signal = noise[:, 10:20].sum(axis=1)
    y = (signal > np.median(signal)).astype(int)  # ~50% positives

    # Overlay the informative sensors with clear separation
    noise[:, 10:20] += y[:, None] * 2.0  # class 1 shifted up by 2
    x = pd.DataFrame(noise, columns=raw_sensor_cols)

    # Early config: only first 2 sensors (pure noise)
    early_access = SensorAccess(access_type="prefix", prefix_end=2)
    cfg_early = _make_eval_cfg(
        early_access,
        model_family="random_forest",
        model_params={"n_estimators": 20},
    )

    # Late config: all 20 sensors (includes informative ones 10-19)
    late_access = SensorAccess(access_type="prefix", prefix_end=n_sensors)
    cfg_late = _make_eval_cfg(
        late_access,
        model_family="random_forest",
        model_params={"n_estimators": 20},
    )

    common_kwargs: dict[str, object] = {
        "x": x,
        "y": y,
        "raw_sensor_cols": raw_sensor_cols,
        "inner_cv_folds": 3,
        "detection_metric": "pr_auc",
        "random_seed": 0,
    }

    # Low alpha=0: late should score > early (more sensors = more info)
    low_alpha = 0.0
    late_low = evaluate_config(cfg=cfg_late, alpha=low_alpha, **common_kwargs)  # type: ignore[arg-type]
    early_low = evaluate_config(cfg=cfg_early, alpha=low_alpha, **common_kwargs)  # type: ignore[arg-type]

    assert late_low.feasible, "Late config must be feasible at low alpha"
    assert early_low.feasible, "Early config must be feasible at low alpha"
    # Late sees informative sensors → higher detection metric
    assert late_low.penalized_score > early_low.penalized_score, (
        f"At alpha=0 expected late ({late_low.penalized_score:.3f}) > "
        f"early ({early_low.penalized_score:.3f})"
    )

    # High alpha: the obs_fraction penalty should flip the ordering.
    # late obs_fraction = 20/20 = 1.0, early obs_fraction = 2/20 = 0.1
    # at alpha=2.0: late penalty = 2.0*1.0=2.0, early penalty = 2.0*0.1=0.2
    high_alpha = 2.0
    late_high = evaluate_config(cfg=cfg_late, alpha=high_alpha, **common_kwargs)  # type: ignore[arg-type]
    early_high = evaluate_config(cfg=cfg_early, alpha=high_alpha, **common_kwargs)  # type: ignore[arg-type]

    assert late_high.feasible
    assert early_high.feasible
    assert late_high.penalized_score < early_high.penalized_score, (
        f"At alpha=2 expected late ({late_high.penalized_score:.3f}) < "
        f"early ({early_high.penalized_score:.3f})"
    )


# ---------------------------------------------------------------------------
# evaluate_config — determinism (spec acceptance test 4)
# ---------------------------------------------------------------------------


def test_evaluate_config_determinism() -> None:
    """Two evaluate_config calls with identical args return equal ConfigScore."""
    from yield_risk.early_detection import evaluate_config

    x, y, raw_sensor_cols = _make_eval_df(n_rows=120, n_sensors=20, seed=99)
    access = SensorAccess(access_type="prefix", prefix_end=10)
    cfg = _make_eval_cfg(
        access,
        model_family="random_forest",
        model_params={"n_estimators": 10},
    )

    kwargs: dict[str, object] = {
        "x": x,
        "y": y,
        "raw_sensor_cols": raw_sensor_cols,
        "cfg": cfg,
        "inner_cv_folds": 3,
        "alpha": 0.1,
        "detection_metric": "pr_auc",
        "random_seed": 42,
    }

    r1 = evaluate_config(**kwargs)  # type: ignore[arg-type]
    r2 = evaluate_config(**kwargs)  # type: ignore[arg-type]

    # Compare scalar fields
    if np.isnan(r1.detection_metric) and np.isnan(r2.detection_metric):
        pass  # both nan is equal
    else:
        assert r1.detection_metric == pytest.approx(r2.detection_metric)

    if np.isnan(r1.penalized_score) and np.isnan(r2.penalized_score):
        pass
    else:
        assert r1.penalized_score == pytest.approx(r2.penalized_score)

    assert r1.observation_fraction == r2.observation_fraction
    assert r1.n_features_selected == pytest.approx(r2.n_features_selected)
    assert r1.feasible == r2.feasible

    # Compare per_fold lists
    assert len(r1.per_fold) == len(r2.per_fold)
    for fd1, fd2 in zip(r1.per_fold, r2.per_fold):
        assert set(fd1.keys()) == set(fd2.keys())
        for k in fd1:
            v1, v2 = fd1[k], fd2[k]
            if math.isnan(v1) and math.isnan(v2):
                pass
            else:
                assert v1 == pytest.approx(v2), f"per_fold key {k!r}: {v1} != {v2}"


# ---------------------------------------------------------------------------
# evaluate_config — feasible result has correct fields
# ---------------------------------------------------------------------------


def test_evaluate_config_feasible_result_fields() -> None:
    """A feasible evaluate_config result has correct per_fold list and field types."""
    from yield_risk.early_detection import evaluate_config

    x, y, raw_sensor_cols = _make_eval_df(n_rows=120, n_sensors=20, seed=1)
    access = SensorAccess(access_type="prefix", prefix_end=10)
    cfg = _make_eval_cfg(
        access,
        model_family="random_forest",
        model_params={"n_estimators": 10},
    )

    result = evaluate_config(
        x, y, raw_sensor_cols,
        cfg,
        inner_cv_folds=3,
        alpha=0.1,
        detection_metric="pr_auc",
        random_seed=0,
    )

    assert isinstance(result.per_fold, list)
    assert len(result.per_fold) == 3  # inner_cv_folds=3
    for fold_dict in result.per_fold:
        assert "metric" in fold_dict
        assert "n_features" in fold_dict
        assert "threshold" in fold_dict

    if result.feasible:
        assert np.isfinite(result.penalized_score)
        # obs_fraction = 10/20 = 0.5
        assert result.observation_fraction == pytest.approx(0.5)
        # penalized_score = detection_metric - alpha * obs_fraction
        assert result.penalized_score == pytest.approx(
            result.detection_metric - 0.1 * result.observation_fraction
        )


# ---------------------------------------------------------------------------
# evaluate_config — scale_pos_weight auto-injection for xgboost/lightgbm
# ---------------------------------------------------------------------------


def test_evaluate_config_xgboost_runs_without_scale_pos_weight() -> None:
    """evaluate_config with xgboost and no scale_pos_weight runs without error."""
    from yield_risk.early_detection import evaluate_config

    x, y, raw_sensor_cols = _make_eval_df(n_rows=120, n_sensors=10, seed=3)
    access = SensorAccess(access_type="prefix", prefix_end=10)
    cfg = _make_eval_cfg(
        access,
        model_family="xgboost",
        # No scale_pos_weight → evaluate_config should compute it automatically
        model_params={"n_estimators": 5},
    )

    result = evaluate_config(
        x, y, raw_sensor_cols,
        cfg,
        inner_cv_folds=3,
        alpha=0.0,
        detection_metric="pr_auc",
        random_seed=0,
    )

    # Just confirm it returned a result without raising
    from yield_risk.early_detection import ConfigScore

    assert isinstance(result, ConfigScore)


# ---------------------------------------------------------------------------
# EarlyDetectionConfig + load_early_detection_config tests (Spec B §1)
# ---------------------------------------------------------------------------

_YAML_PATH = Path(
    "C:/Users/Virgil/Code/semiconductor-yield-root-cause"
    "/configs/early_detection_config.yaml"
)


def test_load_early_detection_config_defaults_on_missing_file(
    tmp_path: Path,
) -> None:
    """Missing config file returns all documented defaults without raising."""
    from yield_risk.early_detection import load_early_detection_config

    non_existent = tmp_path / "no_such_file.yaml"
    cfg = load_early_detection_config(non_existent)

    assert cfg.n_trials == 100
    assert cfg.sampler_seed == 42
    assert cfg.inner_cv_folds == 5
    assert cfg.detection_metric == "pr_auc"
    assert cfg.alpha == pytest.approx(0.10)
    assert cfg.access_types == ["prefix", "window"]
    assert cfg.min_window_size == 8
    assert cfg.max_window_size == 256
    assert cfg.model_families == [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    ]
    assert cfg.missing_threshold == (0.2, 0.6)
    assert cfg.cv_threshold == pytest.approx((1.0e-3, 1.0))
    assert cfg.correlation_threshold == pytest.approx((0.85, 0.99))
    assert cfg.selection_methods == [
        "none",
        "univariate",
        "mutual_info",
        "model_importance",
    ]
    assert cfg.max_features == (10, 200)
    assert cfg.threshold_policy == "tune"
    assert cfg.threshold_range == pytest.approx((0.01, 0.80))
    assert cfg.false_alarm_rate == pytest.approx(0.10)
    assert cfg.performance_tolerance == pytest.approx(0.05)
    assert cfg.curve_prefixes == [16, 32, 64, 128, 256, 512]


def test_load_early_detection_config_shipped_yaml_matches_defaults() -> None:
    """The shipped YAML produces identical values to the hard-coded defaults."""
    from yield_risk.early_detection import load_early_detection_config

    cfg_file = load_early_detection_config(_YAML_PATH)
    cfg_default = load_early_detection_config(
        _YAML_PATH.parent / "__nonexistent__.yaml"
    )
    assert cfg_file == cfg_default


def test_load_early_detection_config_unknown_key_raises(tmp_path: Path) -> None:
    """An unknown YAML key raises ValueError listing the offending key."""
    from yield_risk.early_detection import load_early_detection_config

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("n_trials: 50\nunknown_param: 99\n")
    with pytest.raises(ValueError, match="unknown_param"):
        load_early_detection_config(bad_yaml)


def test_load_early_detection_config_low_gt_high_raises(tmp_path: Path) -> None:
    """A bound pair with low > high raises ValueError."""
    from yield_risk.early_detection import load_early_detection_config

    bad_yaml = tmp_path / "bad_bounds.yaml"
    bad_yaml.write_text("missing_threshold: [0.8, 0.2]\n")
    with pytest.raises(ValueError, match="missing_threshold"):
        load_early_detection_config(bad_yaml)


def test_load_early_detection_config_empty_model_families_raises(
    tmp_path: Path,
) -> None:
    """Empty model_families list raises ValueError."""
    from yield_risk.early_detection import load_early_detection_config

    bad_yaml = tmp_path / "empty_fam.yaml"
    bad_yaml.write_text("model_families: []\n")
    with pytest.raises(ValueError, match="model_families"):
        load_early_detection_config(bad_yaml)


def test_load_early_detection_config_bad_detection_metric_raises(
    tmp_path: Path,
) -> None:
    """An unrecognised detection_metric value raises ValueError."""
    from yield_risk.early_detection import load_early_detection_config

    bad_yaml = tmp_path / "bad_metric.yaml"
    bad_yaml.write_text("detection_metric: accuracy\n")
    with pytest.raises(ValueError, match="detection_metric"):
        load_early_detection_config(bad_yaml)


def test_load_early_detection_config_bad_threshold_policy_raises(
    tmp_path: Path,
) -> None:
    """An unrecognised threshold_policy value raises ValueError."""
    from yield_risk.early_detection import load_early_detection_config

    bad_yaml = tmp_path / "bad_policy.yaml"
    bad_yaml.write_text("threshold_policy: greedy\n")
    with pytest.raises(ValueError, match="threshold_policy"):
        load_early_detection_config(bad_yaml)


def test_load_early_detection_config_overrides_applied_last(tmp_path: Path) -> None:
    """CLI overrides are applied last and win over YAML values."""
    from yield_risk.early_detection import load_early_detection_config

    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("n_trials: 20\nsampler_seed: 7\n")
    cfg = load_early_detection_config(yaml_file, overrides={"n_trials": 99})
    assert cfg.n_trials == 99
    assert cfg.sampler_seed == 7  # from YAML, not clobbered


def test_load_early_detection_config_none_overrides_ignored(tmp_path: Path) -> None:
    """None values in overrides dict do not clobber real config values."""
    from yield_risk.early_detection import load_early_detection_config

    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("n_trials: 30\n")
    cfg = load_early_detection_config(
        yaml_file, overrides={"n_trials": None, "sampler_seed": None}
    )
    assert cfg.n_trials == 30  # YAML value kept; None did not clobber
    assert cfg.sampler_seed == 42  # default kept; None did not clobber


def test_early_detection_config_is_frozen_dataclass() -> None:
    """EarlyDetectionConfig is frozen — fields cannot be reassigned."""
    from yield_risk.early_detection import load_early_detection_config

    cfg = load_early_detection_config(
        _YAML_PATH.parent / "__nonexistent__.yaml"
    )
    with pytest.raises(Exception):
        cfg.n_trials = 999  # type: ignore[misc]


def test_load_early_detection_config_n_trials_less_than_1_raises(
    tmp_path: Path,
) -> None:
    """n_trials < 1 raises ValueError."""
    from yield_risk.early_detection import load_early_detection_config

    bad_yaml = tmp_path / "bad_trials.yaml"
    bad_yaml.write_text("n_trials: 0\n")
    with pytest.raises(ValueError, match="n_trials"):
        load_early_detection_config(bad_yaml)


def test_load_early_detection_config_inner_cv_folds_less_than_2_raises(
    tmp_path: Path,
) -> None:
    """inner_cv_folds < 2 raises ValueError."""
    from yield_risk.early_detection import load_early_detection_config

    bad_yaml = tmp_path / "bad_folds.yaml"
    bad_yaml.write_text("inner_cv_folds: 1\n")
    with pytest.raises(ValueError, match="inner_cv_folds"):
        load_early_detection_config(bad_yaml)


def test_load_early_detection_config_empty_access_types_raises(
    tmp_path: Path,
) -> None:
    """Empty access_types list raises ValueError."""
    from yield_risk.early_detection import load_early_detection_config

    bad_yaml = tmp_path / "bad_access.yaml"
    bad_yaml.write_text("access_types: []\n")
    with pytest.raises(ValueError, match="access_types"):
        load_early_detection_config(bad_yaml)


def test_load_early_detection_config_invalid_access_type_entry_raises(
    tmp_path: Path,
) -> None:
    """An unrecognised entry in access_types raises ValueError."""
    from yield_risk.early_detection import load_early_detection_config

    bad_yaml = tmp_path / "bad_access_entry.yaml"
    bad_yaml.write_text("access_types: [prefix, sliding]\n")
    with pytest.raises(ValueError, match="access_types"):
        load_early_detection_config(bad_yaml)


def test_load_early_detection_config_invalid_model_family_raises(
    tmp_path: Path,
) -> None:
    """A model family outside the four Spec-A families raises ValueError."""
    from yield_risk.early_detection import load_early_detection_config

    bad_yaml = tmp_path / "bad_family.yaml"
    bad_yaml.write_text("model_families: [logistic_regression, neural_net]\n")
    with pytest.raises(ValueError, match="model_families"):
        load_early_detection_config(bad_yaml)


def test_load_early_detection_config_empty_selection_methods_raises(
    tmp_path: Path,
) -> None:
    """Empty selection_methods raises ValueError."""
    from yield_risk.early_detection import load_early_detection_config

    bad_yaml = tmp_path / "bad_sel.yaml"
    bad_yaml.write_text("selection_methods: []\n")
    with pytest.raises(ValueError, match="selection_methods"):
        load_early_detection_config(bad_yaml)


def test_load_early_detection_config_invalid_selection_method_raises(
    tmp_path: Path,
) -> None:
    """An unrecognised selection method raises ValueError."""
    from yield_risk.early_detection import load_early_detection_config

    bad_yaml = tmp_path / "bad_sel_entry.yaml"
    bad_yaml.write_text("selection_methods: [none, chi2]\n")
    with pytest.raises(ValueError, match="selection_methods"):
        load_early_detection_config(bad_yaml)


# ---------------------------------------------------------------------------
# Real-data sanity check (skipped unless SECOM data present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _SECOM_FILES_PRESENT,
    reason="manual real-data sanity check; requires SECOM data",
)
def test_real_data_sanity_check() -> None:
    """Sanity check: full-prefix config on SECOM data yields finite, feasible PR-AUC.

    Loads SECOM data, builds a full-range prefix config with no feature
    selection, runs evaluate_config with 5 inner CV folds, and asserts that
    the mean PR-AUC is in a loose neighbourhood of the tabular baseline (~0.19).
    Range [0.10, 0.40] is intentionally wide — this is a wiring sanity check.
    """
    from yield_risk.data import load_secom
    from yield_risk.early_detection import evaluate_config

    df = load_secom(_SECOM_RAW_DIR)
    raw_sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    n_sensors = len(raw_sensor_cols)
    y = df["label"].to_numpy(dtype=int)

    access = SensorAccess(access_type="prefix", prefix_end=n_sensors)
    cfg = HyperparamConfig(
        access=access,
        missing_threshold=0.5,
        cv_threshold=0.0,
        correlation_threshold=1.0,
        selection_method="none",
        max_features=None,
        model_family="random_forest",
        model_params={"n_estimators": 50},
        threshold_policy="tune",
        threshold=0.5,
        false_alarm_rate=None,
    )

    result = evaluate_config(
        df, y, raw_sensor_cols,
        cfg,
        inner_cv_folds=5,
        alpha=0.0,
        detection_metric="pr_auc",
        random_seed=42,
    )

    assert result.feasible, "Expected feasible result on SECOM data"
    assert np.isfinite(result.detection_metric), (
        "Expected finite mean PR-AUC on SECOM data"
    )
    assert 0.10 < result.detection_metric < 0.40, (
        f"Mean PR-AUC {result.detection_metric:.3f} outside expected range [0.10, 0.40]"
    )


# ---------------------------------------------------------------------------
# suggest_config tests (Spec B §2)
# ---------------------------------------------------------------------------


def _make_ed_cfg_for_suggest(
    access_types: list[str] | None = None,
    model_families: list[str] | None = None,
    selection_methods: list[str] | None = None,
    threshold_policy: str = "tune",
) -> EarlyDetectionConfig:
    """Return a minimal EarlyDetectionConfig for suggest_config tests.

    Args:
        access_types: Override access_types list.
        model_families: Override model_families list.
        selection_methods: Override selection_methods list.
        threshold_policy: Override threshold_policy.

    Returns:
        An EarlyDetectionConfig ready for suggest_config.
    """
    return EarlyDetectionConfig(
        n_trials=10,
        sampler_seed=0,
        inner_cv_folds=3,
        detection_metric="pr_auc",
        alpha=0.1,
        access_types=(
            access_types if access_types is not None else ["prefix", "window"]
        ),
        min_window_size=2,
        max_window_size=10,
        model_families=(
            model_families if model_families is not None else ["random_forest"]
        ),
        missing_threshold=(0.2, 0.6),
        cv_threshold=(1e-3, 1.0),
        correlation_threshold=(0.85, 0.99),
        selection_methods=(
            selection_methods if selection_methods is not None else ["none"]
        ),
        max_features=(5, 50),
        threshold_policy=threshold_policy,
        threshold_range=(0.1, 0.8),
        false_alarm_rate=0.1,
        performance_tolerance=0.05,
        curve_prefixes=[16, 32],
    )


def _ask_trial(study: optuna.Study) -> optuna.Trial:
    """Return a fresh trial from the study.

    Args:
        study: An Optuna study.

    Returns:
        A trial ready for parameter suggestions.
    """
    return study.ask()


def test_suggest_config_returns_hyperparam_config_for_prefix() -> None:
    """suggest_config with access_types=['prefix'] returns a valid HyperparamConfig.

    A returned HyperparamConfig with prefix access must have access_type='prefix'
    and prefix_end >= 1.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest(access_types=["prefix"])
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg, n_sensors=20)

    assert isinstance(cfg, HyperparamConfig)
    assert cfg.access.access_type == "prefix"
    assert cfg.access.prefix_end is not None
    assert cfg.access.prefix_end >= 1
    assert cfg.access.window_start is None
    assert cfg.access.window_size is None


def test_suggest_config_returns_hyperparam_config_for_window() -> None:
    """suggest_config with access_types=['window'] returns a valid HyperparamConfig.

    A returned HyperparamConfig with window access must have access_type='window'
    with valid window_start and window_size, and no prefix_end.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest(access_types=["window"])
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg, n_sensors=20)

    assert isinstance(cfg, HyperparamConfig)
    assert cfg.access.access_type == "window"
    assert cfg.access.window_start is not None
    assert cfg.access.window_size is not None
    assert cfg.access.window_start >= 0
    assert cfg.access.window_size >= 1
    assert cfg.access.prefix_end is None


def test_suggest_config_selection_none_max_features_is_none() -> None:
    """suggest_config with selection_methods=['none'] sets max_features=None.

    When the only available selection method is 'none', the returned config
    must have max_features=None and not sample a value.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest(selection_methods=["none"])
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg, n_sensors=20)

    assert cfg.selection_method == "none"
    assert cfg.max_features is None


def test_suggest_config_selection_non_none_max_features_is_int() -> None:
    """suggest_config with non-none selection method sets max_features to an int.

    When selection_methods=['univariate'], the returned config must have a
    non-None integer max_features in the configured range.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest(selection_methods=["univariate"])
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg, n_sensors=20)

    assert cfg.selection_method == "univariate"
    assert cfg.max_features is not None
    assert isinstance(cfg.max_features, int)
    assert ed_cfg.max_features[0] <= cfg.max_features <= ed_cfg.max_features[1]


def test_suggest_config_logistic_regression_model_params_keys() -> None:
    """suggest_config for logistic_regression populates C, penalty, class_weight.

    The model_params dict must contain bare estimator keys C, penalty, class_weight.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest(model_families=["logistic_regression"])
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg, n_sensors=20)

    assert cfg.model_family == "logistic_regression"
    assert "C" in cfg.model_params
    assert "penalty" in cfg.model_params
    assert "class_weight" in cfg.model_params
    assert "lr_C" not in cfg.model_params
    assert "scale_pos_weight" not in cfg.model_params


def test_suggest_config_logistic_regression_l1_penalty_uses_liblinear() -> None:
    """suggest_config sets solver='liblinear' whenever lr penalty=='l1'.

    Runs multiple trials and checks the invariant: every config with penalty='l1'
    must have solver='liblinear' in model_params.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest(model_families=["logistic_regression"])
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    for _ in range(20):
        trial = _ask_trial(study)
        cfg = suggest_config(trial, ed_cfg, n_sensors=20)
        if cfg.model_params.get("penalty") == "l1":
            assert cfg.model_params.get("solver") == "liblinear", (
                "penalty='l1' must force solver='liblinear'"
            )


def test_suggest_config_logistic_regression_l2_penalty_no_solver_override() -> None:
    """suggest_config does NOT set solver='liblinear' when penalty=='l2'.

    When penalty is 'l2', the model_params must not contain a 'solver' key.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest(model_families=["logistic_regression"])
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    found_l2 = False
    for _ in range(30):
        trial = _ask_trial(study)
        cfg = suggest_config(trial, ed_cfg, n_sensors=20)
        if cfg.model_params.get("penalty") == "l2":
            assert "solver" not in cfg.model_params, (
                "penalty='l2' must not add solver key"
            )
            found_l2 = True
            break
    assert found_l2, "No l2 penalty sample found in 30 trials"


def test_suggest_config_random_forest_model_params_keys() -> None:
    """suggest_config for random_forest populates the correct bare estimator keys.

    The model_params dict must contain n_estimators, max_depth, min_samples_leaf,
    max_features, class_weight — with bare keys, not rf_-prefixed Optuna names.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest(model_families=["random_forest"])
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg, n_sensors=20)

    assert cfg.model_family == "random_forest"
    assert "n_estimators" in cfg.model_params
    assert "max_depth" in cfg.model_params
    assert "min_samples_leaf" in cfg.model_params
    assert "max_features" in cfg.model_params
    assert "class_weight" in cfg.model_params
    assert "rf_n_estimators" not in cfg.model_params
    assert "scale_pos_weight" not in cfg.model_params


def test_suggest_config_xgboost_model_params_keys() -> None:
    """suggest_config for xgboost populates the correct bare estimator keys.

    The model_params dict must contain learning_rate, n_estimators, max_depth,
    subsample — without scale_pos_weight.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest(model_families=["xgboost"])
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg, n_sensors=20)

    assert cfg.model_family == "xgboost"
    assert "learning_rate" in cfg.model_params
    assert "n_estimators" in cfg.model_params
    assert "max_depth" in cfg.model_params
    assert "subsample" in cfg.model_params
    assert "scale_pos_weight" not in cfg.model_params
    assert "xgboost_learning_rate" not in cfg.model_params


def test_suggest_config_lightgbm_model_params_keys() -> None:
    """suggest_config for lightgbm populates the correct bare estimator keys.

    The model_params dict must contain learning_rate, n_estimators, max_depth,
    subsample — without scale_pos_weight.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest(model_families=["lightgbm"])
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg, n_sensors=20)

    assert cfg.model_family == "lightgbm"
    assert "learning_rate" in cfg.model_params
    assert "n_estimators" in cfg.model_params
    assert "max_depth" in cfg.model_params
    assert "subsample" in cfg.model_params
    assert "scale_pos_weight" not in cfg.model_params
    assert "lightgbm_learning_rate" not in cfg.model_params


def test_suggest_config_never_sets_scale_pos_weight_for_any_family() -> None:
    """suggest_config never sets scale_pos_weight for any model family.

    Runs a trial for each of the four families and asserts scale_pos_weight
    is absent from model_params in every case.
    """
    from yield_risk.early_detection import suggest_config

    for family in ["logistic_regression", "random_forest", "xgboost", "lightgbm"]:
        ed_cfg = _make_ed_cfg_for_suggest(model_families=[family])
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
        )
        trial = _ask_trial(study)
        cfg = suggest_config(trial, ed_cfg, n_sensors=20)
        assert "scale_pos_weight" not in cfg.model_params, (
            f"scale_pos_weight must not be set for family '{family}'"
        )


def test_suggest_config_threshold_policy_tune_sets_threshold_not_far() -> None:
    """suggest_config with threshold_policy='tune' sets threshold, not false_alarm_rate.

    The returned config must have threshold in threshold_range and
    false_alarm_rate=None.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest(threshold_policy="tune")
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg, n_sensors=20)

    assert cfg.threshold_policy == "tune"
    assert cfg.threshold is not None
    assert ed_cfg.threshold_range[0] <= cfg.threshold <= ed_cfg.threshold_range[1]
    assert cfg.false_alarm_rate is None


def test_suggest_config_far_constraint_policy_sets_far_not_threshold() -> None:
    """suggest_config with 'far_constraint' sets false_alarm_rate, not threshold.

    The returned config must have threshold=None and false_alarm_rate equal to
    ed_cfg.false_alarm_rate.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest(threshold_policy="far_constraint")
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg, n_sensors=20)

    assert cfg.threshold_policy == "far_constraint"
    assert cfg.threshold is None
    assert cfg.false_alarm_rate == ed_cfg.false_alarm_rate


def test_suggest_config_window_bounds_clamped_when_min_window_gt_n_sensors() -> None:
    """suggest_config clamps window bounds when min_window_size > n_sensors.

    When min_window_size=15 and n_sensors=5, ws_high=min(20,5)=5,
    ws_low=min(15,5)=5. Both become 5, making suggest_int valid (low==high).
    The returned window_size must be 5.
    """
    from yield_risk.early_detection import suggest_config

    # Force min_window_size > n_sensors by constructing a custom config
    ed_cfg_tiny = EarlyDetectionConfig(
        n_trials=10,
        sampler_seed=0,
        inner_cv_folds=3,
        detection_metric="pr_auc",
        alpha=0.1,
        access_types=["window"],
        min_window_size=15,  # larger than n_sensors
        max_window_size=20,  # larger than n_sensors
        model_families=["random_forest"],
        missing_threshold=(0.2, 0.6),
        cv_threshold=(1e-3, 1.0),
        correlation_threshold=(0.85, 0.99),
        selection_methods=["none"],
        max_features=(5, 50),
        threshold_policy="tune",
        threshold_range=(0.1, 0.8),
        false_alarm_rate=0.1,
        performance_tolerance=0.05,
        curve_prefixes=[16, 32],
    )
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg_tiny, n_sensors=5)

    # ws_high = min(20, 5) = 5; ws_low = min(15, 5) = 5; window_size == 5
    assert cfg.access.window_size == 5


def test_suggest_config_window_single_sensor_valid() -> None:
    """suggest_config with n_sensors=1 and window access produces valid config.

    window_start must be 0 (range [0, 0]) and window_size must be 1.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = EarlyDetectionConfig(
        n_trials=10,
        sampler_seed=0,
        inner_cv_folds=3,
        detection_metric="pr_auc",
        alpha=0.1,
        access_types=["window"],
        min_window_size=1,
        max_window_size=1,
        model_families=["random_forest"],
        missing_threshold=(0.2, 0.6),
        cv_threshold=(1e-3, 1.0),
        correlation_threshold=(0.85, 0.99),
        selection_methods=["none"],
        max_features=(5, 50),
        threshold_policy="tune",
        threshold_range=(0.1, 0.8),
        false_alarm_rate=0.1,
        performance_tolerance=0.05,
        curve_prefixes=[16, 32],
    )
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg, n_sensors=1)

    assert cfg.access.access_type == "window"
    assert cfg.access.window_start == 0
    assert cfg.access.window_size == 1


def test_suggest_config_preprocessing_fields_in_bounds() -> None:
    """suggest_config samples preprocessing floats within configured bounds.

    missing_threshold, cv_threshold, and correlation_threshold must all
    fall within their respective configured ranges.
    """
    from yield_risk.early_detection import suggest_config

    ed_cfg = _make_ed_cfg_for_suggest()
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction="maximize"
    )
    trial = _ask_trial(study)
    cfg = suggest_config(trial, ed_cfg, n_sensors=20)

    lo, hi = ed_cfg.missing_threshold
    assert lo <= cfg.missing_threshold <= hi
    lo, hi = ed_cfg.cv_threshold
    assert lo <= cfg.cv_threshold <= hi
    lo, hi = ed_cfg.correlation_threshold
    assert lo <= cfg.correlation_threshold <= hi


# ---------------------------------------------------------------------------
# run_study tests (Spec B §3)
# ---------------------------------------------------------------------------


def _make_ed_cfg_for_run_study(
    n_trials: int = 3,
    inner_cv_folds: int = 2,
    detection_metric: str = "pr_auc",
    n_sensors: int = 10,
) -> EarlyDetectionConfig:
    """Return a minimal EarlyDetectionConfig for run_study tests.

    Args:
        n_trials: Number of Optuna trials.
        inner_cv_folds: Number of inner CV folds.
        detection_metric: Primary scoring metric.
        n_sensors: Used to set max_window_size sensibly.

    Returns:
        An EarlyDetectionConfig ready for run_study.
    """
    return EarlyDetectionConfig(
        n_trials=n_trials,
        sampler_seed=7,
        inner_cv_folds=inner_cv_folds,
        detection_metric=detection_metric,
        alpha=0.1,
        access_types=["prefix"],
        min_window_size=2,
        max_window_size=n_sensors,
        model_families=["random_forest"],
        missing_threshold=(0.2, 0.9),
        cv_threshold=(1e-3, 1.0),
        correlation_threshold=(0.85, 0.99),
        selection_methods=["none"],
        max_features=(2, n_sensors),
        threshold_policy="tune",
        threshold_range=(0.1, 0.8),
        false_alarm_rate=0.1,
        performance_tolerance=0.05,
        curve_prefixes=[16, 32],
    )


def _make_run_study_data(
    n_rows: int = 80,
    n_sensors: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return synthetic training data for run_study tests.

    Args:
        n_rows: Number of samples.
        n_sensors: Number of sensor columns.
        seed: RNG seed.

    Returns:
        Tuple of (x_train, y_train, raw_sensor_cols).
    """
    rng = np.random.default_rng(seed)
    raw_sensor_cols = [f"sensor_{i:03d}" for i in range(n_sensors)]
    x = pd.DataFrame(
        {col: rng.standard_normal(n_rows) for col in raw_sensor_cols}
    )
    y = (rng.random(n_rows) < 0.35).astype(int)
    return x, y, raw_sensor_cols


def test_run_study_returns_optuna_study() -> None:
    """run_study returns an optuna.Study instance."""
    from yield_risk.early_detection import run_study

    x, y, raw_sensor_cols = _make_run_study_data()
    ed_cfg = _make_ed_cfg_for_run_study(n_trials=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = run_study(
            x, y, raw_sensor_cols, ed_cfg, random_seed=0
        )

    assert isinstance(study, optuna.Study)


def test_run_study_trial_count_matches_n_trials() -> None:
    """run_study produces exactly n_trials trials."""
    from yield_risk.early_detection import run_study

    n_trials = 3
    x, y, raw_sensor_cols = _make_run_study_data()
    ed_cfg = _make_ed_cfg_for_run_study(n_trials=n_trials)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = run_study(
            x, y, raw_sensor_cols, ed_cfg, random_seed=0
        )

    assert len(study.trials) == n_trials


def test_run_study_trials_carry_all_documented_user_attrs() -> None:
    """Every trial in the study has all seven documented user attrs."""
    from yield_risk.early_detection import run_study

    x, y, raw_sensor_cols = _make_run_study_data()
    ed_cfg = _make_ed_cfg_for_run_study(n_trials=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = run_study(
            x, y, raw_sensor_cols, ed_cfg, random_seed=0
        )

    expected_attrs = {
        "detection_metric",
        "observation_fraction",
        "n_features_selected",
        "feasible",
        "access_type",
        "latest_index",
        "model_family",
    }
    for trial in study.trials:
        assert expected_attrs <= set(trial.user_attrs.keys()), (
            f"Trial {trial.number} missing attrs: "
            f"{expected_attrs - set(trial.user_attrs.keys())}"
        )


def test_run_study_infeasible_trials_score_floor() -> None:
    """Infeasible trials (feasible=False) score exactly EARLY_DETECTION_FLOOR.

    Any trial whose user_attr 'feasible' is False must have value equal to
    EARLY_DETECTION_FLOOR (-1.0).
    """
    from yield_risk.early_detection import EARLY_DETECTION_FLOOR, run_study

    x, y, raw_sensor_cols = _make_run_study_data()
    ed_cfg = _make_ed_cfg_for_run_study(n_trials=3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = run_study(
            x, y, raw_sensor_cols, ed_cfg, random_seed=0
        )

    for trial in study.trials:
        if not trial.user_attrs.get("feasible", True):
            assert trial.value == EARLY_DETECTION_FLOOR, (
                f"Trial {trial.number}: infeasible trial value "
                f"{trial.value} != {EARLY_DETECTION_FLOOR}"
            )


def test_run_study_infeasible_no_raise() -> None:
    """run_study completes without raising even when all trials are infeasible.

    Forces infeasibility via cv_threshold so large all columns are dropped.
    """
    from yield_risk.early_detection import EARLY_DETECTION_FLOOR, run_study

    x, y, raw_sensor_cols = _make_run_study_data()
    # Huge cv_threshold: all columns dropped → all trials infeasible
    ed_cfg = EarlyDetectionConfig(
        n_trials=3,
        sampler_seed=7,
        inner_cv_folds=2,
        detection_metric="pr_auc",
        alpha=0.1,
        access_types=["prefix"],
        min_window_size=2,
        max_window_size=10,
        model_families=["random_forest"],
        missing_threshold=(0.2, 0.9),
        cv_threshold=(1e10, 2e10),  # so large all columns dropped
        correlation_threshold=(0.85, 0.99),
        selection_methods=["none"],
        max_features=(2, 10),
        threshold_policy="tune",
        threshold_range=(0.1, 0.8),
        false_alarm_rate=0.1,
        performance_tolerance=0.05,
        curve_prefixes=[16, 32],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # Must not raise
        study = run_study(
            x, y, raw_sensor_cols, ed_cfg, random_seed=0
        )

    assert len(study.trials) == 3
    for trial in study.trials:
        assert trial.value == EARLY_DETECTION_FLOOR


def test_run_study_determinism() -> None:
    """Two run_study calls with identical inputs produce identical results.

    Same best_params, same best_value, and same per-trial (value, params) pairs.
    """
    from yield_risk.early_detection import run_study

    x, y, raw_sensor_cols = _make_run_study_data()
    ed_cfg = _make_ed_cfg_for_run_study(n_trials=4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study1 = run_study(x, y, raw_sensor_cols, ed_cfg, random_seed=0)
        study2 = run_study(x, y, raw_sensor_cols, ed_cfg, random_seed=0)

    assert study1.best_value == pytest.approx(study2.best_value)
    assert study1.best_params == study2.best_params

    for t1, t2 in zip(study1.trials, study2.trials):
        assert t1.value == pytest.approx(t2.value), (
            f"Trial {t1.number}: values differ {t1.value} vs {t2.value}"
        )
        assert t1.params == t2.params, (
            f"Trial {t1.number}: params differ"
        )


def test_run_study_neg_expected_cost_without_cost_matrix_raises() -> None:
    """run_study raises ValueError for neg_expected_cost without cost_matrix."""
    from yield_risk.early_detection import run_study

    x, y, raw_sensor_cols = _make_run_study_data()
    ed_cfg = _make_ed_cfg_for_run_study(
        n_trials=2, detection_metric="neg_expected_cost"
    )

    with pytest.raises(ValueError, match="cost_matrix"):
        run_study(
            x, y, raw_sensor_cols, ed_cfg, random_seed=0, cost_matrix=None
        )


def test_run_study_user_attr_types_are_correct() -> None:
    """User attrs on every trial have correct Python types.

    detection_metric: float (may be nan), observation_fraction: float,
    n_features_selected: float, feasible: bool,
    access_type: str, latest_index: int, model_family: str.
    """
    from yield_risk.early_detection import run_study

    x, y, raw_sensor_cols = _make_run_study_data()
    ed_cfg = _make_ed_cfg_for_run_study(n_trials=3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = run_study(
            x, y, raw_sensor_cols, ed_cfg, random_seed=0
        )

    for trial in study.trials:
        attrs = trial.user_attrs
        assert isinstance(attrs["detection_metric"], float)
        assert isinstance(attrs["observation_fraction"], float)
        assert isinstance(attrs["n_features_selected"], float)
        assert isinstance(attrs["feasible"], bool)
        assert isinstance(attrs["access_type"], str)
        assert isinstance(attrs["latest_index"], int)
        assert isinstance(attrs["model_family"], str)


# ---------------------------------------------------------------------------
# Helpers shared by build_best_record tests
# ---------------------------------------------------------------------------


def _make_ed_cfg_for_build(
    n_trials: int = 4,
    n_sensors: int = 10,
) -> EarlyDetectionConfig:
    """Return a small EarlyDetectionConfig for build_best_record tests.

    Args:
        n_trials: Number of Optuna trials.
        n_sensors: Total number of sensor columns (sets max_window_size).

    Returns:
        A minimal EarlyDetectionConfig.
    """
    return EarlyDetectionConfig(
        n_trials=n_trials,
        sampler_seed=7,
        inner_cv_folds=2,
        detection_metric="pr_auc",
        alpha=0.1,
        access_types=["prefix"],
        min_window_size=2,
        max_window_size=n_sensors,
        model_families=["random_forest"],
        missing_threshold=(0.2, 0.9),
        cv_threshold=(1e-3, 1.0),
        correlation_threshold=(0.85, 0.99),
        selection_methods=["none"],
        max_features=(2, n_sensors),
        threshold_policy="tune",
        threshold_range=(0.1, 0.8),
        false_alarm_rate=0.1,
        performance_tolerance=0.05,
        curve_prefixes=[16, 32],
    )


def _make_build_data(
    n_rows: int = 80,
    n_sensors: int = 10,
    seed: int = 77,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return synthetic data for build_best_record tests.

    Args:
        n_rows: Number of samples.
        n_sensors: Number of sensor columns.
        seed: RNG seed.

    Returns:
        Tuple of (x_train, y_train, raw_sensor_cols).
    """
    rng = np.random.default_rng(seed)
    raw_sensor_cols = [f"sensor_{i:03d}" for i in range(n_sensors)]
    x = pd.DataFrame(
        {col: rng.standard_normal(n_rows) for col in raw_sensor_cols}
    )
    y = (rng.random(n_rows) < 0.35).astype(int)
    return x, y, raw_sensor_cols


# ---------------------------------------------------------------------------
# hyperparam_config_to_dict — shape and round-trip
# ---------------------------------------------------------------------------


def test_hyperparam_config_to_dict_shape_prefix_tune() -> None:
    """hyperparam_config_to_dict with prefix/tune config has correct keys/values."""
    from yield_risk.early_detection import hyperparam_config_to_dict

    access = SensorAccess(access_type="prefix", prefix_end=5)
    cfg = HyperparamConfig(
        access=access,
        missing_threshold=0.4,
        cv_threshold=0.001,
        correlation_threshold=0.9,
        selection_method="none",
        max_features=None,
        model_family="random_forest",
        model_params={"n_estimators": 100, "class_weight": None},
        threshold_policy="tune",
        threshold=0.3,
        false_alarm_rate=None,
    )
    d = hyperparam_config_to_dict(cfg)

    assert d["access"] == {
        "access_type": "prefix",
        "prefix_end": 5,
        "window_start": None,
        "window_size": None,
    }
    assert d["missing_threshold"] == pytest.approx(0.4)
    assert d["cv_threshold"] == pytest.approx(0.001)
    assert d["correlation_threshold"] == pytest.approx(0.9)
    assert d["selection_method"] == "none"
    assert d["max_features"] is None
    assert d["model_family"] == "random_forest"
    assert d["model_params"] == {"n_estimators": 100, "class_weight": None}
    assert d["threshold_policy"] == "tune"
    assert d["threshold"] == pytest.approx(0.3)
    assert d["false_alarm_rate"] is None


def test_hyperparam_config_to_dict_shape_window_far() -> None:
    """hyperparam_config_to_dict with window/far_constraint config has correct shape."""
    from yield_risk.early_detection import hyperparam_config_to_dict

    access = SensorAccess(access_type="window", window_start=2, window_size=4)
    cfg = HyperparamConfig(
        access=access,
        missing_threshold=0.3,
        cv_threshold=0.005,
        correlation_threshold=0.95,
        selection_method="univariate",
        max_features=10,
        model_family="logistic_regression",
        model_params={"C": 1.0, "penalty": "l2", "class_weight": "balanced"},
        threshold_policy="far_constraint",
        threshold=None,
        false_alarm_rate=0.05,
    )
    d = hyperparam_config_to_dict(cfg)

    assert d["access"] == {
        "access_type": "window",
        "prefix_end": None,
        "window_start": 2,
        "window_size": 4,
    }
    assert d["max_features"] == 10
    assert d["threshold"] is None
    assert d["false_alarm_rate"] == pytest.approx(0.05)


def test_hyperparam_config_to_dict_round_trip() -> None:
    """Serialized HyperparamConfig can be reconstructed to an equal object."""
    from yield_risk.early_detection import hyperparam_config_to_dict

    access = SensorAccess(access_type="prefix", prefix_end=7)
    cfg = HyperparamConfig(
        access=access,
        missing_threshold=0.45,
        cv_threshold=0.002,
        correlation_threshold=0.88,
        selection_method="mutual_info",
        max_features=15,
        model_family="xgboost",
        model_params={"learning_rate": 0.05, "n_estimators": 200},
        threshold_policy="tune",
        threshold=0.4,
        false_alarm_rate=None,
    )
    d = hyperparam_config_to_dict(cfg)

    # Rebuild from the dict
    access_reconstructed = SensorAccess(**d["access"])  # type: ignore[arg-type]
    cfg_reconstructed = HyperparamConfig(
        access=access_reconstructed,
        missing_threshold=d["missing_threshold"],  # type: ignore[arg-type]
        cv_threshold=d["cv_threshold"],  # type: ignore[arg-type]
        correlation_threshold=d["correlation_threshold"],  # type: ignore[arg-type]
        selection_method=d["selection_method"],  # type: ignore[arg-type]
        max_features=d["max_features"],  # type: ignore[arg-type]
        model_family=d["model_family"],  # type: ignore[arg-type]
        model_params=d["model_params"],  # type: ignore[arg-type]
        threshold_policy=d["threshold_policy"],  # type: ignore[arg-type]
        threshold=d["threshold"],  # type: ignore[arg-type]
        false_alarm_rate=d["false_alarm_rate"],  # type: ignore[arg-type]
    )
    assert cfg_reconstructed == cfg


# ---------------------------------------------------------------------------
# build_best_record — provenance keys present with expected values
# ---------------------------------------------------------------------------


def test_build_best_record_provenance_keys_present() -> None:
    """build_best_record record contains all 9 provenance keys with correct values."""
    from yield_risk.early_detection import build_best_record, run_study

    x, y, raw_sensor_cols = _make_build_data()
    n_sensors = len(raw_sensor_cols)
    ed_cfg = _make_ed_cfg_for_build(n_trials=4, n_sensors=n_sensors)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = run_study(x, y, raw_sensor_cols, ed_cfg, random_seed=5)

    record = build_best_record(
        study, ed_cfg, n_sensors=n_sensors, test_size=0.2, random_seed=5
    )

    prov = record["provenance"]
    assert isinstance(prov, dict)

    assert prov["n_sensors"] == n_sensors
    assert prov["n_trials"] == ed_cfg.n_trials
    assert prov["sampler_seed"] == ed_cfg.sampler_seed
    assert prov["inner_cv_folds"] == ed_cfg.inner_cv_folds
    assert prov["detection_metric"] == ed_cfg.detection_metric
    assert prov["alpha"] == pytest.approx(ed_cfg.alpha)
    assert prov["threshold_policy"] == ed_cfg.threshold_policy
    assert prov["test_size"] == pytest.approx(0.2)
    assert prov["random_seed"] == 5


# ---------------------------------------------------------------------------
# build_best_record — top-level fields
# ---------------------------------------------------------------------------


def test_build_best_record_top_level_fields_present() -> None:
    """build_best_record record contains best_trial_number, penalized_score, attrs."""
    from yield_risk.early_detection import build_best_record, run_study

    x, y, raw_sensor_cols = _make_build_data()
    n_sensors = len(raw_sensor_cols)
    ed_cfg = _make_ed_cfg_for_build(n_trials=4, n_sensors=n_sensors)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = run_study(x, y, raw_sensor_cols, ed_cfg, random_seed=5)

    record = build_best_record(
        study, ed_cfg, n_sensors=n_sensors, test_size=0.2, random_seed=5
    )

    assert "best_trial_number" in record
    assert record["best_trial_number"] == study.best_trial.number

    assert "penalized_score" in record
    assert record["penalized_score"] == pytest.approx(study.best_trial.value)

    # User attrs from best trial
    best_attrs = study.best_trial.user_attrs
    assert record["observation_fraction"] == pytest.approx(
        best_attrs["observation_fraction"]
    )
    assert record["n_features_selected"] == pytest.approx(
        best_attrs["n_features_selected"]
    )
    assert record["feasible"] == best_attrs["feasible"]
    assert record["latest_index"] == best_attrs["latest_index"]


# ---------------------------------------------------------------------------
# build_best_record — config round-trip
# ---------------------------------------------------------------------------


def test_build_best_record_config_round_trip() -> None:
    """Serialized config in record reconstructs to equal HyperparamConfig.

    Rebuilds via SensorAccess(**d["access"]) + HyperparamConfig(...) and asserts
    equality with suggest_config(study.best_trial, ed_cfg, n_sensors).
    """
    from yield_risk.early_detection import (
        build_best_record,
        run_study,
        suggest_config,
    )

    x, y, raw_sensor_cols = _make_build_data()
    n_sensors = len(raw_sensor_cols)
    ed_cfg = _make_ed_cfg_for_build(n_trials=4, n_sensors=n_sensors)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = run_study(x, y, raw_sensor_cols, ed_cfg, random_seed=5)

    record = build_best_record(
        study, ed_cfg, n_sensors=n_sensors, test_size=0.2, random_seed=5
    )

    cfg_d = record["config"]
    assert isinstance(cfg_d, dict)

    # Reconstruct manually
    access_reconstructed = SensorAccess(**cfg_d["access"])  # type: ignore[arg-type]
    cfg_reconstructed = HyperparamConfig(
        access=access_reconstructed,
        missing_threshold=cfg_d["missing_threshold"],  # type: ignore[arg-type]
        cv_threshold=cfg_d["cv_threshold"],  # type: ignore[arg-type]
        correlation_threshold=cfg_d["correlation_threshold"],  # type: ignore[arg-type]
        selection_method=cfg_d["selection_method"],  # type: ignore[arg-type]
        max_features=cfg_d["max_features"],  # type: ignore[arg-type]
        model_family=cfg_d["model_family"],  # type: ignore[arg-type]
        model_params=cfg_d["model_params"],  # type: ignore[arg-type]
        threshold_policy=cfg_d["threshold_policy"],  # type: ignore[arg-type]
        threshold=cfg_d["threshold"],  # type: ignore[arg-type]
        false_alarm_rate=cfg_d["false_alarm_rate"],  # type: ignore[arg-type]
    )

    # The winning config via suggest_config on the frozen best trial
    cfg_from_suggest = suggest_config(
        study.best_trial,  # type: ignore[arg-type]
        ed_cfg,
        n_sensors,
    )

    assert cfg_reconstructed == cfg_from_suggest


# ---------------------------------------------------------------------------
# build_best_record — nan → null serialization
# ---------------------------------------------------------------------------


def test_build_best_record_json_roundtrip_no_nan_tokens() -> None:
    """json.dumps(record) produces valid JSON; json.loads succeeds without error."""
    import json

    from yield_risk.early_detection import build_best_record, run_study

    x, y, raw_sensor_cols = _make_build_data()
    n_sensors = len(raw_sensor_cols)
    ed_cfg = _make_ed_cfg_for_build(n_trials=4, n_sensors=n_sensors)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = run_study(x, y, raw_sensor_cols, ed_cfg, random_seed=5)

    record = build_best_record(
        study, ed_cfg, n_sensors=n_sensors, test_size=0.2, random_seed=5
    )

    json_str = json.dumps(record)
    # Must not raise and must not contain the NaN token
    assert "NaN" not in json_str
    assert "Infinity" not in json_str
    reloaded = json.loads(json_str)
    assert isinstance(reloaded, dict)


def test_build_best_record_infeasible_study_detection_metric_null() -> None:
    """All-infeasible study: record['detection_metric'] serializes as null (None)."""
    import json

    from yield_risk.early_detection import build_best_record, run_study

    x, y, raw_sensor_cols = _make_build_data()
    n_sensors = len(raw_sensor_cols)

    # CV threshold so high all columns are dropped → all infeasible
    infeasible_ed_cfg = EarlyDetectionConfig(
        n_trials=3,
        sampler_seed=7,
        inner_cv_folds=2,
        detection_metric="pr_auc",
        alpha=0.1,
        access_types=["prefix"],
        min_window_size=2,
        max_window_size=n_sensors,
        model_families=["random_forest"],
        missing_threshold=(0.2, 0.9),
        cv_threshold=(1e10, 2e10),  # all columns dropped
        correlation_threshold=(0.85, 0.99),
        selection_methods=["none"],
        max_features=(2, n_sensors),
        threshold_policy="tune",
        threshold_range=(0.1, 0.8),
        false_alarm_rate=0.1,
        performance_tolerance=0.05,
        curve_prefixes=[16, 32],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = run_study(
            x, y, raw_sensor_cols, infeasible_ed_cfg, random_seed=5
        )

    # Must not raise
    record = build_best_record(
        study, infeasible_ed_cfg, n_sensors=n_sensors, test_size=0.2, random_seed=5
    )

    # detection_metric should be None (nan sanitized)
    assert record["detection_metric"] is None
    assert record["feasible"] is False

    # And the full JSON round-trip must succeed
    json_str = json.dumps(record)
    reloaded = json.loads(json_str)
    assert reloaded["detection_metric"] is None


def test_sanitize_nan_directly() -> None:
    """The nan→null sanitizer converts nan/inf scalars and nested structures."""
    import json

    from yield_risk.early_detection import build_best_record, run_study

    # We test nan sanitization indirectly: a record with a known nan field
    # must have that field as None after build_best_record.
    # Direct unit test of the sanitizer via a known-infeasible record:
    x, y, raw_sensor_cols = _make_build_data()
    n_sensors = len(raw_sensor_cols)

    infeasible_ed_cfg = EarlyDetectionConfig(
        n_trials=2,
        sampler_seed=7,
        inner_cv_folds=2,
        detection_metric="pr_auc",
        alpha=0.1,
        access_types=["prefix"],
        min_window_size=2,
        max_window_size=n_sensors,
        model_families=["random_forest"],
        missing_threshold=(0.2, 0.9),
        cv_threshold=(1e10, 2e10),
        correlation_threshold=(0.85, 0.99),
        selection_methods=["none"],
        max_features=(2, n_sensors),
        threshold_policy="tune",
        threshold_range=(0.1, 0.8),
        false_alarm_rate=0.1,
        performance_tolerance=0.05,
        curve_prefixes=[16, 32],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = run_study(
            x, y, raw_sensor_cols, infeasible_ed_cfg, random_seed=5
        )

    record = build_best_record(
        study, infeasible_ed_cfg, n_sensors=n_sensors, test_size=0.2, random_seed=5
    )

    # The raw detection_metric user attr is nan → must be None in the record
    raw_dm = study.best_trial.user_attrs["detection_metric"]
    assert math.isnan(raw_dm), "Expected nan detection_metric for all-infeasible study"
    assert record["detection_metric"] is None

    # Entire record must be valid JSON (no NaN tokens)
    json_str = json.dumps(record)
    assert "NaN" not in json_str
    json.loads(json_str)  # must not raise


# ---------------------------------------------------------------------------
# Core holdout fit/evaluate helpers
# ---------------------------------------------------------------------------


def _make_detector_data(
    n_rows: int = 80,
    n_sensors: int = 6,
    seed: int = 123,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return separable sensor data for fitted-detector tests.

    Args:
        n_rows: Number of samples.
        n_sensors: Number of sensor columns.
        seed: RNG seed.

    Returns:
        Tuple of feature frame, binary labels, and ordered raw sensor columns.
    """
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n_rows // 2), dtype=int)
    raw_sensor_cols = [f"sensor_{i:03d}" for i in range(n_sensors)]
    data = {
        col: rng.standard_normal(n_rows) + y * (1.5 if i == 0 else 0.15)
        for i, col in enumerate(raw_sensor_cols)
    }
    return pd.DataFrame(data), y, raw_sensor_cols


def _make_detector_cfg(
    *,
    threshold_policy: str = "tune",
    threshold: float | None = 0.5,
    false_alarm_rate: float | None = None,
    model_family: str = "logistic_regression",
    model_params: dict[str, object] | None = None,
    cv_threshold: float = 0.0,
) -> HyperparamConfig:
    """Return a HyperparamConfig for fitted-detector tests.

    Args:
        threshold_policy: Decision-threshold policy.
        threshold: Frozen threshold for the ``"tune"`` policy.
        false_alarm_rate: FAR target for the ``"far_constraint"`` policy.
        model_family: Estimator family.
        model_params: Optional estimator parameter overrides.
        cv_threshold: Coefficient-of-variation feature filter threshold.

    Returns:
        Hyperparameter configuration for the helper tests.
    """
    return HyperparamConfig(
        access=SensorAccess(access_type="prefix", prefix_end=4),
        missing_threshold=0.9,
        cv_threshold=cv_threshold,
        correlation_threshold=1.0,
        selection_method="none",
        max_features=None,
        model_family=model_family,
        model_params={} if model_params is None else model_params,
        threshold_policy=threshold_policy,
        threshold=threshold,
        false_alarm_rate=false_alarm_rate,
    )


def _manual_oof_threshold(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    raw_sensor_cols: list[str],
    cfg: HyperparamConfig,
    *,
    inner_cv_folds: int,
    random_seed: int,
) -> float:
    """Resolve a FAR threshold from public train-fold primitives.

    Args:
        x_train: Training frame.
        y_train: Training labels.
        raw_sensor_cols: Ordered raw sensor column names.
        cfg: Hyperparameter configuration.
        inner_cv_folds: Number of stratified folds.
        random_seed: Random state for splitting and estimators.

    Returns:
        Threshold resolved from concatenated out-of-fold train predictions.
    """
    from sklearn.model_selection import StratifiedKFold

    from yield_risk.early_detection import (
        FoldPreprocessor,
        build_estimator,
        resolve_threshold,
        select_ordered_sensors,
    )

    cv = StratifiedKFold(
        n_splits=inner_cv_folds, shuffle=True, random_state=random_seed
    )
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for train_idx, val_idx in cv.split(x_train, y_train):
        y_fold_train = y_train[train_idx]
        if len(np.unique(y_fold_train)) < 2:
            continue

        x_fold_train = x_train.iloc[train_idx]
        x_fold_val = x_train.iloc[val_idx]
        x_fold_train_window, _ = select_ordered_sensors(
            x_fold_train, raw_sensor_cols, cfg.access
        )
        x_fold_val_window, _ = select_ordered_sensors(
            x_fold_val, raw_sensor_cols, cfg.access
        )
        preprocessor = FoldPreprocessor.fit(
            x_fold_train_window, y_fold_train, cfg, random_seed
        )
        if not preprocessor.retained_cols or len(preprocessor.selected_idx) == 0:
            continue

        estimator = build_estimator(cfg.model_family, cfg.model_params, random_seed)
        estimator.fit(preprocessor.transform(x_fold_train_window), y_fold_train)
        proba_out = estimator.predict_proba(
            preprocessor.transform(x_fold_val_window)
        )
        if proba_out.shape[1] < 2:
            continue

        labels.append(y_train[val_idx])
        probabilities.append(proba_out[:, 1])

    assert labels, "test data must produce at least one valid OOF fold"
    return resolve_threshold(
        np.concatenate(labels), np.concatenate(probabilities), cfg
    )


def test_hyperparam_config_from_dict_round_trips_best_json_config() -> None:
    """hyperparam_config_from_dict reconstructs serialized best config data."""
    from yield_risk.early_detection import (
        hyperparam_config_from_dict,
        hyperparam_config_to_dict,
    )

    cfg = HyperparamConfig(
        access=SensorAccess(access_type="window", window_start=2, window_size=4),
        missing_threshold=0.4,
        cv_threshold=0.001,
        correlation_threshold=0.9,
        selection_method="univariate",
        max_features=8,
        model_family="xgboost",
        model_params={"learning_rate": 0.05, "n_estimators": 20},
        threshold_policy="far_constraint",
        threshold=None,
        false_alarm_rate=0.05,
    )

    assert hyperparam_config_from_dict(hyperparam_config_to_dict(cfg)) == cfg


def test_fit_early_detector_tune_uses_config_threshold() -> None:
    """fit_early_detector freezes the serialized tune threshold from cfg."""
    from yield_risk.early_detection import fit_early_detector

    x_train, y_train, raw_sensor_cols = _make_detector_data()
    cfg = _make_detector_cfg(threshold_policy="tune", threshold=0.73)

    detector = fit_early_detector(
        x_train,
        y_train,
        raw_sensor_cols,
        cfg,
        inner_cv_folds=3,
        random_seed=0,
    )

    assert detector.threshold == pytest.approx(0.73)


def test_fit_early_detector_far_constraint_uses_train_oof_threshold() -> None:
    """fit_early_detector resolves FAR threshold from train OOF predictions."""
    from yield_risk.early_detection import fit_early_detector

    x_train, y_train, raw_sensor_cols = _make_detector_data(n_rows=90)
    cfg = _make_detector_cfg(
        threshold_policy="far_constraint",
        threshold=None,
        false_alarm_rate=0.2,
    )
    expected_threshold = _manual_oof_threshold(
        x_train,
        y_train,
        raw_sensor_cols,
        cfg,
        inner_cv_folds=3,
        random_seed=4,
    )

    detector = fit_early_detector(
        x_train,
        y_train,
        raw_sensor_cols,
        cfg,
        inner_cv_folds=3,
        random_seed=4,
    )

    assert detector.threshold == pytest.approx(expected_threshold)


def test_fit_early_detector_empty_raw_sensor_cols_raises() -> None:
    """fit_early_detector raises the raw-sensor guard for empty sensor lists."""
    from yield_risk.early_detection import fit_early_detector

    x_train, y_train, _ = _make_detector_data()
    cfg = _make_detector_cfg()

    with pytest.raises(ValueError, match="No raw sensor_ columns found"):
        fit_early_detector(
            x_train,
            y_train,
            [],
            cfg,
            inner_cv_folds=3,
            random_seed=0,
        )


def test_fit_early_detector_zero_features_raises() -> None:
    """fit_early_detector rejects full-train preprocessing with zero features."""
    from yield_risk.early_detection import fit_early_detector

    x_train, y_train, raw_sensor_cols = _make_detector_data()
    cfg = _make_detector_cfg(cv_threshold=1e10)

    with pytest.raises(ValueError, match="Early detector retained zero features"):
        fit_early_detector(
            x_train,
            y_train,
            raw_sensor_cols,
            cfg,
            inner_cv_folds=3,
            random_seed=0,
        )


def test_fit_early_detector_far_constraint_no_valid_oof_raises() -> None:
    """fit_early_detector raises when FAR threshold has no valid OOF rows."""
    from yield_risk.early_detection import fit_early_detector

    x_train, y_train, raw_sensor_cols = _make_detector_data()
    cfg = _make_detector_cfg(
        threshold_policy="far_constraint",
        threshold=None,
        false_alarm_rate=0.1,
        cv_threshold=1e10,
    )

    with pytest.raises(
        ValueError, match="Cannot resolve final threshold from training data"
    ):
        fit_early_detector(
            x_train,
            y_train,
            raw_sensor_cols,
            cfg,
            inner_cv_folds=3,
            random_seed=0,
        )


def test_fit_early_detector_xgboost_injects_scale_pos_weight() -> None:
    """fit_early_detector computes xgboost scale_pos_weight from train labels."""
    from yield_risk.early_detection import fit_early_detector

    x_train, _, raw_sensor_cols = _make_detector_data(n_rows=60)
    y_train = np.array([0] * 45 + [1] * 15, dtype=int)
    cfg = _make_detector_cfg(
        model_family="xgboost",
        model_params={"n_estimators": 2, "max_depth": 2},
    )

    detector = fit_early_detector(
        x_train,
        y_train,
        raw_sensor_cols,
        cfg,
        inner_cv_folds=3,
        random_seed=0,
    )

    assert detector.estimator.get_params()["scale_pos_weight"] == pytest.approx(3.0)


def test_evaluate_fitted_early_detector_returns_json_safe_metrics() -> None:
    """evaluate_fitted_early_detector returns sanitized holdout metrics."""
    from yield_risk.config import CostMatrix
    from yield_risk.early_detection import (
        evaluate_fitted_early_detector,
        fit_early_detector,
    )

    x, y, raw_sensor_cols = _make_detector_data(n_rows=80)
    cfg = _make_detector_cfg(threshold_policy="tune", threshold=0.5)
    detector = fit_early_detector(
        x.iloc[:60],
        y[:60],
        raw_sensor_cols,
        cfg,
        inner_cv_folds=3,
        random_seed=0,
    )

    metrics = evaluate_fitted_early_detector(
        detector,
        x.iloc[60:],
        y[60:],
        cost_matrix=CostMatrix(
            true_pass=float("inf"),
            true_fail=float("inf"),
            false_fail=float("inf"),
            false_pass=float("inf"),
        ),
    )

    assert metrics["threshold"] == pytest.approx(detector.threshold)
    assert metrics["expected_cost"] is None
    assert metrics["n_sensors_used"] == 4
    assert metrics["sensor_fraction"] == pytest.approx(4 / len(raw_sensor_cols))
    json.dumps(metrics, allow_nan=False)


def test_evaluate_fitted_early_detector_single_class_test_raises() -> None:
    """evaluate_fitted_early_detector rejects single-class test holdouts."""
    from yield_risk.early_detection import (
        evaluate_fitted_early_detector,
        fit_early_detector,
    )

    x, y, raw_sensor_cols = _make_detector_data(n_rows=80)
    cfg = _make_detector_cfg(threshold_policy="tune", threshold=0.5)
    detector = fit_early_detector(
        x.iloc[:60],
        y[:60],
        raw_sensor_cols,
        cfg,
        inner_cv_folds=3,
        random_seed=0,
    )

    with pytest.raises(ValueError, match="Test holdout must contain both classes"):
        evaluate_fitted_early_detector(detector, x.iloc[60:], np.zeros(20, dtype=int))


def test_evaluate_fitted_early_detector_one_column_proba_raises() -> None:
    """evaluate_fitted_early_detector rejects one-column predict_proba output."""
    from sklearn.dummy import DummyClassifier

    from yield_risk.early_detection import (
        FittedEarlyDetector,
        FoldPreprocessor,
        evaluate_fitted_early_detector,
        select_ordered_sensors,
    )

    x, y, raw_sensor_cols = _make_detector_data(n_rows=80)
    cfg = _make_detector_cfg(threshold_policy="tune", threshold=0.5)
    x_train_window, window_cols = select_ordered_sensors(
        x.iloc[:60], raw_sensor_cols, cfg.access
    )
    preprocessor = FoldPreprocessor.fit(x_train_window, y[:60], cfg, random_seed=0)
    x_train_arr = preprocessor.transform(x_train_window)
    estimator = DummyClassifier(strategy="most_frequent")
    estimator.fit(x_train_arr, np.zeros(len(x_train_arr), dtype=int))
    detector = FittedEarlyDetector(
        config=cfg,
        preprocessor=preprocessor,
        estimator=estimator,
        threshold=0.5,
        raw_sensor_cols=raw_sensor_cols,
        window_cols=window_cols,
    )

    with pytest.raises(
        ValueError, match="Estimator did not produce positive-class probabilities"
    ):
        evaluate_fitted_early_detector(detector, x.iloc[60:], y[60:])


# ---------------------------------------------------------------------------
# Holdout comparison orchestration
# ---------------------------------------------------------------------------


def _make_holdout_df(
    *,
    n_rows: int = 80,
    n_sensors: int = 5,
    early_signal: bool = True,
    seed: int = 321,
) -> pd.DataFrame:
    """Return a SECOM-shaped DataFrame for holdout orchestration tests.

    Args:
        n_rows: Number of rows to generate.
        n_sensors: Number of raw ``sensor_*`` columns.
        early_signal: Whether the first sensor should be strongly predictive.
            When false, a later sensor is strongly predictive instead.
        seed: Random-number seed.

    Returns:
        DataFrame containing ``timestamp``, ordered sensor columns, and
        binary ``label``.
    """
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n_rows // 2), dtype=int)
    data: dict[str, object] = {"timestamp": list(range(n_rows))}
    for i in range(n_sensors):
        signal = 0.0
        if early_signal and i == 0:
            signal = 3.0
        if not early_signal and i == n_sensors - 1:
            signal = 4.0
        data[f"sensor_{i:03d}"] = rng.standard_normal(n_rows) + y * signal
    data["label"] = y
    return pd.DataFrame(data)


def _make_holdout_ed_cfg(
    *,
    curve_prefixes: list[int] | None = None,
    performance_tolerance: float = 0.05,
) -> EarlyDetectionConfig:
    """Return a compact early-detection config for holdout tests.

    Args:
        curve_prefixes: Prefixes to request for the diagnostic curve.
        performance_tolerance: Fractional PR-AUC tolerance for the verdict.

    Returns:
        Early-detection configuration with fast logistic-regression settings.
    """
    return EarlyDetectionConfig(
        n_trials=1,
        sampler_seed=0,
        inner_cv_folds=2,
        detection_metric="pr_auc",
        alpha=0.1,
        access_types=["prefix"],
        min_window_size=1,
        max_window_size=5,
        model_families=["logistic_regression"],
        missing_threshold=(0.9, 0.9),
        cv_threshold=(0.0, 0.0),
        correlation_threshold=(1.0, 1.0),
        selection_methods=["none"],
        max_features=(1, 5),
        threshold_policy="tune",
        threshold_range=(0.5, 0.5),
        false_alarm_rate=0.1,
        performance_tolerance=performance_tolerance,
        curve_prefixes=[] if curve_prefixes is None else curve_prefixes,
    )


def _make_holdout_best_record(
    *,
    n_sensors: int = 5,
    prefix_end: int = 2,
    feasible: bool = True,
    test_size: float = 0.25,
    random_seed: int = 7,
) -> dict[str, object]:
    """Return a minimal serialized best-trial record for holdout tests.

    Args:
        n_sensors: Raw sensor count recorded in provenance.
        prefix_end: Winning early prefix length.
        feasible: Serialized trial feasibility flag.
        test_size: Holdout size recorded in provenance.
        random_seed: Split seed recorded in provenance.

    Returns:
        Best-record mapping shaped like ``early_detection_best.json``.
    """
    from yield_risk.early_detection import hyperparam_config_to_dict

    cfg = HyperparamConfig(
        access=SensorAccess(access_type="prefix", prefix_end=prefix_end),
        missing_threshold=0.9,
        cv_threshold=0.0,
        correlation_threshold=1.0,
        selection_method="none",
        max_features=None,
        model_family="logistic_regression",
        model_params={},
        threshold_policy="tune",
        threshold=0.5,
        false_alarm_rate=None,
    )
    return {
        "best_trial_number": 0,
        "penalized_score": 0.1,
        "detection_metric": 0.2,
        "observation_fraction": prefix_end / n_sensors,
        "n_features_selected": float(prefix_end),
        "feasible": feasible,
        "latest_index": prefix_end,
        "config": hyperparam_config_to_dict(cfg),
        "provenance": {
            "n_sensors": n_sensors,
            "n_trials": 1,
            "sampler_seed": 0,
            "inner_cv_folds": 2,
            "detection_metric": "pr_auc",
            "alpha": 0.1,
            "threshold_policy": "tune",
            "test_size": test_size,
            "random_seed": random_seed,
        },
    }


def _holdout_cost_matrix() -> object:
    """Return a finite cost matrix for holdout orchestration tests.

    Returns:
        Cost matrix object accepted by early-detection evaluators.
    """
    from yield_risk.config import CostMatrix

    return CostMatrix(
        true_pass=0.0,
        true_fail=0.0,
        false_fail=1.0,
        false_pass=10.0,
    )


def test_comparison_schema_includes_early_full_tabular_rows(
    tmp_path: Path,
) -> None:
    """evaluate_best_on_holdout returns required comparison row fields."""
    from yield_risk.early_detection import evaluate_best_on_holdout

    model_comparison_path = tmp_path / "model_comparison.json"
    model_comparison_path.write_text(
        json.dumps(
            [
                {
                    "model": "logistic_regression",
                    "test_pr_auc": 0.2,
                    "selected": False,
                },
                {
                    "model": "random_forest",
                    "test_pr_auc": 0.91,
                    "test_roc_auc": 0.88,
                    "test_precision": 0.31,
                    "test_recall": 0.72,
                    "frozen_threshold": 0.14,
                    "expected_cost": 83.0,
                    "selected": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    result = evaluate_best_on_holdout(
        _make_holdout_df(),
        _make_holdout_best_record(prefix_end=2),
        _make_holdout_ed_cfg(curve_prefixes=[1]),
        _holdout_cost_matrix(),
        model_comparison_path=model_comparison_path,
    )

    json.dumps(result, allow_nan=False)
    assert result["primary_metric"] == "test_pr_auc"
    assert result["performance_tolerance"] == pytest.approx(0.05)
    assert result["sensor_count"] == 5
    assert result["test_size"] == pytest.approx(0.25)
    assert result["random_seed"] == 7

    comparison_rows = result["comparison_rows"]
    assert isinstance(comparison_rows, list)
    by_role = {row["role"]: row for row in comparison_rows}
    assert set(by_role) == {
        "early_optuna_best",
        "full_prefix_same_config",
        "tabular_selected_model",
    }

    required_fields = {
        "model",
        "role",
        "test_pr_auc",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_balanced_accuracy",
        "test_false_alarm_rate",
        "expected_cost",
        "threshold",
        "confusion_matrix",
        "n_sensors_used",
        "sensor_fraction",
        "latest_index",
        "source",
    }
    for row in comparison_rows:
        assert required_fields <= row.keys()

    early_row = by_role["early_optuna_best"]
    assert early_row["n_sensors_used"] == 2
    assert early_row["sensor_fraction"] == pytest.approx(2 / 5)
    assert early_row["latest_index"] == 2

    full_row = by_role["full_prefix_same_config"]
    assert full_row["n_sensors_used"] == 5
    assert full_row["sensor_fraction"] == pytest.approx(1.0)
    assert full_row["latest_index"] == 5

    tabular_row = by_role["tabular_selected_model"]
    assert tabular_row["model"] == "random_forest"
    assert tabular_row["threshold"] == pytest.approx(0.14)
    assert tabular_row["test_f1"] is None
    assert tabular_row["test_balanced_accuracy"] is None
    assert tabular_row["test_false_alarm_rate"] is None
    assert tabular_row["confusion_matrix"] is None
    assert tabular_row["n_sensors_used"] == 5
    assert tabular_row["sensor_fraction"] == pytest.approx(1.0)
    assert tabular_row["latest_index"] == 5
    assert tabular_row["source"] == str(model_comparison_path)


def test_verdict_marks_competitive_early_model() -> None:
    """evaluate_best_on_holdout chooses the competitive early verdict."""
    from yield_risk.early_detection import evaluate_best_on_holdout

    result = evaluate_best_on_holdout(
        _make_holdout_df(early_signal=True),
        _make_holdout_best_record(prefix_end=1),
        _make_holdout_ed_cfg(curve_prefixes=[1], performance_tolerance=0.10),
        _holdout_cost_matrix(),
    )

    assert result["verdict"] == "early_model_competitive"


def test_verdict_prefers_baseline_when_early_model_lags() -> None:
    """evaluate_best_on_holdout chooses baseline_preferred for weak early data."""
    from yield_risk.early_detection import evaluate_best_on_holdout

    result = evaluate_best_on_holdout(
        _make_holdout_df(early_signal=False),
        _make_holdout_best_record(prefix_end=1),
        _make_holdout_ed_cfg(curve_prefixes=[1], performance_tolerance=0.05),
        _holdout_cost_matrix(),
    )

    assert result["verdict"] == "baseline_preferred"


def test_curve_prefix_rows_use_unique_sorted_valid_prefixes() -> None:
    """evaluate_best_on_holdout normalizes diagnostic curve prefixes."""
    from yield_risk.early_detection import evaluate_best_on_holdout

    result = evaluate_best_on_holdout(
        _make_holdout_df(),
        _make_holdout_best_record(prefix_end=3),
        _make_holdout_ed_cfg(curve_prefixes=[0, 2, 2, 10, -1]),
        _holdout_cost_matrix(),
    )

    curve_rows = result["curve_rows"]
    assert isinstance(curve_rows, list)
    assert [row["latest_index"] for row in curve_rows] == [2, 3, 5]
    assert [row["n_sensors_used"] for row in curve_rows] == [2, 3, 5]
    assert {row["role"] for row in curve_rows} == {"early_detection_curve"}


def test_evaluate_best_on_holdout_rejects_infeasible_best_trial() -> None:
    """evaluate_best_on_holdout rejects serialized infeasible winners."""
    from yield_risk.early_detection import evaluate_best_on_holdout

    with pytest.raises(
        ValueError, match="Cannot evaluate an infeasible early-detection best trial"
    ):
        evaluate_best_on_holdout(
            _make_holdout_df(),
            _make_holdout_best_record(feasible=False),
            _make_holdout_ed_cfg(curve_prefixes=[1]),
            _holdout_cost_matrix(),
        )


def test_evaluate_best_on_holdout_rejects_sensor_count_mismatch() -> None:
    """evaluate_best_on_holdout rejects stale best-record sensor counts."""
    from yield_risk.early_detection import evaluate_best_on_holdout

    with pytest.raises(
        ValueError, match="early_detection_best.json n_sensors does not match data"
    ):
        evaluate_best_on_holdout(
            _make_holdout_df(n_sensors=5),
            _make_holdout_best_record(n_sensors=6),
            _make_holdout_ed_cfg(curve_prefixes=[1]),
            _holdout_cost_matrix(),
        )


def test_evaluate_best_on_holdout_sanitizes_json_result(
    tmp_path: Path,
) -> None:
    """evaluate_best_on_holdout strips non-finite values from its result."""
    from yield_risk.config import CostMatrix
    from yield_risk.early_detection import evaluate_best_on_holdout

    model_comparison_path = tmp_path / "model_comparison.json"
    model_comparison_path.write_text(
        json.dumps(
            [
                {
                    "model": "random_forest",
                    "test_pr_auc": 0.91,
                    "test_f1": float("nan"),
                    "selected": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = evaluate_best_on_holdout(
        _make_holdout_df(),
        _make_holdout_best_record(prefix_end=2),
        _make_holdout_ed_cfg(curve_prefixes=[1]),
        CostMatrix(
            true_pass=float("inf"),
            true_fail=float("inf"),
            false_fail=float("inf"),
            false_pass=float("inf"),
        ),
        model_comparison_path=model_comparison_path,
    )

    json.dumps(result, allow_nan=False)
    tabular_rows = [
        row for row in result["comparison_rows"]
        if row["role"] == "tabular_selected_model"
    ]
    assert tabular_rows[0]["test_f1"] is None


# ---------------------------------------------------------------------------
# CLI tests — run_early_detection.py  (Spec §5 / §6)
# ---------------------------------------------------------------------------


def _load_cli() -> types.ModuleType:
    """Load scripts/run_early_detection.py as a module by file path.

    Returns:
        The loaded module object.
    """
    script_path = (
        Path(__file__).parent.parent / "scripts" / "run_early_detection.py"
    )
    spec = importlib.util.spec_from_file_location("run_early_detection", script_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _tiny_secom(n_sensors: int = 6, n_rows: int = 60) -> pd.DataFrame:
    """Return a tiny SECOM-shaped DataFrame for fast tests.

    Args:
        n_sensors: Number of ``sensor_*`` columns to include.
        n_rows: Total number of rows.

    Returns:
        DataFrame with columns ``timestamp``, ``sensor_000``..., ``label``.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    data: dict[str, object] = {"timestamp": list(range(n_rows))}
    for i in range(n_sensors):
        data[f"sensor_{i:03d}"] = rng.standard_normal(n_rows).tolist()
    labels = [0] * (n_rows // 2) + [1] * (n_rows - n_rows // 2)
    data["label"] = labels
    return pd.DataFrame(data)


def _fake_config(tmp_path: Path) -> object:
    """Build a fake Config whose paths point at tmp_path.

    Args:
        tmp_path: Temporary directory for models/reports.

    Returns:
        A Config-like object with .paths.models_dir, .paths.reports_dir,
        .paths.raw_dir, .run.test_size, .run.random_seed.
    """
    from yield_risk.config import Config, PathsConfig, RunConfig

    paths = PathsConfig(
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        splits_dir=tmp_path / "splits",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
    )
    run = RunConfig(
        random_seed=0,
        test_size=0.2,
        cv_folds=2,
        missing_threshold=0.5,
        cv_threshold=1e-6,
        correlation_threshold=0.95,
    )
    return Config(paths=paths, run=run)


def _write_minimal_ed_yaml(tmp_path: Path) -> Path:
    """Write a minimal early_detection config YAML to tmp_path.

    Uses 2 trials, 2 CV folds, logistic_regression only for speed.

    Args:
        tmp_path: Directory to write the YAML.

    Returns:
        Path to the written YAML file.
    """
    yaml_path = tmp_path / "ed_cfg.yaml"
    yaml_path.write_text(
        "n_trials: 2\n"
        "inner_cv_folds: 2\n"
        "model_families:\n"
        "  - logistic_regression\n"
        "access_types:\n"
        "  - prefix\n"
        "selection_methods:\n"
        "  - none\n",
        encoding="utf-8",
    )
    return yaml_path


def _fake_best_record_for_cli() -> dict[str, object]:
    """Return a minimal best-record payload for CLI orchestration tests.

    Returns:
        JSON-safe best-record mapping with provenance required by the CLI.
    """
    return {
        "best_trial_number": 1,
        "penalized_score": 0.12,
        "detection_metric": 0.2,
        "observation_fraction": 0.5,
        "n_features_selected": 2.0,
        "feasible": True,
        "latest_index": 3,
        "config": {"model_family": "random_forest"},
        "provenance": {
            "n_sensors": 6,
            "n_trials": 2,
            "sampler_seed": 0,
            "inner_cv_folds": 2,
            "detection_metric": "pr_auc",
            "alpha": 0.1,
            "threshold_policy": "tune",
            "test_size": 0.2,
            "random_seed": 0,
        },
    }


def _fake_holdout_result() -> dict[str, object]:
    """Return a JSON-safe holdout result shaped like Spec C output.

    Returns:
        Result mapping with comparison rows and diagnostic curve rows.
    """
    required_base: dict[str, object] = {
        "test_roc_auc": 0.7,
        "test_precision": 0.3,
        "test_recall": 0.4,
        "test_f1": 0.34,
        "test_balanced_accuracy": 0.6,
        "test_false_alarm_rate": 0.2,
        "expected_cost": 10.0,
        "threshold": 0.25,
        "confusion_matrix": [[8, 2], [3, 4]],
        "source": "early_detection_best.json",
    }
    early_row = {
        **required_base,
        "model": "random_forest",
        "role": "early_optuna_best",
        "test_pr_auc": 0.21,
        "n_sensors_used": 3,
        "sensor_fraction": 0.5,
        "latest_index": 3,
    }
    full_row = {
        **required_base,
        "model": "random_forest",
        "role": "full_prefix_same_config",
        "test_pr_auc": 0.23,
        "n_sensors_used": 6,
        "sensor_fraction": 1.0,
        "latest_index": 6,
    }
    tabular_row = {
        **required_base,
        "model": "xgboost",
        "role": "tabular_selected_model",
        "test_pr_auc": 0.22,
        "n_sensors_used": 6,
        "sensor_fraction": 1.0,
        "latest_index": 6,
        "source": "reports/model_comparison.json",
    }
    curve_rows = [
        {
            **required_base,
            "model": "random_forest",
            "role": "early_detection_curve",
            "test_pr_auc": 0.18,
            "n_sensors_used": 1,
            "sensor_fraction": 1 / 6,
            "latest_index": 1,
        },
        {
            **required_base,
            "model": "random_forest",
            "role": "early_detection_curve",
            "test_pr_auc": 0.21,
            "n_sensors_used": 3,
            "sensor_fraction": 0.5,
            "latest_index": 3,
        },
        {
            **required_base,
            "model": "random_forest",
            "role": "early_detection_curve",
            "test_pr_auc": 0.23,
            "n_sensors_used": 6,
            "sensor_fraction": 1.0,
            "latest_index": 6,
        },
    ]
    return {
        "primary_metric": "test_pr_auc",
        "performance_tolerance": 0.05,
        "sensor_count": 6,
        "test_size": 0.2,
        "random_seed": 0,
        "verdict": "baseline_preferred",
        "comparison_rows": [early_row, full_row, tabular_row],
        "curve_rows": curve_rows,
    }


class _FakeTrial:
    """Small Optuna-trial stand-in for CLI orchestration tests."""

    number = 1
    user_attrs = {"model_family": "random_forest"}


class _FakeStudy:
    """Small Optuna-study stand-in for CLI orchestration tests."""

    best_trial = _FakeTrial()

    def trials_dataframe(self) -> pd.DataFrame:
        """Return one fake completed-trial row.

        Returns:
            DataFrame shaped like an Optuna trials export.
        """
        return pd.DataFrame({"number": [1], "value": [0.12]})


def _assert_spec_c_artifacts_exist(
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    """Assert all namespaced Spec C artifacts exist.

    Args:
        reports_dir: Configured reports directory.
        figures_dir: Configured figures directory.
    """
    assert (reports_dir / "early_detection_metrics.json").exists()
    assert (reports_dir / "early_detection_comparison.json").exists()
    assert (reports_dir / "early_detection_comparison.csv").exists()
    assert (reports_dir / "early_detection_curve.csv").exists()
    assert (figures_dir / "early_detection_curve.png").exists()


# ---------------------------------------------------------------------------
# Test: _build_overrides only includes user-supplied flags
# ---------------------------------------------------------------------------


def test_build_overrides_only_supplied_flags() -> None:
    """_build_overrides returns only the keys the user actually passed."""
    import argparse

    mod = _load_cli()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--detection-metric", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args(["--n-trials", "10", "--alpha", "0.2"])
    overrides = mod._build_overrides(args)

    assert overrides.get("n_trials") == 10
    assert overrides.get("alpha") == pytest.approx(0.2)
    # seed -> sampler_seed; not supplied -> None (loader will drop it)
    assert overrides.get("sampler_seed") is None
    assert overrides.get("detection_metric") is None


def test_build_overrides_seed_maps_to_sampler_seed() -> None:
    """_build_overrides maps --seed to the sampler_seed key."""
    import argparse

    mod = _load_cli()

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--detection-metric", type=str, default=None)

    args = parser.parse_args(["--seed", "99"])
    overrides = mod._build_overrides(args)

    assert overrides.get("sampler_seed") == 99


# ---------------------------------------------------------------------------
# Tests: Spec C artifact writing and --evaluate-existing
# ---------------------------------------------------------------------------


def test_spec_c_artifact_writer_outputs_documented_schema(tmp_path: Path) -> None:
    """Spec C writer emits JSON, CSV, curve CSV, and PNG under configured paths."""
    mod = _load_cli()
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures"

    mod._write_spec_c_artifacts(_fake_holdout_result(), reports_dir, figures_dir)

    _assert_spec_c_artifacts_exist(reports_dir, figures_dir)
    comparison = json.loads(
        (reports_dir / "early_detection_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert comparison["primary_metric"] == "test_pr_auc"
    assert comparison["verdict"] == "baseline_preferred"
    assert {row["role"] for row in comparison["comparison_rows"]} == {
        "early_optuna_best",
        "full_prefix_same_config",
        "tabular_selected_model",
    }

    comparison_csv = pd.read_csv(reports_dir / "early_detection_comparison.csv")
    required_columns = {
        "model",
        "role",
        "test_pr_auc",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_balanced_accuracy",
        "test_false_alarm_rate",
        "expected_cost",
        "threshold",
        "confusion_matrix",
        "n_sensors_used",
        "sensor_fraction",
        "latest_index",
        "source",
    }
    assert required_columns <= set(comparison_csv.columns)
    assert set(comparison_csv["role"]) == {
        "early_optuna_best",
        "full_prefix_same_config",
        "tabular_selected_model",
    }

    curve_csv = pd.read_csv(reports_dir / "early_detection_curve.csv")
    assert list(curve_csv["latest_index"]) == [1, 3, 6]
    png_bytes = (figures_dir / "early_detection_curve.png").read_bytes()
    assert png_bytes.startswith(b"\x89PNG")


def test_evaluate_existing_skips_optuna_and_writes_spec_c_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--evaluate-existing loads best.json, skips study artifacts, and writes Spec C."""
    from yield_risk.config import CostMatrix

    mod = _load_cli()
    df = _tiny_secom()
    fake_cfg = _fake_config(tmp_path)
    yaml_path = _write_minimal_ed_yaml(tmp_path)
    expected_cost_matrix = CostMatrix(
        true_pass=0.0,
        true_fail=0.0,
        false_fail=1.0,
        false_pass=10.0,
    )
    fake_cfg.paths.models_dir.mkdir(parents=True)
    best_path = fake_cfg.paths.models_dir / "early_detection_best.json"
    best_payload = _fake_best_record_for_cli()
    best_path.write_text(json.dumps(best_payload), encoding="utf-8")

    monkeypatch.setattr(mod, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(
        mod,
        "load_cost_config",
        lambda: types.SimpleNamespace(cost_matrix=expected_cost_matrix),
    )
    monkeypatch.setattr(mod, "load_secom", lambda raw_dir: df)
    monkeypatch.setattr(mod, "validate_secom", lambda df_: None)
    monkeypatch.setattr(
        mod,
        "run_study",
        lambda *args, **kwargs: pytest.fail("run_study should be skipped"),
    )
    monkeypatch.setattr(
        mod,
        "build_best_record",
        lambda *args, **kwargs: pytest.fail("build_best_record should be skipped"),
    )
    monkeypatch.setattr(
        mod.joblib,
        "dump",
        lambda *args, **kwargs: pytest.fail("study dump should be skipped"),
    )

    calls: list[Path | None] = []

    def fake_evaluate_best_on_holdout(
        df_: pd.DataFrame,
        best_record: dict[str, object],
        ed_cfg: EarlyDetectionConfig,
        cost_matrix: object,
        *,
        model_comparison_path: Path | None = None,
    ) -> dict[str, object]:
        """Record the model-comparison path and return fake Spec C results."""
        assert df_ is df
        assert best_record == best_payload
        assert ed_cfg.n_trials == 2
        assert cost_matrix is expected_cost_matrix
        calls.append(model_comparison_path)
        return _fake_holdout_result()

    monkeypatch.setattr(
        mod,
        "evaluate_best_on_holdout",
        fake_evaluate_best_on_holdout,
        raising=False,
    )

    mod.main(["--config", str(yaml_path), "--evaluate-existing"])

    assert calls == [fake_cfg.paths.reports_dir / "model_comparison.json"]
    _assert_spec_c_artifacts_exist(
        fake_cfg.paths.reports_dir,
        fake_cfg.paths.figures_dir,
    )
    assert not (fake_cfg.paths.models_dir / "early_detection_study.pkl").exists()
    assert not (fake_cfg.paths.reports_dir / "early_detection_trials.csv").exists()


def test_evaluate_existing_missing_best_raises_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--evaluate-existing raises FileNotFoundError mentioning best.json path."""
    mod = _load_cli()
    fake_cfg = _fake_config(tmp_path)
    yaml_path = _write_minimal_ed_yaml(tmp_path)

    monkeypatch.setattr(mod, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(
        mod,
        "load_secom",
        lambda raw_dir: pytest.fail("best artifact should be checked first"),
    )
    monkeypatch.setattr(mod, "validate_secom", lambda df_: None)
    monkeypatch.setattr(
        mod,
        "run_study",
        lambda *args, **kwargs: pytest.fail("run_study should be skipped"),
    )

    expected_path = fake_cfg.paths.models_dir / "early_detection_best.json"
    match_path = str(expected_path).replace("\\", "\\\\")
    with pytest.raises(FileNotFoundError, match=match_path):
        mod.main(["--config", str(yaml_path), "--evaluate-existing"])


def test_default_path_runs_study_and_writes_spec_c_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default CLI path writes Spec B artifacts, then evaluates and writes Spec C."""
    from yield_risk.config import CostMatrix

    mod = _load_cli()
    df = _tiny_secom()
    fake_cfg = _fake_config(tmp_path)
    yaml_path = _write_minimal_ed_yaml(tmp_path)
    best_payload = _fake_best_record_for_cli()
    expected_cost_matrix = CostMatrix(
        true_pass=0.0,
        true_fail=0.0,
        false_fail=1.0,
        false_pass=10.0,
    )

    monkeypatch.setattr(mod, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(
        mod,
        "load_cost_config",
        lambda: types.SimpleNamespace(cost_matrix=expected_cost_matrix),
    )
    monkeypatch.setattr(mod, "load_secom", lambda raw_dir: df)
    monkeypatch.setattr(mod, "validate_secom", lambda df_: None)

    study_calls = 0

    def fake_run_study(*args: object, **kwargs: object) -> _FakeStudy:
        """Return a fake study and record that the study path executed."""
        nonlocal study_calls
        study_calls += 1
        assert kwargs["cost_matrix"] is None
        return _FakeStudy()

    dump_calls: list[Path] = []

    def fake_dump(study: _FakeStudy, path: Path) -> None:
        """Write a placeholder study file."""
        dump_calls.append(path)
        path.write_text("study", encoding="utf-8")

    eval_calls: list[Path | None] = []

    def fake_evaluate_best_on_holdout(
        df_: pd.DataFrame,
        best_record: dict[str, object],
        ed_cfg: EarlyDetectionConfig,
        cost_matrix: object,
        *,
        model_comparison_path: Path | None = None,
    ) -> dict[str, object]:
        """Record the model-comparison path and return fake Spec C results."""
        assert df_ is df
        assert best_record == best_payload
        assert ed_cfg.n_trials == 2
        assert cost_matrix is expected_cost_matrix
        eval_calls.append(model_comparison_path)
        return _fake_holdout_result()

    monkeypatch.setattr(mod, "run_study", fake_run_study)
    monkeypatch.setattr(mod.joblib, "dump", fake_dump)
    monkeypatch.setattr(mod, "build_best_record", lambda *args, **kwargs: best_payload)
    monkeypatch.setattr(
        mod,
        "evaluate_best_on_holdout",
        fake_evaluate_best_on_holdout,
        raising=False,
    )

    mod.main(["--config", str(yaml_path), "--n-trials", "2"])

    assert study_calls == 1
    assert dump_calls == [fake_cfg.paths.models_dir / "early_detection_study.pkl"]
    assert eval_calls == [fake_cfg.paths.reports_dir / "model_comparison.json"]
    assert (fake_cfg.paths.models_dir / "early_detection_best.json").exists()
    assert (fake_cfg.paths.reports_dir / "early_detection_trials.csv").exists()
    _assert_spec_c_artifacts_exist(
        fake_cfg.paths.reports_dir,
        fake_cfg.paths.figures_dir,
    )


# ---------------------------------------------------------------------------
# Test: artifacts written by main() with monkeypatching
# ---------------------------------------------------------------------------


def test_main_writes_three_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() writes study.pkl, best.json, and trials.csv to configured dirs."""
    mod = _load_cli()

    df = _tiny_secom()
    fake_cfg = _fake_config(tmp_path)
    yaml_path = _write_minimal_ed_yaml(tmp_path)

    monkeypatch.setattr(mod, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(mod, "load_secom", lambda raw_dir: df)
    monkeypatch.setattr(mod, "validate_secom", lambda df_: None)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod.main(["--config", str(yaml_path), "--n-trials", "2"])

    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    assert (models_dir / "early_detection_study.pkl").exists()
    assert (models_dir / "early_detection_best.json").exists()
    assert (reports_dir / "early_detection_trials.csv").exists()


def test_main_best_json_is_valid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """early_detection_best.json written by main() is valid JSON."""
    mod = _load_cli()

    df = _tiny_secom()
    fake_cfg = _fake_config(tmp_path)
    yaml_path = _write_minimal_ed_yaml(tmp_path)

    monkeypatch.setattr(mod, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(mod, "load_secom", lambda raw_dir: df)
    monkeypatch.setattr(mod, "validate_secom", lambda df_: None)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod.main(["--config", str(yaml_path), "--n-trials", "2"])

    best_json_path = tmp_path / "models" / "early_detection_best.json"
    content = best_json_path.read_text(encoding="utf-8")
    parsed = json.loads(content)  # must not raise
    assert "best_trial_number" in parsed


def test_main_trials_csv_has_correct_row_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """trials CSV has one row per trial (header excluded)."""
    mod = _load_cli()

    df = _tiny_secom()
    fake_cfg = _fake_config(tmp_path)
    yaml_path = _write_minimal_ed_yaml(tmp_path)

    monkeypatch.setattr(mod, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(mod, "load_secom", lambda raw_dir: df)
    monkeypatch.setattr(mod, "validate_secom", lambda df_: None)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod.main(["--config", str(yaml_path), "--n-trials", "2"])

    csv_path = tmp_path / "reports" / "early_detection_trials.csv"
    trials_df = pd.read_csv(csv_path)
    assert len(trials_df) == 2  # n_trials == 2


# ---------------------------------------------------------------------------
# Test: n_sensors == 0 guard exits with code 1
# ---------------------------------------------------------------------------


def test_main_no_sensor_columns_exits_with_code_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() exits with code 1 when no sensor_ columns are present."""
    mod = _load_cli()

    # DataFrame with no sensor_ columns
    df_no_sensors = pd.DataFrame(
        {"timestamp": range(20), "label": [0] * 10 + [1] * 10}
    )
    fake_cfg = _fake_config(tmp_path)
    yaml_path = _write_minimal_ed_yaml(tmp_path)

    monkeypatch.setattr(mod, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(mod, "load_secom", lambda raw_dir: df_no_sensors)
    monkeypatch.setattr(mod, "validate_secom", lambda df_: None)

    with pytest.raises(SystemExit) as exc_info:
        mod.main(["--config", str(yaml_path)])

    assert exc_info.value.code == 1
