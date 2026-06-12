"""Tests for early_detection sensor access primitives."""

from __future__ import annotations

import pandas as pd
import pytest

from yield_risk.early_detection import (
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
        variance_threshold=0.01,
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
        variance_threshold=0.005,
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
    assert cfg.variance_threshold == 0.005
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

import numpy as np  # noqa: E402


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
    variance_threshold: float = 0.0,
    correlation_threshold: float = 1.0,
    model_family: str = "random_forest",
) -> HyperparamConfig:
    """Return a HyperparamConfig for FoldPreprocessor tests."""
    return HyperparamConfig(
        access=SensorAccess(access_type="prefix", prefix_end=1),
        missing_threshold=missing_threshold,
        variance_threshold=variance_threshold,
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
# FoldPreprocessor — test 2: zero-variance scale → 1.0
# ---------------------------------------------------------------------------


def test_fold_preprocessor_zero_variance_scale_is_one() -> None:
    """A constant column on the train fold gets scaler_scale entry of 1.0.

    Transforming that column must produce finite values (no inf/nan).
    """
    from yield_risk.early_detection import FoldPreprocessor

    rng = np.random.default_rng(99)
    n = 20
    data = {
        "sensor_0": rng.standard_normal(n),
        "sensor_1": np.full(n, 3.14),  # constant → zero variance
        "sensor_2": rng.standard_normal(n),
    }
    df = pd.DataFrame(data)
    y = (rng.random(n) > 0.5).astype(int)

    cfg = _make_fold_cfg(selection_method="none", variance_threshold=0.0)
    fp = FoldPreprocessor.fit(df, y, cfg, random_seed=0)

    # sensor_1 should survive (variance == 0 is NOT strictly below 0.0)
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
