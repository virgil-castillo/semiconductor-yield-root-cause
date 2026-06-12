"""Tests for early_detection sensor access primitives."""

from __future__ import annotations

import pandas as pd
import pytest

from yield_risk.early_detection import (
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
