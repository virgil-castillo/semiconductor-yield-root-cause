"""Tests for data loading (load_secom) and schema validation (validate_secom)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yield_risk.data import load_secom
from yield_risk.validation import validate_secom


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    """5-row x 3-sensor synthetic SECOM dataset in a temporary directory."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "secom.data").write_text(
        "1.0 2.0 3.0\n"
        "4.0 NaN 6.0\n"
        "7.0 8.0 9.0\n"
        "NaN 11.0 12.0\n"
        "13.0 14.0 NaN\n"
    )
    (raw / "secom_labels.data").write_text(
        "-1 2008-11-18 01:00:00\n"
        "1 2008-11-18 02:00:00\n"
        "-1 2008-11-18 03:00:00\n"
        "-1 2008-11-18 04:00:00\n"
        "1 2008-11-18 05:00:00\n"
    )
    return raw


class TestLoadSecom:
    """Tests for load_secom."""

    def test_returns_dataframe(self, raw_dir: Path) -> None:
        assert isinstance(load_secom(raw_dir), pd.DataFrame)

    def test_shape(self, raw_dir: Path) -> None:
        """3 sensors + label + timestamp = 5 columns."""
        assert load_secom(raw_dir).shape == (5, 5)

    def test_sensor_column_names(self, raw_dir: Path) -> None:
        df = load_secom(raw_dir)
        assert list(df.columns[:3]) == ["sensor_000", "sensor_001", "sensor_002"]

    def test_has_label_and_timestamp_columns(self, raw_dir: Path) -> None:
        df = load_secom(raw_dir)
        assert "label" in df.columns
        assert "timestamp" in df.columns

    def test_label_is_binary(self, raw_dir: Path) -> None:
        df = load_secom(raw_dir)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_label_minus_one_becomes_zero(self, raw_dir: Path) -> None:
        df = load_secom(raw_dir)
        assert df["label"].iloc[0] == 0

    def test_label_one_stays_one(self, raw_dir: Path) -> None:
        df = load_secom(raw_dir)
        assert df["label"].iloc[1] == 1

    def test_nan_string_parsed_as_float_nan(self, raw_dir: Path) -> None:
        df = load_secom(raw_dir)
        assert math.isnan(float(df["sensor_001"].iloc[1]))

    def test_row_index_unique(self, raw_dir: Path) -> None:
        assert load_secom(raw_dir).index.is_unique


class TestValidateSecom:
    """Tests for validate_secom."""

    @pytest.fixture()
    def valid_df(self) -> pd.DataFrame:
        """Minimal valid DataFrame matching the full SECOM schema (1567 x 592)."""
        sensor_cols = [f"sensor_{i:03d}" for i in range(590)]
        df = pd.DataFrame(
            np.zeros((1567, 590), dtype=float),
            columns=pd.Index(sensor_cols),
        )
        df["label"] = 0
        df["timestamp"] = "2008-01-01 00:00:00"
        return df

    def test_valid_df_passes(self, valid_df: pd.DataFrame) -> None:
        validate_secom(valid_df)  # must not raise

    def test_wrong_row_count_raises(self, valid_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="shape"):
            validate_secom(valid_df.iloc[:100])

    def test_wrong_column_count_raises(self, valid_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="shape"):
            validate_secom(valid_df.drop(columns=["sensor_000"]))

    def test_missing_sensor_column_raises(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.rename(columns={"sensor_000": "bad_col"})
        with pytest.raises(ValueError, match="sensor_000"):
            validate_secom(bad)

    def test_missing_label_column_raises(self, valid_df: pd.DataFrame) -> None:
        # Keep shape intact by swapping label for a dummy column.
        bad = valid_df.drop(columns=["label"]).assign(extra=0)
        with pytest.raises(ValueError, match="label"):
            validate_secom(bad)

    def test_missing_timestamp_column_raises(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.drop(columns=["timestamp"]).assign(extra=0)
        with pytest.raises(ValueError, match="timestamp"):
            validate_secom(bad)

    def test_invalid_label_value_raises(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad["label"] = 2
        with pytest.raises(ValueError, match="label"):
            validate_secom(bad)

    def test_duplicate_index_raises(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.index = pd.RangeIndex(len(bad)) * 0  # all zeros
        with pytest.raises(ValueError, match="duplicate"):
            validate_secom(bad)
