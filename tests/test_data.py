"""Tests for data loading (load_secom) and schema validation (validate_secom)."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from yield_risk.data import load_secom


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
