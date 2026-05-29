"""Data loading for the SECOM semiconductor yield dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_secom(raw_dir: Path) -> pd.DataFrame:
    """Load and join SECOM sensor readings and labels into a single DataFrame.

    Reads ``secom.data`` (space-delimited sensor columns, ``NaN`` for missing)
    and ``secom_labels.data`` (label ``-1``/``1`` and timestamp), renames sensor
    columns to ``sensor_000``...``sensor_N``, converts labels to binary
    (0 = pass, 1 = fail), and joins on row index.

    Args:
        raw_dir: Directory containing ``secom.data`` and ``secom_labels.data``.

    Returns:
        DataFrame with columns ``sensor_000``...``sensor_N``, ``label`` (0/1),
        and ``timestamp``, indexed by zero-based integer row position.

    Raises:
        FileNotFoundError: If either data file is absent from *raw_dir*.
    """
    sensors = pd.read_csv(
        raw_dir / "secom.data",
        sep=r"\s+",
        header=None,
        na_values=["NaN"],
        engine="python",
    )
    n = len(sensors.columns)
    sensors.columns = pd.Index([f"sensor_{i:03d}" for i in range(n)])

    raw_labels = pd.read_csv(
        raw_dir / "secom_labels.data",
        sep=r"\s+",
        header=None,
        engine="python",
    )
    labels = pd.DataFrame(
        {
            "label": (raw_labels.iloc[:, 0] == 1).astype(int),
            "timestamp": (
                raw_labels.iloc[:, 1].astype(str)
                + " "
                + raw_labels.iloc[:, 2].astype(str)
            ),
        }
    )

    return sensors.join(labels)
