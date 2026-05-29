"""Schema validation for the loaded SECOM DataFrame."""
from __future__ import annotations

import pandas as pd

_EXPECTED_SHAPE: tuple[int, int] = (1567, 592)
_N_SENSORS: int = 590


def validate_secom(df: pd.DataFrame) -> None:
    """Assert that *df* conforms to the expected SECOM schema.

    Checks performed in order:
    - Shape is (1567, 592).
    - All sensor columns ``sensor_000``...``sensor_589`` are present.
    - ``label`` and ``timestamp`` columns are present.
    - ``label`` contains only the values 0 and 1.
    - Row index has no duplicates.

    Args:
        df: DataFrame to validate (typically the output of :func:`load_secom`).

    Raises:
        ValueError: If any assertion fails, with a descriptive message.
    """
    if df.shape != _EXPECTED_SHAPE:
        raise ValueError(
            f"Expected shape {_EXPECTED_SHAPE}, got {df.shape}."
        )

    required = (
        {f"sensor_{i:03d}" for i in range(_N_SENSORS)} | {"label", "timestamp"}
    )
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(sorted(missing))}"
        )

    invalid_labels = set(df["label"].unique().tolist()) - {0, 1}
    if invalid_labels:
        raise ValueError(
            f"label contains invalid values: {invalid_labels}"
        )

    if not df.index.is_unique:
        raise ValueError("DataFrame has duplicate row indices.")
