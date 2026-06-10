"""Pytest configuration: make the scripts/ directory importable."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def make_synthetic(
    n_rows: int = 40,
    n_sensors: int = 8,
    n_pos: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a small synthetic raw SECOM-like sensor matrix and labels.

    The matrix exercises every cleaning stage of the sequence preprocessing
    pipeline:

    - Column 0 is high-missing (mostly ``NaN``) and should be dropped.
    - Column 1 is constant (zero variance) and should be dropped.
    - Column 2 is an exact duplicate of column 3, so one of the highly
      correlated pair is dropped (the later column).
    - Column 4 carries a few ordinary ``NaN`` values for median imputation.
    - Column 5 is a signal column correlated with the label.
    - Remaining columns are ordinary noise.

    Args:
        n_rows: Number of wafer rows to generate.
        n_sensors: Number of raw sensor columns (must be >= 6).
        n_pos: Number of positive (fail) labels.
        seed: Seed for the random generator.

    Returns:
        A tuple ``(x, y)`` where ``x`` is float32 shape ``(n_rows, n_sensors)``
        and ``y`` is int64 shape ``(n_rows,)``.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_rows, n_sensors)).astype(np.float32)

    y = np.zeros(n_rows, dtype=np.int64)
    pos_idx = rng.choice(n_rows, size=n_pos, replace=False)
    y[pos_idx] = 1

    # Column 0: high-missing (drop). Keep only the first two rows present.
    x[:, 0] = np.nan
    x[0, 0] = 1.0
    x[1, 0] = 2.0

    # Column 1: constant / low-variance (drop).
    x[:, 1] = 3.0

    # Column 3 then column 2: duplicate pair (later column 3 dropped).
    x[:, 2] = rng.standard_normal(n_rows).astype(np.float32)
    x[:, 3] = x[:, 2]

    # Column 4: ordinary NaN values for median imputation.
    x[:, 4] = rng.standard_normal(n_rows).astype(np.float32)
    x[5, 4] = np.nan
    x[7, 4] = np.nan

    # Column 5: signal correlated with the label.
    x[:, 5] = (
        y.astype(np.float32) * 3.0
        + rng.standard_normal(n_rows).astype(np.float32) * 0.5
    )

    return x.astype(np.float32), y
