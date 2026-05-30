"""Feature engineering for the SECOM yield-risk pipeline."""
from __future__ import annotations

import pandas as pd

_NON_SENSOR: frozenset[str] = frozenset({"label", "timestamp"})


def get_sensor_cols(df: pd.DataFrame) -> list[str]:
    """Return column names that are sensor readings (excluding metadata).

    Excludes ``label`` and ``timestamp`` columns.

    Args:
        df: DataFrame with sensor, label, and timestamp columns.

    Returns:
        List of sensor column names in their original order.
    """
    return [c for c in df.columns if c not in _NON_SENSOR]


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-level aggregate statistics across all sensor columns.

    Appends four columns to the DataFrame:

    * ``sensor_row_mean`` — row mean of sensor readings
    * ``sensor_row_std`` — row standard deviation of sensor readings
    * ``sensor_row_min`` — row minimum of sensor readings
    * ``sensor_row_max`` — row maximum of sensor readings

    Args:
        df: Preprocessed DataFrame with no NaN values in sensor columns.

    Returns:
        Copy of *df* with four additional aggregate columns appended.
    """
    cols = get_sensor_cols(df)
    result = df.copy()
    sensor_vals = df[cols]
    result["sensor_row_mean"] = sensor_vals.mean(axis=1)
    result["sensor_row_std"] = sensor_vals.std(axis=1)
    result["sensor_row_min"] = sensor_vals.min(axis=1)
    result["sensor_row_max"] = sensor_vals.max(axis=1)
    return result


def add_top_interactions(
    df: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """Add pairwise product features for the top-*k* highest-variance sensors.

    Selects the *top_k* sensor columns by variance, then appends a column for
    every unique pair ``i < j`` containing the elementwise product
    ``col_i * col_j``.  Column names use the ``<col_i>__x__<col_j>`` format.

    Args:
        df: Preprocessed DataFrame with no NaN values in sensor columns.
        top_k: Number of highest-variance sensors to use.

    Returns:
        Copy of *df* with pairwise interaction columns appended.
    """
    cols = get_sensor_cols(df)
    variances = df[cols].var()
    top_cols = variances.nlargest(top_k).index.tolist()
    result = df.copy()
    for i, col_i in enumerate(top_cols):
        for col_j in top_cols[i + 1 :]:
            interaction_name = f"{col_i}__x__{col_j}"
            result[interaction_name] = df[col_i] * df[col_j]
    return result


def engineer_features(
    df: pd.DataFrame,
    top_k_interactions: int = 10,
) -> pd.DataFrame:
    """Apply the full feature engineering pipeline to a preprocessed DataFrame.

    Pipeline order:

    1. :func:`add_aggregate_features` — row-level sensor statistics
    2. :func:`add_top_interactions` — pairwise products of top-variance sensors

    Interaction features are computed from the original sensor columns only,
    so aggregate columns (``sensor_row_mean`` etc.) do not compete for the
    top-k variance slots.

    Args:
        df: Preprocessed DataFrame with no NaN values in sensor columns.
        top_k_interactions: Number of highest-variance sensors to use for
            pairwise interaction features.

    Returns:
        DataFrame with original columns plus engineered feature columns.
    """
    result = add_aggregate_features(df)
    interactions = add_top_interactions(df, top_k=top_k_interactions)
    new_cols = [c for c in interactions.columns if "__x__" in c]
    return pd.concat([result, interactions[new_cols]], axis=1)
