"""Feature selection, imputation, and train/test split for the SECOM pipeline."""
from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_NON_SENSOR: frozenset[str] = frozenset({"label", "timestamp"})


def _sensor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_SENSOR]


def drop_high_missing(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop sensor columns whose missing-value fraction is strictly above threshold.

    Args:
        df: DataFrame with sensor, label, and timestamp columns.
        threshold: Columns with a missing fraction strictly above this value
            are removed.

    Returns:
        DataFrame with high-missing sensor columns removed.
    """
    cols = _sensor_cols(df)
    missing_rate = df[cols].isnull().mean()
    to_drop = [str(c) for c in missing_rate[missing_rate > threshold].index]
    return df.drop(columns=to_drop)


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in sensor columns with each column's median.

    The median is computed from non-null values in that column.

    Args:
        df: DataFrame with sensor, label, and timestamp columns.

    Returns:
        Copy of *df* with NaN values in sensor columns replaced by column medians.
    """
    cols = _sensor_cols(df)
    medians = df[cols].median()
    result = df.copy()
    result[cols] = df[cols].fillna(medians)
    return result


def drop_low_variance(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop sensor columns whose variance is strictly below threshold.

    Args:
        df: DataFrame with sensor, label, and timestamp columns.
        threshold: Columns with variance strictly below this value are removed.

    Returns:
        DataFrame with low-variance sensor columns removed.
    """
    cols = _sensor_cols(df)
    variances = df[cols].var()
    to_drop = [str(c) for c in variances[variances < threshold].index]
    return df.drop(columns=to_drop)


def drop_high_correlation(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop one column from each highly-correlated pair.

    Uses the upper triangle of the absolute correlation matrix.  For each
    correlated pair the later column (by position) is removed.

    Args:
        df: DataFrame with sensor, label, and timestamp columns.
        threshold: Drop a column if its absolute correlation with any earlier
            column strictly exceeds this value.

    Returns:
        DataFrame with one column from each highly-correlated pair removed.
    """
    cols = _sensor_cols(df)
    corr = df[cols].corr().abs()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper = corr.where(mask)
    to_drop = [str(c) for c in upper.columns if bool((upper[c] > threshold).any())]
    return df.drop(columns=to_drop)


def split_stratified(
    df: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split that preserves the class ratio of ``label``.

    Args:
        df: DataFrame containing a ``label`` column.
        test_size: Fraction of rows to place in the test split.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(train_df, test_df)``.
    """
    splits = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df["label"],
    )
    return cast(pd.DataFrame, splits[0]), cast(pd.DataFrame, splits[1])
