"""Feature selection, imputation, and train/test split for the SECOM pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from yield_risk.config import RunConfig

_NON_SENSOR: frozenset[str] = frozenset({"label", "timestamp"})


def _sensor_cols(df: pd.DataFrame) -> list[str]:
    """Return column names that are not label or timestamp.

    Args:
        df: DataFrame with mixed sensor and metadata columns.

    Returns:
        List of sensor column names.
    """
    return [c for c in df.columns if c not in _NON_SENSOR]


def _drop_high_missing(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop columns whose missing-value fraction is strictly above threshold.

    Args:
        df: Sensor-only DataFrame (no label/timestamp).
        threshold: Columns with missing fraction strictly above this are removed.

    Returns:
        DataFrame with high-missing columns removed.
    """
    missing_rate = df.isnull().mean()
    to_drop = [str(c) for c in missing_rate[missing_rate > threshold].index]
    return df.drop(columns=to_drop)


def _impute_with_medians(
    df: pd.DataFrame, medians: pd.Series
) -> pd.DataFrame:
    """Fill NaN values in *df* using the provided per-column medians.

    Args:
        df: Sensor-only DataFrame (subset of columns present in medians).
        medians: Series of median values indexed by column name.

    Returns:
        Copy of *df* with NaN values replaced by the supplied medians.
    """
    return df.fillna(medians)


def _drop_low_variance(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop columns whose variance is strictly below threshold.

    Args:
        df: Sensor-only DataFrame.
        threshold: Columns with variance strictly below this are removed.

    Returns:
        DataFrame with low-variance columns removed.
    """
    variances = df.var()
    to_drop = [str(c) for c in variances[variances < threshold].index]
    return df.drop(columns=to_drop)


def _drop_high_correlation(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop one column from each highly-correlated pair (later by position).

    Args:
        df: Sensor-only DataFrame.
        threshold: Drop a column if its absolute correlation with any earlier
            column strictly exceeds this value.

    Returns:
        DataFrame with one column from each correlated pair removed.
    """
    corr = df.corr().abs()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper = corr.where(mask)
    to_drop = [str(c) for c in upper.columns if bool((upper[c] > threshold).any())]
    return df.drop(columns=to_drop)


class SecomPreprocessor(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """sklearn-compatible transformer that learns feature selection and medians.

    Applies, in order: missing-rate filter, median imputation, variance filter,
    correlation filter.  All statistics are learned from the training data only.

    Args:
        missing_threshold: Drop columns with missing fraction strictly above this.
        variance_threshold: Drop columns with variance strictly below this.
        correlation_threshold: Drop the later of each pair with |r| strictly
            above this value.
    """

    def __init__(
        self,
        missing_threshold: float,
        variance_threshold: float,
        correlation_threshold: float,
    ) -> None:
        """Store hyperparameters verbatim (sklearn convention — no mutation here).

        Args:
            missing_threshold: Missing-rate upper bound for kept columns.
            variance_threshold: Variance lower bound for kept columns.
            correlation_threshold: Absolute-correlation upper bound between
                any two kept columns.
        """
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

    def fit(
        self,
        X: pd.DataFrame,
        y: object = None,
    ) -> SecomPreprocessor:
        """Learn kept columns and per-column medians from training rows.

        Pipeline order:
        1. Missing-rate filter (strictly above missing_threshold → dropped).
        2. Per-column median computation and internal imputation.
        3. Variance filter (strictly below variance_threshold → dropped).
        4. Correlation filter (later column of pair with |r| > correlation_threshold
           → dropped).

        Args:
            X: Sensor-only DataFrame (no label/timestamp columns).
            y: Ignored; present for sklearn API compatibility.

        Returns:
            self (fitted transformer).
        """
        # Step 1: missing-rate filter
        after_missing = _drop_high_missing(X, self.missing_threshold)

        # Step 2: compute medians on training non-null values; impute internally
        train_medians: pd.Series = after_missing.median()
        after_imputed = _impute_with_medians(after_missing, train_medians)

        # Step 3: variance filter (on imputed data)
        after_variance = _drop_low_variance(after_imputed, self.variance_threshold)

        # Step 4: correlation filter
        after_correlation = _drop_high_correlation(
            after_variance, self.correlation_threshold
        )

        # Store fitted attributes
        self.kept_columns_: list[str] = list(after_correlation.columns)
        # Restrict medians to kept columns only
        self.medians_: pd.Series = train_medians[self.kept_columns_]
        self.n_features_in_: int = X.shape[1]
        self.feature_names_in_: np.ndarray = np.array(list(X.columns), dtype=object)

        return self

    def transform(self, X: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Select kept columns and impute NaNs with training medians.

        Args:
            X: Sensor-only DataFrame with at least the kept columns present.
            y: Ignored; present for sklearn API compatibility.

        Returns:
            DataFrame containing only the kept columns, with NaNs filled by
            the training-set medians.

        Raises:
            NotFittedError: If called before fit.
        """
        check_is_fitted(self, ["kept_columns_", "medians_"])
        out = X[self.kept_columns_].copy()
        out = _impute_with_medians(out, self.medians_)
        return out

    def get_feature_names_out(
        self, input_features: object = None
    ) -> list[str]:
        """Return the names of the kept sensor columns.

        Args:
            input_features: Ignored; present for sklearn API compatibility.

        Returns:
            List of kept column name strings.

        Raises:
            NotFittedError: If called before fit.
        """
        check_is_fitted(self, ["kept_columns_"])
        return list(self.kept_columns_)


def split_by_time(
    df: pd.DataFrame,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stable-sort by timestamp and take the last test_size fraction as test.

    The split is position-based after the stable sort.  Identical timestamps
    at the boundary fall on whichever side the position cut lands.

    Args:
        df: DataFrame containing ``timestamp`` and ``label`` columns.
        test_size: Fraction of rows (contiguous tail after sort) held out as test.

    Returns:
        Tuple of ``(train_df, test_df)`` with NaNs intact and all columns.

    Raises:
        ValueError: If either split contains only one distinct class in ``label``.
    """
    sorted_df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n = len(sorted_df)
    n_test = max(1, round(n * test_size))
    n_train = n - n_test

    train = sorted_df.iloc[:n_train]
    test = sorted_df.iloc[n_train:]

    for name, split in (("train", train), ("test", test)):
        n_classes = split["label"].nunique()
        if n_classes < 2:
            raise ValueError(
                f"single-class split: the {name} split contains only "
                f"{n_classes} distinct class(es) in 'label'. "
                "Adjust test_size so both splits have at least two classes."
            )

    return train, test


def run_preprocessing(
    df: pd.DataFrame,
    run_cfg: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stable-sort by timestamp and split into raw train/test halves.

    Returns raw splits with NaNs intact and all sensor columns retained.
    No imputation or feature selection is performed here; apply
    ``SecomPreprocessor`` to the training split to learn those transforms.

    Args:
        df: Raw SECOM DataFrame (output of load_secom, already validated).
        run_cfg: RunConfig with split parameters (test_size used).

    Returns:
        Tuple of (train_df, test_df).  Both contain all sensor columns, label,
        and timestamp.  NaN values in sensor columns are preserved.
    """
    return split_by_time(df, run_cfg.test_size)
