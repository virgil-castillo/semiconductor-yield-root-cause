"""Monitoring utilities for model and feature drift checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class MissingnessDriftResult:
    """Per-feature missingness drift result.

    Attributes:
        feature: Feature column name.
        reference_missing_rate: Null fraction in the reference batch.
        current_missing_rate: Null fraction in the current batch.
        missing_rate_delta: Absolute difference between current and reference
            null rates.
        alert: Whether the missing-rate delta exceeds the configured threshold.
    """

    feature: str
    reference_missing_rate: float
    current_missing_rate: float
    missing_rate_delta: float
    alert: bool


@dataclass
class MissingnessDriftSummary:
    """Collection of missingness drift results.

    Attributes:
        results: Per-feature missingness drift results.
        threshold: Alert threshold applied to absolute missing-rate deltas.
    """

    results: list[MissingnessDriftResult]
    threshold: float

    def to_frame(self) -> pd.DataFrame:
        """Convert missingness drift results to a DataFrame.

        Returns:
            DataFrame with one row per feature and drift metrics as columns.
        """
        frame = pd.DataFrame([asdict(result) for result in self.results])
        if "alert" in frame:
            frame["alert"] = frame["alert"].astype(object)
        return frame


@dataclass
class FeatureDriftResult:
    """Per-feature distribution drift result.

    Attributes:
        feature: Feature column name.
        reference_mean: Mean of non-null reference values.
        current_mean: Mean of non-null current values.
        mean_delta: Absolute difference between current and reference means.
        reference_std: Population standard deviation of non-null reference
            values.
        current_std: Population standard deviation of non-null current values.
        std_delta: Absolute difference between current and reference standard
            deviations.
        reference_median: Median of non-null reference values.
        current_median: Median of non-null current values.
        median_delta: Absolute difference between current and reference medians.
        ks_statistic: Two-sample Kolmogorov-Smirnov statistic.
        ks_p_value: Two-sample Kolmogorov-Smirnov p-value.
        alert: Whether any monitored delta exceeds the configured threshold.
    """

    feature: str
    reference_mean: float
    current_mean: float
    mean_delta: float
    reference_std: float
    current_std: float
    std_delta: float
    reference_median: float
    current_median: float
    median_delta: float
    ks_statistic: float
    ks_p_value: float
    alert: bool


@dataclass
class FeatureDriftSummary:
    """Collection of feature distribution drift results.

    Attributes:
        results: Per-feature distribution drift results.
        threshold: Alert threshold applied to absolute metric deltas and KS
            statistics.
    """

    results: list[FeatureDriftResult]
    threshold: float

    def to_frame(self) -> pd.DataFrame:
        """Convert feature drift results to a DataFrame.

        Returns:
            DataFrame with one row per feature and distribution drift metrics
            as columns.
        """
        frame = pd.DataFrame([asdict(result) for result in self.results])
        if "alert" in frame:
            frame["alert"] = frame["alert"].astype(object)
        return frame


@dataclass
class PredictionDriftResult:
    """Prediction score distribution drift result.

    Attributes:
        reference_mean: Mean of reference scores.
        current_mean: Mean of current scores.
        mean_delta: Absolute difference between current and reference means.
        reference_median: Median of reference scores.
        current_median: Median of current scores.
        median_delta: Absolute difference between current and reference medians.
        reference_std: Population standard deviation of reference scores.
        current_std: Population standard deviation of current scores.
        std_delta: Absolute difference between current and reference standard
            deviations.
        reference_p90: 90th percentile of reference scores.
        current_p90: 90th percentile of current scores.
        p90_delta: Absolute difference between current and reference 90th
            percentiles.
        ks_statistic: Two-sample Kolmogorov-Smirnov statistic.
        ks_p_value: Two-sample Kolmogorov-Smirnov p-value.
        alert: Whether mean, median, p90, or KS drift exceeds the configured
            threshold.
    """

    reference_mean: float
    current_mean: float
    mean_delta: float
    reference_median: float
    current_median: float
    median_delta: float
    reference_std: float
    current_std: float
    std_delta: float
    reference_p90: float
    current_p90: float
    p90_delta: float
    ks_statistic: float
    ks_p_value: float
    alert: bool


@dataclass
class HighRiskRateDriftResult:
    """High-risk prediction rate drift result.

    Attributes:
        threshold: Operating threshold used to classify scores as high risk.
        reference_rate: Fraction of reference scores at or above threshold.
        current_rate: Fraction of current scores at or above threshold.
        rate_delta: Absolute difference between current and reference rates.
        rate_delta_threshold: Alert threshold applied to high-risk rate drift.
        alert: Whether the high-risk rate delta exceeds the configured
            threshold.
    """

    threshold: float
    reference_rate: float
    current_rate: float
    rate_delta: float
    rate_delta_threshold: float
    alert: bool


@dataclass
class MonitoringSummary:
    """Combined monitoring summary for features and prediction scores.

    Attributes:
        missingness: Missingness drift summary for feature columns.
        features: Distribution drift summary for feature columns.
        predictions: Prediction score distribution drift result.
        high_risk_rate: High-risk score rate drift result.
    """

    missingness: MissingnessDriftSummary
    features: FeatureDriftSummary
    predictions: PredictionDriftResult
    high_risk_rate: HighRiskRateDriftResult


def compare_missingness(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.05,
) -> MissingnessDriftSummary:
    """Compare per-feature missing rates between two batches.

    Args:
        reference: Baseline feature batch.
        current: Current feature batch to compare against the baseline.
        feature_cols: Feature columns to evaluate.
        threshold: Alert threshold for absolute missing-rate deltas.

    Returns:
        MissingnessDriftSummary containing one result per feature.

    Raises:
        ValueError: If inputs are empty, feature columns are missing, the
            feature list is empty, or the threshold is negative.
    """
    _validate_feature_inputs(reference, current, feature_cols, threshold)

    results = []
    for feature in feature_cols:
        reference_rate = float(reference[feature].isna().mean())
        current_rate = float(current[feature].isna().mean())
        delta = abs(current_rate - reference_rate)
        results.append(
            MissingnessDriftResult(
                feature=feature,
                reference_missing_rate=reference_rate,
                current_missing_rate=current_rate,
                missing_rate_delta=delta,
                alert=delta > threshold,
            )
        )
    return MissingnessDriftSummary(results=results, threshold=threshold)


def compare_feature_distributions(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.1,
) -> FeatureDriftSummary:
    """Compare per-feature distributions between two batches.

    Args:
        reference: Baseline feature batch.
        current: Current feature batch to compare against the baseline.
        feature_cols: Feature columns to evaluate.
        threshold: Alert threshold for absolute statistic deltas and KS
            statistics.

    Returns:
        FeatureDriftSummary containing one result per feature.

    Raises:
        ValueError: If inputs are empty, feature columns are missing, the
            feature list is empty, or the threshold is negative.
    """
    _validate_feature_inputs(reference, current, feature_cols, threshold)

    results = []
    for feature in feature_cols:
        reference_values = _finite_feature_values(reference[feature])
        current_values = _finite_feature_values(current[feature])
        result = _feature_drift_result(
            feature, reference_values, current_values, threshold
        )
        results.append(result)
    return FeatureDriftSummary(results=results, threshold=threshold)


def compare_prediction_distributions(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    threshold: float = 0.1,
) -> PredictionDriftResult:
    """Compare prediction score distributions between two batches.

    Args:
        reference_scores: Baseline prediction scores, shape (n,).
        current_scores: Current prediction scores, shape (n,).
        threshold: Alert threshold for mean, median, p90, and KS drift.

    Returns:
        PredictionDriftResult with distribution statistics and alert status.

    Raises:
        ValueError: If score arrays are empty, non-1-D, non-finite, or the
            threshold is negative.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    reference = _validate_scores(reference_scores, "reference_scores")
    current = _validate_scores(current_scores, "current_scores")

    reference_mean = float(np.mean(reference))
    current_mean = float(np.mean(current))
    reference_median = float(np.median(reference))
    current_median = float(np.median(current))
    reference_std = float(np.std(reference))
    current_std = float(np.std(current))
    reference_p90 = float(np.percentile(reference, 90))
    current_p90 = float(np.percentile(current, 90))
    ks_result = ks_2samp(reference, current, method="asymp")
    ks_statistic = float(ks_result.statistic)
    ks_p_value = float(ks_result.pvalue)

    mean_delta = abs(current_mean - reference_mean)
    median_delta = abs(current_median - reference_median)
    std_delta = abs(current_std - reference_std)
    p90_delta = abs(current_p90 - reference_p90)

    return PredictionDriftResult(
        reference_mean=reference_mean,
        current_mean=current_mean,
        mean_delta=mean_delta,
        reference_median=reference_median,
        current_median=current_median,
        median_delta=median_delta,
        reference_std=reference_std,
        current_std=current_std,
        std_delta=std_delta,
        reference_p90=reference_p90,
        current_p90=current_p90,
        p90_delta=p90_delta,
        ks_statistic=ks_statistic,
        ks_p_value=ks_p_value,
        alert=any(
            drift > threshold
            for drift in (mean_delta, median_delta, p90_delta, ks_statistic)
        ),
    )


def compare_high_risk_rate(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    threshold: float,
    rate_delta_threshold: float = 0.05,
) -> HighRiskRateDriftResult:
    """Compare high-risk score rates between two batches.

    Args:
        reference_scores: Baseline prediction scores, shape (n,).
        current_scores: Current prediction scores, shape (n,).
        threshold: Operating threshold used to classify scores as high risk.
        rate_delta_threshold: Alert threshold for high-risk rate drift.

    Returns:
        HighRiskRateDriftResult with rates, delta, and alert status.

    Raises:
        ValueError: If score arrays are empty, non-1-D, non-finite, or either
            threshold is negative.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    if rate_delta_threshold < 0:
        raise ValueError("rate_delta_threshold must be non-negative.")
    reference = _validate_scores(reference_scores, "reference_scores")
    current = _validate_scores(current_scores, "current_scores")

    reference_rate = float(np.mean(reference >= threshold))
    current_rate = float(np.mean(current >= threshold))
    rate_delta = abs(current_rate - reference_rate)
    return HighRiskRateDriftResult(
        threshold=threshold,
        reference_rate=reference_rate,
        current_rate=current_rate,
        rate_delta=rate_delta,
        rate_delta_threshold=rate_delta_threshold,
        alert=rate_delta > rate_delta_threshold,
    )


def build_monitoring_summary(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    high_risk_threshold: float,
) -> MonitoringSummary:
    """Build a combined monitoring summary for feature and score drift.

    Args:
        reference: Baseline feature batch.
        current: Current feature batch to compare against the baseline.
        feature_cols: Feature columns to evaluate.
        reference_scores: Baseline prediction scores, shape (n,).
        current_scores: Current prediction scores, shape (n,).
        high_risk_threshold: Operating threshold for high-risk rate drift.

    Returns:
        MonitoringSummary containing missingness, feature, prediction, and
        high-risk rate drift sections.

    Raises:
        ValueError: If any delegated monitoring comparison receives invalid
            inputs.
    """
    return MonitoringSummary(
        missingness=compare_missingness(reference, current, feature_cols),
        features=compare_feature_distributions(reference, current, feature_cols),
        predictions=compare_prediction_distributions(reference_scores, current_scores),
        high_risk_rate=compare_high_risk_rate(
            reference_scores,
            current_scores,
            threshold=high_risk_threshold,
        ),
    )


def _validate_feature_inputs(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> None:
    if reference.empty:
        raise ValueError("reference batch must not be empty.")
    if current.empty:
        raise ValueError("current batch must not be empty.")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    missing_reference = sorted(set(feature_cols).difference(reference.columns))
    missing_current = sorted(set(feature_cols).difference(current.columns))
    missing = missing_reference + missing_current
    if missing:
        raise ValueError(f"missing feature columns: {sorted(set(missing))}")


def _validate_scores(scores: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite scores.")
    return values


def _finite_feature_values(values: pd.Series) -> np.ndarray:
    return values.dropna().to_numpy(dtype=float)


def _feature_drift_result(
    feature: str,
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float,
) -> FeatureDriftResult:
    reference_mean = _safe_mean(reference)
    current_mean = _safe_mean(current)
    reference_std = _safe_std(reference)
    current_std = _safe_std(current)
    reference_median = _safe_median(reference)
    current_median = _safe_median(current)

    if reference.size == 0 or current.size == 0:
        ks_statistic = 0.0
        ks_p_value = 1.0
    else:
        ks_result = ks_2samp(reference, current, method="asymp")
        ks_statistic = float(ks_result.statistic)
        ks_p_value = float(ks_result.pvalue)

    has_comparable_values = reference.size > 0 and current.size > 0
    mean_delta = (
        _absolute_delta(reference_mean, current_mean) if has_comparable_values else 0.0
    )
    std_delta = (
        _absolute_delta(reference_std, current_std) if has_comparable_values else 0.0
    )
    median_delta = (
        _absolute_delta(reference_median, current_median)
        if has_comparable_values
        else 0.0
    )

    return FeatureDriftResult(
        feature=feature,
        reference_mean=reference_mean,
        current_mean=current_mean,
        mean_delta=mean_delta,
        reference_std=reference_std,
        current_std=current_std,
        std_delta=std_delta,
        reference_median=reference_median,
        current_median=current_median,
        median_delta=median_delta,
        ks_statistic=ks_statistic,
        ks_p_value=ks_p_value,
        alert=any(
            drift > threshold
            for drift in (mean_delta, std_delta, median_delta, ks_statistic)
        ),
    )


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _safe_median(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def _absolute_delta(left: float, right: float) -> float:
    return abs(right - left)
