"""Tests for model monitoring drift utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yield_risk.monitoring import (
    FeatureDriftSummary,
    MissingnessDriftSummary,
    MonitoringSummary,
    build_monitoring_summary,
    compare_feature_distributions,
    compare_high_risk_rate,
    compare_missingness,
    compare_prediction_distributions,
)


def test_missingness_drift_flags_expected_columns() -> None:
    reference = pd.DataFrame(
        {"sensor_a": [1.0, 2.0, np.nan, 4.0], "sensor_b": [1.0, 2.0, 3.0, 4.0]}
    )
    current = pd.DataFrame(
        {
            "sensor_a": [1.0, np.nan, np.nan, 4.0],
            "sensor_b": [np.nan, 2.0, 3.0, 4.0],
        }
    )

    summary = compare_missingness(
        reference, current, ["sensor_a", "sensor_b"], threshold=0.2
    )
    frame = summary.to_frame()

    assert isinstance(summary, MissingnessDriftSummary)
    assert list(frame.columns) == [
        "feature",
        "reference_missing_rate",
        "current_missing_rate",
        "missing_rate_delta",
        "alert",
    ]
    assert frame.loc[frame["feature"] == "sensor_a", "alert"].item() is True
    assert frame.loc[frame["feature"] == "sensor_b", "alert"].item() is True
    assert frame.loc[
        frame["feature"] == "sensor_a", "missing_rate_delta"
    ].item() == pytest.approx(0.25)


def test_feature_distribution_drift_computes_statistics_and_alerts() -> None:
    reference = pd.DataFrame(
        {
            "sensor_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "sensor_b": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    current = pd.DataFrame(
        {
            "sensor_a": [2.0, 3.0, 4.0, 5.0, 6.0],
            "sensor_b": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    summary = compare_feature_distributions(
        reference, current, ["sensor_a", "sensor_b"], threshold=0.5
    )
    frame = summary.to_frame()
    sensor_a = frame.loc[frame["feature"] == "sensor_a"].iloc[0]
    sensor_b = frame.loc[frame["feature"] == "sensor_b"].iloc[0]

    assert isinstance(summary, FeatureDriftSummary)
    assert list(frame.columns) == [
        "feature",
        "reference_mean",
        "current_mean",
        "mean_delta",
        "reference_std",
        "current_std",
        "std_delta",
        "reference_median",
        "current_median",
        "median_delta",
        "ks_statistic",
        "ks_p_value",
        "alert",
    ]
    assert sensor_a["mean_delta"] == pytest.approx(1.0)
    assert sensor_a["std_delta"] == pytest.approx(0.0)
    assert sensor_a["median_delta"] == pytest.approx(1.0)
    assert sensor_a["ks_statistic"] == pytest.approx(0.2)
    assert sensor_a["alert"] is True
    assert sensor_b["ks_statistic"] == pytest.approx(0.0)
    assert sensor_b["ks_p_value"] == pytest.approx(1.0)
    assert sensor_b["alert"] is False


def test_prediction_distribution_drift_handles_changed_scores() -> None:
    reference_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    current_scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    result = compare_prediction_distributions(
        reference_scores, current_scores, threshold=0.2
    )

    assert result.reference_mean == pytest.approx(0.3)
    assert result.current_mean == pytest.approx(0.7)
    assert result.mean_delta == pytest.approx(0.4)
    assert result.reference_median == pytest.approx(0.3)
    assert result.current_median == pytest.approx(0.7)
    assert result.median_delta == pytest.approx(0.4)
    assert result.std_delta == pytest.approx(0.0)
    assert result.reference_p90 == pytest.approx(0.46)
    assert result.current_p90 == pytest.approx(0.86)
    assert result.p90_delta == pytest.approx(0.4)
    assert result.ks_statistic == pytest.approx(0.8)
    assert result.alert is True


def test_high_risk_rate_drift_uses_supplied_operating_threshold() -> None:
    reference_scores = np.array([0.1, 0.2, 0.8, 0.9])
    current_scores = np.array([0.6, 0.7, 0.8, 0.9])

    result = compare_high_risk_rate(
        reference_scores,
        current_scores,
        threshold=0.7,
        rate_delta_threshold=0.2,
    )

    assert result.threshold == pytest.approx(0.7)
    assert result.reference_rate == pytest.approx(0.5)
    assert result.current_rate == pytest.approx(0.75)
    assert result.rate_delta == pytest.approx(0.25)
    assert result.alert is True


def test_build_monitoring_summary_contains_all_sections() -> None:
    reference = pd.DataFrame({"sensor_a": [1.0, 2.0, 3.0], "sensor_b": [1.0, 2.0, 3.0]})
    current = pd.DataFrame(
        {"sensor_a": [2.0, 3.0, 4.0], "sensor_b": [1.0, np.nan, 3.0]}
    )
    reference_scores = np.array([0.1, 0.2, 0.3])
    current_scores = np.array([0.4, 0.5, 0.6])

    summary = build_monitoring_summary(
        reference,
        current,
        ["sensor_a", "sensor_b"],
        reference_scores,
        current_scores,
        high_risk_threshold=0.5,
    )

    assert isinstance(summary, MonitoringSummary)
    assert isinstance(summary.missingness, MissingnessDriftSummary)
    assert isinstance(summary.features, FeatureDriftSummary)
    assert summary.predictions.alert is True
    assert summary.high_risk_rate.threshold == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("reference", "current", "features", "threshold", "message"),
    [
        (pd.DataFrame(), pd.DataFrame({"sensor_a": [1.0]}), ["sensor_a"], 0.1, "empty"),
        (pd.DataFrame({"sensor_a": [1.0]}), pd.DataFrame(), ["sensor_a"], 0.1, "empty"),
        (
            pd.DataFrame({"sensor_a": [1.0]}),
            pd.DataFrame({"sensor_a": [1.0]}),
            [],
            0.1,
            "feature_cols",
        ),
        (
            pd.DataFrame({"sensor_a": [1.0]}),
            pd.DataFrame({"sensor_b": [1.0]}),
            ["sensor_a"],
            0.1,
            "missing",
        ),
        (
            pd.DataFrame({"sensor_a": [1.0]}),
            pd.DataFrame({"sensor_a": [1.0]}),
            ["sensor_a"],
            -0.1,
            "threshold",
        ),
    ],
)
def test_feature_batch_validation_raises_clear_value_errors(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    threshold: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_missingness(reference, current, features, threshold=threshold)


@pytest.mark.parametrize(
    ("reference_scores", "current_scores", "message"),
    [
        (np.array([]), np.array([0.1]), "empty"),
        (np.array([0.1]), np.array([]), "empty"),
        (np.array([[0.1, 0.2]]), np.array([0.1, 0.2]), "1-D"),
        (np.array([0.1, np.nan]), np.array([0.1, 0.2]), "finite"),
        (np.array([0.1, np.inf]), np.array([0.1, 0.2]), "finite"),
    ],
)
def test_score_validation_raises_clear_value_errors(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_prediction_distributions(reference_scores, current_scores)


def test_high_risk_rate_rejects_negative_rate_delta_threshold() -> None:
    with pytest.raises(ValueError, match="rate_delta_threshold"):
        compare_high_risk_rate(
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            threshold=0.5,
            rate_delta_threshold=-0.1,
        )
