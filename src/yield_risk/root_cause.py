"""Root-cause candidate ranking for yield excursions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from yield_risk.explainability import ShapExplanations


def spc_flag_rate(
    df: pd.DataFrame,
    sensor_cols: list[str],
    n_sigma: float = 3.0,
) -> pd.Series:
    """Compute the fraction of rows outside mean ± n_sigma for each sensor.

    Implements Western Electric Rule 1: a point is flagged when it falls more
    than *n_sigma* standard deviations from the column mean.  Sensors with
    zero variance are never flagged (division by zero is avoided by treating
    std=0 as infinite spread).

    Args:
        df: DataFrame containing *sensor_cols*.
        sensor_cols: Sensor column names to evaluate.
        n_sigma: Number of standard deviations for the control limits.

    Returns:
        Series indexed by sensor name, values are the fraction of rows
        flagged in [0, 1].
    """
    subset = df[sensor_cols]
    means = subset.mean()
    stds = subset.std().replace(0.0, float("inf"))
    z_scores = (subset - means).abs() / stds
    flags = z_scores > n_sigma
    rate: pd.Series = flags.mean()
    return rate


def fail_shap_lift(
    explanations: ShapExplanations,
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Compute mean SHAP values for failing vs passing wafers per feature.

    Args:
        explanations: SHAP results from
            :func:`~yield_risk.explainability.compute_shap_values`.
        y_true: True binary labels (0=pass, 1=fail), shape ``(n_samples,)``.

    Returns:
        DataFrame with columns:

        - ``feature``: sensor name
        - ``fail_mean_shap``: mean SHAP value across failing wafers
        - ``pass_mean_shap``: mean SHAP value across passing wafers
        - ``shap_lift``: ``fail_mean_shap - pass_mean_shap``

        Sorted by ``|shap_lift|`` descending, index reset to 0..N-1.

    Raises:
        ValueError: If *y_true* contains no failing samples or no passing samples.
    """
    sv = explanations.shap_values
    fail_mask = y_true == 1
    pass_mask = ~fail_mask

    if not fail_mask.any():
        raise ValueError("y_true contains no failing samples (label=1).")
    if not pass_mask.any():
        raise ValueError("y_true contains no passing samples (label=0).")

    fail_mean = sv[fail_mask].mean(axis=0)
    pass_mean = sv[pass_mask].mean(axis=0)
    lift = fail_mean - pass_mean

    df = pd.DataFrame(
        {
            "feature": explanations.feature_names,
            "fail_mean_shap": fail_mean.tolist(),
            "pass_mean_shap": pass_mean.tolist(),
            "shap_lift": lift.tolist(),
        }
    )
    return (
        df.assign(abs_lift=df["shap_lift"].abs())
        .sort_values("abs_lift", ascending=False)
        .drop(columns=["abs_lift"])
        .reset_index(drop=True)
    )


def _min_max_normalize(s: pd.Series) -> pd.Series:
    """Min-max scale a Series to [0, 1]; returns zeros when range is zero.

    Args:
        s: Numeric Series to normalize.

    Returns:
        Series of the same index with values in [0, 1].
    """
    rng = float(s.max() - s.min())
    if rng == 0.0:
        return pd.Series(0.0, index=s.index)
    result: pd.Series = (s - s.min()) / rng
    return result


def rank_root_cause_candidates(
    global_importance: pd.DataFrame,
    shap_lift_df: pd.DataFrame,
    flag_rates: pd.Series,
) -> pd.DataFrame:
    """Combine SHAP importance, fail/pass lift, and SPC flag rate into a ranked list.

    Merges the three signals on the feature/sensor name, min-max normalises
    each signal, then computes a weighted composite score::

        composite_score = 0.5 * norm(mean_abs_shap)
                        + 0.3 * norm(|shap_lift|)
                        + 0.2 * norm(spc_flag_rate)

    Sensors absent from *flag_rates* are assigned a flag rate of 0.0.

    Args:
        global_importance: DataFrame from
            :func:`~yield_risk.explainability.global_feature_importance` with
            columns ``[feature, mean_abs_shap]``.
        shap_lift_df: DataFrame from :func:`fail_shap_lift` with columns
            ``[feature, fail_mean_shap, pass_mean_shap, shap_lift]``.
        flag_rates: Series from :func:`spc_flag_rate`, indexed by sensor
            name with fraction-flagged values.

    Returns:
        DataFrame with columns:

        - ``sensor``: sensor name
        - ``mean_abs_shap``: global SHAP importance
        - ``shap_lift``: fail mean SHAP − pass mean SHAP
        - ``spc_flag_rate``: fraction of rows triggering SPC flag
        - ``composite_score``: weighted normalised composite in [0, 1]

        Sorted by ``composite_score`` descending, index reset to 0..N-1.
    """
    merged = global_importance.merge(
        shap_lift_df[["feature", "shap_lift"]], on="feature", how="left"
    )
    flag_df = pd.DataFrame(
        {
            "feature": flag_rates.index.tolist(),
            "spc_flag_rate": flag_rates.to_numpy().tolist(),
        }
    )
    merged = merged.merge(flag_df, on="feature", how="left")
    merged["shap_lift"] = merged["shap_lift"].fillna(0.0)
    merged["spc_flag_rate"] = merged["spc_flag_rate"].fillna(0.0)

    merged["composite_score"] = (
        0.5 * _min_max_normalize(merged["mean_abs_shap"])
        + 0.3 * _min_max_normalize(merged["shap_lift"].abs())
        + 0.2 * _min_max_normalize(merged["spc_flag_rate"])
    )

    return (
        merged.rename(columns={"feature": "sensor"})[
            [
                "sensor",
                "mean_abs_shap",
                "shap_lift",
                "spc_flag_rate",
                "composite_score",
            ]
        ]
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
