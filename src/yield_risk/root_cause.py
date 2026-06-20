"""Root-cause candidate ranking for yield excursions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from yield_risk.explainability import ShapExplanations


def spc_flag_rate(
    df: pd.DataFrame,
    sensor_cols: list[str],
    n_sigma: float = 3.0,
    reference: pd.DataFrame | None = None,
) -> pd.Series:
    """Compute the fraction of *df* rows outside mean ± n_sigma for each sensor.

    Implements Western Electric Rule 1: a point is flagged when it falls more
    than *n_sigma* standard deviations from the control-limit centre.  Sensors
    with zero variance are never flagged (division by zero is avoided by
    treating std=0 as infinite spread).

    The control limits (mean and standard deviation) are computed from
    *reference* when provided, then applied to *df*.  Deriving the limits from a
    stable reference population — for example the training pass population —
    avoids self-normalisation: a broad excursion in *df* would otherwise inflate
    its own mean and std and mute the very shift being looked for.  When
    *reference* is ``None`` the limits fall back to *df* itself (in-sample,
    less stable).

    Args:
        df: DataFrame containing *sensor_cols* whose rows are evaluated.
        sensor_cols: Sensor column names to evaluate.
        n_sigma: Number of standard deviations for the control limits.
        reference: Population used to compute the control limits. When ``None``,
            *df* is used as its own reference.

    Returns:
        Series indexed by sensor name, values are the fraction of *df* rows
        flagged in [0, 1].
    """
    ref = df if reference is None else reference
    ref_subset = ref[sensor_cols]
    means = ref_subset.mean()
    stds = ref_subset.std().replace(0.0, float("inf"))
    z_scores = (df[sensor_cols] - means).abs() / stds
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


def rank_root_cause_candidates(
    global_importance: pd.DataFrame,
    shap_lift_df: pd.DataFrame,
    flag_rates: pd.Series,
) -> pd.DataFrame:
    """Rank sensors by SHAP attribution with SPC flag rate as supplementary signal.

    Merges mean absolute SHAP, fail/pass SHAP lift, and SPC flag rate on the
    feature/sensor name and sorts by ``mean_abs_shap`` descending.  The SPC
    flag rate and SHAP lift are supplementary columns for engineering review;
    the primary ranking is model attribution only.

    Sensors absent from *shap_lift_df* receive lift of 0.0.  Sensors absent
    from *flag_rates* receive a flag rate of 0.0.

    Args:
        global_importance: DataFrame from global_feature_importance with
            columns [feature, mean_abs_shap].
        shap_lift_df: DataFrame from fail_shap_lift with columns
            [feature, fail_mean_shap, pass_mean_shap, shap_lift].
        flag_rates: Series from spc_flag_rate, indexed by sensor name.

    Returns:
        DataFrame with columns [sensor, mean_abs_shap, shap_lift,
        spc_flag_rate], sorted by mean_abs_shap descending, index reset to
        0..N-1.
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

    return (
        merged.rename(columns={"feature": "sensor"})[
            ["sensor", "mean_abs_shap", "shap_lift", "spc_flag_rate"]
        ]
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
