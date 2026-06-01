"""Feature importance extraction and visualization for yield-risk models."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def extract_lr_coefficients(
    pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract logistic regression coefficients ranked by absolute value.

    Accesses pipeline.named_steps["classifier"].coef_[0] to get the
    coefficient vector. Pairs each coefficient with its feature name,
    computes the absolute value, and sorts descending.

    Args:
        pipeline: Fitted Pipeline whose "classifier" step is a
            LogisticRegression with a coef_ attribute.
        feature_names: Feature names matching the order of coef_ values.

    Returns:
        DataFrame with columns "feature" (str), "coefficient" (float),
        "abs_coefficient" (float), sorted by abs_coefficient descending,
        index reset to 0..N-1.
    """
    coefs: np.ndarray = pipeline.named_steps["classifier"].coef_[0]
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs.tolist(),
            "abs_coefficient": np.abs(coefs).tolist(),
        }
    )
    return df.sort_values("abs_coefficient", ascending=False).reset_index(
        drop=True
    )


def extract_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract feature importance from a fitted pipeline, regardless of model family.

    Dispatches to the appropriate extraction strategy based on the classifier
    type. Linear models (those with ``coef_``) delegate to
    ``extract_lr_coefficients`` and rename columns for a uniform schema. Tree
    models (those with ``feature_importances_``) use the native importance
    values directly.

    Args:
        pipeline: Fitted Pipeline whose "classifier" step is a supported
            estimator (one that exposes either ``coef_`` or
            ``feature_importances_``).
        feature_names: Feature names matching the order of the importance
            values.

    Returns:
        DataFrame with columns:
            - ``feature`` (str): feature name.
            - ``importance`` (float): signed importance for linear models,
              non-negative importance for tree models.
            - ``abs_importance`` (float): absolute value of ``importance``.
        Sorted by ``abs_importance`` descending, index reset to 0..N-1.

    Raises:
        TypeError: If the classifier exposes neither ``coef_`` nor
            ``feature_importances_``. The message includes the classifier
            class name.
    """
    clf = pipeline.named_steps["classifier"]

    if hasattr(clf, "coef_"):
        lr_df = extract_lr_coefficients(pipeline, feature_names)
        return lr_df.rename(
            columns={"coefficient": "importance", "abs_coefficient": "abs_importance"}
        )

    if hasattr(clf, "feature_importances_"):
        importances: np.ndarray = clf.feature_importances_
        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances.tolist(),
                "abs_importance": np.abs(importances).tolist(),
            }
        )
        return df.sort_values("abs_importance", ascending=False).reset_index(
            drop=True
        )

    raise TypeError(
        f"Classifier {type(clf).__name__!r} exposes neither 'coef_' nor "
        "'feature_importances_'. Cannot extract feature importance."
    )


def plot_top_features(
    importance_df: pd.DataFrame,
    top_n: int,
    output_path: Path,
    value_column: str = "coefficient",
) -> None:
    """Save horizontal bar chart of top-N features by importance.

    Bars are colored by the sign of the plotted value (positive = steelblue,
    negative = salmon). Supports both linear-model coefficient columns and
    tree-model importance columns via ``value_column``.

    Args:
        importance_df: Pre-sorted DataFrame containing at least ``feature``
            and the column named by ``value_column``.
        top_n: Number of features to include in the chart.
        output_path: File path for the saved PNG.
        value_column: Name of the numeric column to plot on the x-axis.
            Defaults to ``"coefficient"`` so existing callers that pass a
            DataFrame from ``extract_lr_coefficients`` continue to work
            unchanged.
    """
    top = importance_df.head(top_n)
    colors = [
        "steelblue" if c >= 0 else "salmon"
        for c in top[value_column].tolist()
    ]
    fig_height = max(4, top_n * 0.4)
    _, ax = plt.subplots(figsize=(8, fig_height))
    features_rev = top["feature"].tolist()[::-1]
    vals_rev = top[value_column].tolist()[::-1]
    colors_rev = colors[::-1]
    ax.barh(features_rev, vals_rev, color=colors_rev)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features by Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
