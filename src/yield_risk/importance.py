"""Feature importance extraction and visualization for linear models."""
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


def plot_top_features(
    importance_df: pd.DataFrame,
    top_n: int,
    output_path: Path,
) -> None:
    """Save horizontal bar chart of top-N features by absolute coefficient.

    Bars colored by coefficient sign (positive = steelblue, negative = salmon).
    X-axis: coefficient value. Y-axis: feature name. Title includes top_n.

    Args:
        importance_df: DataFrame from extract_lr_coefficients (pre-sorted).
        top_n: Number of features to include in the chart.
        output_path: File path for the saved PNG.
    """
    top = importance_df.head(top_n)
    colors = [
        "steelblue" if c >= 0 else "salmon"
        for c in top["coefficient"].tolist()
    ]
    fig_height = max(4, top_n * 0.4)
    _, ax = plt.subplots(figsize=(8, fig_height))
    features_rev = top["feature"].tolist()[::-1]
    coefs_rev = top["coefficient"].tolist()[::-1]
    colors_rev = colors[::-1]
    ax.barh(features_rev, coefs_rev, color=colors_rev)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Top {top_n} Features by Absolute Coefficient")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
