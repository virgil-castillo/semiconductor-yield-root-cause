"""SHAP-based global and local explanations for the yield-risk pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


@dataclass
class ShapExplanations:
    """Container for SHAP explanation results.

    Attributes:
        shap_values: SHAP values for the positive class (fail),
            shape ``(n_samples, n_features)``.
        feature_names: Column names in the same order as the last axis of
            *shap_values*.
        base_value: Model's average prediction (expected value) for the
            positive class.
    """

    shap_values: np.ndarray
    feature_names: list[str]
    base_value: float


def _transform_pre_steps(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """Apply all pipeline steps except the final estimator to *X*.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X: Input feature DataFrame.

    Returns:
        Transformed array after all pre-final steps, or the raw numpy
        representation of *X* if the pipeline has only one step.
    """
    if len(pipeline.steps) <= 1:
        return X.to_numpy()
    pre = Pipeline(pipeline.steps[:-1])
    result: np.ndarray = pre.transform(X)
    return result


def compute_shap_values(
    pipeline: Pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
) -> ShapExplanations:
    """Compute SHAP values for *X_explain* using the most efficient explainer.

    Selects the explainer based on the type of the final pipeline step:

    - ``LinearExplainer`` when the final estimator has a ``coef_`` attribute
      (e.g. LogisticRegression).
    - ``TreeExplainer`` when the final estimator has a ``feature_importances_``
      attribute (e.g. RandomForest, XGBoost, LightGBM).

    *X* is transformed through all pre-final pipeline steps before being passed
    to the explainer.  SHAP values are always extracted for the positive class
    (class index 1 for binary classification).

    Args:
        pipeline: Fitted sklearn Pipeline whose last step is the classifier.
        X_background: Background dataset for explainer initialisation. Should
            be a representative sample of training data.
        X_explain: Samples to explain. Must have the same columns as
            *X_background*.

    Returns:
        ShapExplanations containing SHAP values for the positive class.

    Raises:
        NotImplementedError: If the final estimator type is not supported.
    """
    clf: Any = pipeline.steps[-1][1]
    bg_array = _transform_pre_steps(pipeline, X_background)
    ex_array = _transform_pre_steps(pipeline, X_explain)

    if hasattr(clf, "coef_"):
        explainer = shap.LinearExplainer(clf, bg_array)
        sv: np.ndarray | list[np.ndarray] = explainer.shap_values(ex_array)
        ev: Any = explainer.expected_value
    elif hasattr(clf, "feature_importances_"):
        explainer = shap.TreeExplainer(clf, bg_array)
        sv = explainer.shap_values(ex_array)
        ev = explainer.expected_value
    else:
        raise NotImplementedError(
            f"No SHAP explainer implemented for classifier type:"
            f" {type(clf).__name__}. Supported: LinearExplainer (coef_),"
            f" TreeExplainer (feature_importances_)."
        )

    # Normalize to positive-class 2-D array: (n_samples, n_features)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        # SHAP >= 0.45 TreeExplainer binary: (n_samples, n_features, n_classes)
        shap_vals: np.ndarray = sv[:, :, 1]
        base = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
    elif isinstance(sv, list):
        # Legacy SHAP format: list of [class_0_array, class_1_array]
        shap_vals = sv[1]
        base = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
    else:
        # LinearExplainer: 2-D array already
        shap_vals = sv
        base = float(ev[1]) if hasattr(ev, "__len__") else float(ev)

    return ShapExplanations(
        shap_values=shap_vals,
        feature_names=list(X_explain.columns),
        base_value=base,
    )


def global_feature_importance(
    explanations: ShapExplanations,
) -> pd.DataFrame:
    """Compute global feature importance as mean absolute SHAP value.

    Args:
        explanations: SHAP results from :func:`compute_shap_values`.

    Returns:
        DataFrame with columns ``feature`` (str) and ``mean_abs_shap``
        (float), sorted by ``mean_abs_shap`` descending, index reset to
        ``0..N-1``.
    """
    mean_abs = np.abs(explanations.shap_values).mean(axis=0)
    df = pd.DataFrame(
        {
            "feature": explanations.feature_names,
            "mean_abs_shap": mean_abs.tolist(),
        }
    )
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_explanation(
    explanations: ShapExplanations,
    row_idx: int,
) -> pd.DataFrame:
    """Return per-feature SHAP values for a single sample.

    Args:
        explanations: SHAP results from :func:`compute_shap_values`.
        row_idx: Zero-based index of the sample within
            ``explanations.shap_values``.

    Returns:
        DataFrame with columns ``feature`` (str) and ``shap_value`` (float),
        sorted by ``|shap_value|`` descending, index reset to ``0..N-1``.
    """
    sv_row = explanations.shap_values[row_idx]
    df = pd.DataFrame(
        {
            "feature": explanations.feature_names,
            "shap_value": sv_row.tolist(),
        }
    )
    return (
        df.assign(abs_shap=df["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .drop(columns=["abs_shap"])
        .reset_index(drop=True)
    )


def save_shap_values(explanations: ShapExplanations, output_path: Path) -> None:
    """Persist a ShapExplanations object to a compressed ``.npz`` archive.

    Saves three arrays: ``shap_values``, ``feature_names`` (object dtype),
    and ``base_value`` (1-element array).

    Args:
        explanations: SHAP results to serialise.
        output_path: Destination path (should have ``.npz`` extension).
    """
    np.savez_compressed(
        output_path,
        shap_values=explanations.shap_values,
        feature_names=np.array(explanations.feature_names, dtype=object),
        base_value=np.array([explanations.base_value]),
    )


def load_shap_values(path: Path) -> ShapExplanations:
    """Reconstruct a ShapExplanations object from a ``.npz`` archive.

    Args:
        path: Path to the ``.npz`` file created by :func:`save_shap_values`.

    Returns:
        ShapExplanations with arrays loaded from disk.
    """
    data = np.load(path, allow_pickle=True)
    return ShapExplanations(
        shap_values=data["shap_values"],
        feature_names=data["feature_names"].tolist(),
        base_value=float(data["base_value"][0]),
    )
