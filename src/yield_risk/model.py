"""Baseline model construction, training, and cross-validation."""
from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline_pipeline(random_seed: int) -> Pipeline:
    """Construct a StandardScaler -> LogisticRegression pipeline.

    The classifier uses class_weight="balanced" to handle the ~6.6% fail-rate
    imbalance. Solver is "lbfgs" with max_iter=1000.

    Args:
        random_seed: Random state for the LogisticRegression.

    Returns:
        Unfitted sklearn Pipeline with steps "scaler" and "classifier".
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit a sklearn Pipeline on training features and labels.

    Args:
        pipeline: Unfitted sklearn Pipeline.
        X_train: Training feature matrix (sensor columns only).
        y_train: Training labels (0/1).

    Returns:
        The same Pipeline object, now fitted.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Run stratified k-fold cross-validation and return per-fold scores.

    Uses sklearn.model_selection.cross_validate with
    scoring=["roc_auc", "f1"]. The pipeline is cloned internally per fold.

    Args:
        pipeline: sklearn Pipeline (fitted or unfitted; cloned internally).
        X: Feature matrix.
        y: Labels.
        cv_folds: Number of stratified folds.
        random_seed: Random state for StratifiedKFold.

    Returns:
        Dict with keys "test_roc_auc" and "test_f1", each a numpy array
        of length cv_folds.
    """
    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_seed
    )
    results = cross_validate(
        pipeline, X, y, cv=cv, scoring=["roc_auc", "f1"]
    )
    return cast(dict[str, np.ndarray], results)
