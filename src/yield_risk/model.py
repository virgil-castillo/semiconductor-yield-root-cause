"""Baseline model construction, training, cross-validation, and the model registry."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


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


@dataclass
class FamilySpec:
    """Registry entry describing a model family.

    Attributes:
        name: Family identifier, matching the key in model_config.yaml.
        build: Factory taking a random seed and returning an unfitted estimator.
        needs_scaling: Whether the family requires a StandardScaler step.
        tunable: Whether the family has a hyperparameter grid to search.
    """

    name: str
    build: Callable[[int], BaseEstimator]
    needs_scaling: bool
    tunable: bool


def _build_dummy(random_seed: int) -> BaseEstimator:
    """Build a stratified dummy classifier (chance floor)."""
    return DummyClassifier(strategy="stratified", random_state=random_seed)


def _build_logistic_regression(random_seed: int) -> BaseEstimator:
    """Build a class-balanced logistic regression."""
    return LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_seed,
    )


def _build_random_forest(random_seed: int) -> BaseEstimator:
    """Build a class-balanced random forest (single-threaded estimator)."""
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=1,
    )


def _build_xgboost(random_seed: int) -> BaseEstimator:
    """Build a histogram-based XGBoost classifier (single-threaded estimator).

    scale_pos_weight is set from the class ratio at search time, not here.
    """
    return XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_seed,
        n_jobs=1,
    )


MODEL_REGISTRY: dict[str, FamilySpec] = {
    "dummy": FamilySpec("dummy", _build_dummy, needs_scaling=False, tunable=False),
    "logistic_regression": FamilySpec(
        "logistic_regression",
        _build_logistic_regression,
        needs_scaling=True,
        tunable=True,
    ),
    "random_forest": FamilySpec(
        "random_forest", _build_random_forest, needs_scaling=False, tunable=True
    ),
    "xgboost": FamilySpec(
        "xgboost", _build_xgboost, needs_scaling=False, tunable=True
    ),
}


def build_pipeline(name: str, random_seed: int) -> Pipeline:
    """Build an unfitted pipeline for a registered model family.

    Adds a StandardScaler step only for families that require scaling. The
    final step is always named "classifier", so the explainability code's
    pre-final transform logic continues to work.

    Args:
        name: Family identifier present in MODEL_REGISTRY.
        random_seed: Random state passed to the estimator factory.

    Returns:
        Unfitted sklearn Pipeline.

    Raises:
        KeyError: If name is not a registered family.
    """
    spec = MODEL_REGISTRY[name]
    steps: list[tuple[str, BaseEstimator]] = []
    if spec.needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", spec.build(random_seed)))
    return Pipeline(steps)
