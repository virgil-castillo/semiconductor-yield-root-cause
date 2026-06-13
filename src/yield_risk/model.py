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
    TimeSeriesSplit,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from yield_risk.preprocess import SecomPreprocessor


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


def _check_positive_per_fold(
    y: pd.Series,
    cv: TimeSeriesSplit,
) -> None:
    """Raise ValueError if any validation fold contains zero positive examples.

    Args:
        y: Label series in time order.
        cv: TimeSeriesSplit splitter to check.

    Raises:
        ValueError: If a validation fold has no positive examples, naming the
            zero-indexed fold number.
    """
    y_arr = np.asarray(y)
    for fold_idx, (_, val_idx) in enumerate(cv.split(y_arr)):
        n_pos = int((y_arr[val_idx] == 1).sum())
        if n_pos == 0:
            raise ValueError(
                f"Fold {fold_idx} validation split contains zero positive "
                "examples. PR-AUC is undefined. Ensure positives are spread "
                "across the full time axis before calling cross-validation."
            )


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Run forward-chaining time-series cross-validation and return per-fold scores.

    Uses ``TimeSeriesSplit`` (no shuffle, no stratification). Callers must pass
    ``X``/``y`` in time order; the earliest rows form the first training block.
    ``random_seed`` is retained in the signature for caller API compatibility but
    is not consumed by ``TimeSeriesSplit`` (which has no ``random_state``).

    Args:
        pipeline: sklearn Pipeline (fitted or unfitted; cloned internally).
        X: Feature matrix in time order.
        y: Labels in time order.
        cv_folds: Number of forward-chaining folds.
        random_seed: Unused — kept for API compatibility with callers that pass
            a seed for reproducibility.

    Returns:
        Dict with keys "test_roc_auc" and "test_f1", each a numpy array
        of length cv_folds.

    Raises:
        ValueError: If any validation fold contains zero positive examples.
    """
    _ = random_seed  # retained for API compatibility; TimeSeriesSplit has no RNG
    cv = TimeSeriesSplit(n_splits=cv_folds)
    _check_positive_per_fold(y, cv)
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


def build_pipeline(
    name: str,
    random_seed: int,
    missing_threshold: float,
    variance_threshold: float,
    correlation_threshold: float,
) -> Pipeline:
    """Build an unfitted pipeline for a registered model family.

    Prepends a ``SecomPreprocessor`` step (named ``"preprocess"``) for every
    family. Adds a ``StandardScaler`` step only for families that require
    scaling. Step order: ``preprocess`` → (``scaler``) → ``classifier``.

    Args:
        name: Family identifier present in MODEL_REGISTRY.
        random_seed: Random state passed to the estimator factory.
        missing_threshold: Passed to SecomPreprocessor — drop columns with
            missing fraction strictly above this.
        variance_threshold: Passed to SecomPreprocessor — drop columns with
            variance strictly below this.
        correlation_threshold: Passed to SecomPreprocessor — drop the later of
            each pair with absolute correlation strictly above this.

    Returns:
        Unfitted sklearn Pipeline.

    Raises:
        KeyError: If name is not a registered family.
    """
    spec = MODEL_REGISTRY[name]
    steps: list[tuple[str, BaseEstimator]] = [
        (
            "preprocess",
            SecomPreprocessor(
                missing_threshold=missing_threshold,
                variance_threshold=variance_threshold,
                correlation_threshold=correlation_threshold,
            ),
        ),
    ]
    if spec.needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", spec.build(random_seed)))
    return Pipeline(steps)


def model_feature_names(pipeline: Pipeline) -> list[str]:
    """Return feature names output by the pipeline's preprocess step.

    Args:
        pipeline: Fitted sklearn Pipeline containing a ``"preprocess"`` step
            (SecomPreprocessor).

    Returns:
        List of kept column name strings from the fitted preprocessor.

    Raises:
        sklearn.exceptions.NotFittedError: If the preprocess step has not been
            fitted yet.
        KeyError: If the pipeline has no ``"preprocess"`` step.
    """
    preprocessor: SecomPreprocessor = pipeline.named_steps["preprocess"]
    return list(preprocessor.get_feature_names_out())


@dataclass
class SearchResult:
    """Outcome of tuning one model family.

    Attributes:
        name: Family identifier.
        estimator: Refit best pipeline (fitted on all of X, y).
        cv_pr_auc_mean: Mean cross-validated PR-AUC (average precision).
        cv_pr_auc_std: Standard deviation of cross-validated PR-AUC.
        best_params: Chosen hyperparameters (classifier__*), empty for dummy.
    """

    name: str
    estimator: Pipeline
    cv_pr_auc_mean: float
    cv_pr_auc_std: float
    best_params: dict[str, Any]


def _param_distributions(grid: dict[str, Any]) -> dict[str, list[Any]]:
    """Build RandomizedSearchCV distributions from a family grid.

    Only list-valued entries are searched; scalar entries (e.g. a fixed
    max_iter) are baked into the estimator factory and ignored here. Keys are
    prefixed with ``classifier__`` to target the pipeline's final step.

    Args:
        grid: Per-family hyperparameter grid from model_config.yaml.

    Returns:
        Mapping of ``classifier__<param>`` to a list of candidate values.
    """
    return {
        f"classifier__{key}": value
        for key, value in grid.items()
        if isinstance(value, list)
    }


def select_best(results: list[SearchResult]) -> str:
    """Return the family name with the highest mean cross-validated PR-AUC.

    Ties break toward the family appearing first in *results* (which callers
    pass in registry order).

    Args:
        results: Per-family search results.

    Returns:
        The winning family's name.
    """
    return max(results, key=lambda r: r.cv_pr_auc_mean).name


def run_search(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    model_cfg: dict[str, Any],
    cv_folds: int,
    random_seed: int,
    missing_threshold: float,
    variance_threshold: float,
    correlation_threshold: float,
    n_jobs: int = -1,
) -> SearchResult:
    """Tune one model family with forward-chaining CV PR-AUC and refit the winner.

    Uses ``TimeSeriesSplit`` for both hyperparameter search and the dummy
    baseline. Callers must pass ``X``/``y`` in time order. Raises if any
    validation fold contains zero positive examples.

    Tunable families are searched with ``RandomizedSearchCV``; ``n_iter`` is
    capped at the size of the discrete grid so small grids do not raise. The
    dummy family has no grid: it is fit directly and scored with
    ``cross_val_score`` so it still appears as the chance floor. For XGBoost,
    ``scale_pos_weight`` is set from the training class ratio before searching.

    Args:
        name: Registered family identifier.
        X: Training feature matrix in time order.
        y: Training labels (0/1) in time order.
        model_cfg: Parsed model_config.yaml (keys ``models`` and ``search``).
        cv_folds: Number of forward-chaining folds.
        random_seed: Random state for the estimator and RandomizedSearchCV.
        missing_threshold: Passed through to build_pipeline / SecomPreprocessor.
        variance_threshold: Passed through to build_pipeline / SecomPreprocessor.
        correlation_threshold: Passed through to build_pipeline / SecomPreprocessor.
        n_jobs: Parallel jobs for the search (bound to the CPU allocation).

    Returns:
        SearchResult with the refit best estimator and its CV PR-AUC.

    Raises:
        ValueError: If any validation fold contains zero positive examples.
    """
    spec = MODEL_REGISTRY[name]
    pipeline = build_pipeline(
        name,
        random_seed,
        missing_threshold=missing_threshold,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
    )
    cv = TimeSeriesSplit(n_splits=cv_folds)
    _check_positive_per_fold(y, cv)

    if name == "xgboost":
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        scale = (n_neg / n_pos) if n_pos else 1.0
        pipeline.set_params(classifier__scale_pos_weight=scale)

    if not spec.tunable:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="average_precision")
        pipeline.fit(X, y)
        return SearchResult(
            name=name,
            estimator=pipeline,
            cv_pr_auc_mean=float(scores.mean()),
            cv_pr_auc_std=float(scores.std()),
            best_params={},
        )

    grid = model_cfg["models"][name]
    search_cfg = model_cfg["search"]
    distributions = _param_distributions(grid)
    n_combos = prod(len(v) for v in distributions.values()) if distributions else 1
    n_iter = min(int(search_cfg["n_iter"]), n_combos)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        random_state=random_seed,
    )
    search.fit(X, y)
    std = float(search.cv_results_["std_test_score"][search.best_index_])
    return SearchResult(
        name=name,
        estimator=cast(Pipeline, search.best_estimator_),
        cv_pr_auc_mean=float(search.best_score_),
        cv_pr_auc_std=std,
        best_params=dict(search.best_params_),
    )
