"""Baseline model construction, training, cross-validation, and the model registry."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from yield_risk.config import CostMatrix, ThresholdSearchConfig
from yield_risk.preprocess import SecomPreprocessor
from yield_risk.thresholding import find_optimal_threshold


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
    cv: StratifiedKFold,
) -> None:
    """Raise ValueError if any validation fold contains zero positive examples.

    Args:
        y: Label series.
        cv: StratifiedKFold splitter to check.

    Raises:
        ValueError: If a validation fold has no positive examples, naming the
            zero-indexed fold number.
    """
    y_arr = np.asarray(y)
    for fold_idx, (_, val_idx) in enumerate(
        cv.split(np.zeros(len(y_arr)), y_arr)
    ):
        n_pos = int((y_arr[val_idx] == 1).sum())
        if n_pos == 0:
            raise ValueError(
                f"Fold {fold_idx} validation split contains zero positive "
                "examples. PR-AUC is undefined. Ensure positives are spread "
                "across the dataset before calling cross-validation."
            )


def compute_fold_diagnostics(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_seed: int,
) -> list[dict[str, float | int]]:
    """Compute per-fold validation diagnostics using stratified k-fold CV.

    Clones the estimator for each fold, fits on the training indices, and
    evaluates on the validation indices. Raises before any fitting if any
    validation fold contains zero positive examples.

    Args:
        estimator: Unfitted (or previously fitted) sklearn Pipeline. It is
            cloned internally; the original is not mutated.
        X: Feature matrix.
        y: Labels (0/1).
        cv_folds: Number of stratified folds for ``StratifiedKFold``.
        random_seed: Random state for the ``StratifiedKFold`` shuffle,
            ensuring reproducible fold assignments.

    Returns:
        List of length ``cv_folds``. Each dict has keys:

        - ``fold`` (int): Zero-based fold index.
        - ``val_prevalence`` (float): Fraction of positive labels in the
          validation split.
        - ``roc_auc`` (float): ROC-AUC on the validation split.
        - ``pr_auc`` (float): PR-AUC (average precision) on the validation
          split.

    Raises:
        ValueError: If any validation fold contains zero positive examples
            (PR-AUC is undefined in that case).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    _check_positive_per_fold(y, cv)

    diagnostics: list[dict[str, float | int]] = []
    y_arr = np.asarray(y)
    X_vals = X.values

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_vals, y_arr)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y_arr[val_idx]

        fold_estimator: Pipeline = clone(estimator)
        fold_estimator.fit(X_train_fold, y_train_fold)
        y_prob = fold_estimator.predict_proba(X_val_fold)[:, 1]

        diagnostics.append(
            {
                "fold": fold_idx,
                "val_prevalence": float(y_val_fold.mean()),
                "roc_auc": float(roc_auc_score(y_val_fold, y_prob)),
                "pr_auc": float(average_precision_score(y_val_fold, y_prob)),
            }
        )

    return diagnostics


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
    cv_threshold: float,
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
        cv_threshold: Passed to SecomPreprocessor — drop columns with
            coefficient of variation strictly below this.
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
                cv_threshold=cv_threshold,
                correlation_threshold=correlation_threshold,
            ),
        ),
    ]
    if spec.needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", spec.build(random_seed)))
    return Pipeline(steps)


def frozen_operating_threshold(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_seed: int,
    cost_matrix: CostMatrix,
    threshold_search: ThresholdSearchConfig,
) -> float:
    """Derive the cost-optimal operating threshold from pooled OOF CV predictions.

    Runs ``cross_val_predict`` over a ``StratifiedKFold`` to obtain out-of-fold
    probability scores for every training sample, then calls
    ``find_optimal_threshold`` on those pooled predictions. The embedded
    ``SecomPreprocessor`` is refit on each fold's training split inside
    ``cross_val_predict``.

    Args:
        estimator: Unfitted (or fitted) sklearn Pipeline.  It is cloned
            internally by ``cross_val_predict`` so the original is not mutated.
        X: Training feature matrix.
        y: Training labels (0/1).
        cv_folds: Number of stratified folds for ``StratifiedKFold``.
        random_seed: Random state for the ``StratifiedKFold`` shuffle,
            ensuring reproducible fold assignments and identical OOF scores
            across calls with the same seed.
        cost_matrix: Per-outcome costs used by ``find_optimal_threshold``.
        threshold_search: Grid parameters (low, high, steps) for the threshold
            search.

    Returns:
        The threshold (float) that minimises expected cost on pooled OOF
        predictions.  The value lies in
        ``[threshold_search.low, threshold_search.high]``.
    """
    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_seed
    )
    oof = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
    return find_optimal_threshold(
        np.asarray(y), oof[:, 1], cost_matrix, threshold_search
    ).threshold


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

    Raises:
        ValueError: If *results* is empty.
    """
    if not results:
        raise ValueError("select_best requires at least one SearchResult.")
    return max(results, key=lambda r: r.cv_pr_auc_mean).name


def run_search(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    model_cfg: dict[str, Any],
    cv_folds: int,
    random_seed: int,
    missing_threshold: float,
    cv_threshold: float,
    correlation_threshold: float,
    n_jobs: int = -1,
) -> SearchResult:
    """Tune one model family with stratified CV PR-AUC and refit the winner.

    Uses ``StratifiedKFold`` (shuffle=True) for both hyperparameter search and
    the dummy baseline. Raises if any validation fold contains zero positive
    examples.

    Tunable families are searched with ``RandomizedSearchCV``; ``n_iter`` is
    capped at the size of the discrete grid so small grids do not raise. The
    dummy family has no grid: it is fit directly and scored with
    ``cross_val_score`` so it still appears as the chance floor. For XGBoost,
    ``scale_pos_weight`` is set from the training class ratio before searching.

    Args:
        name: Registered family identifier.
        X: Training feature matrix.
        y: Training labels (0/1).
        model_cfg: Parsed model_config.yaml (keys ``models`` and ``search``).
        cv_folds: Number of stratified folds.
        random_seed: Random state for the estimator, the StratifiedKFold
            shuffle, and RandomizedSearchCV.
        missing_threshold: Passed through to build_pipeline / SecomPreprocessor.
        cv_threshold: Passed through to build_pipeline / SecomPreprocessor.
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
        cv_threshold=cv_threshold,
        correlation_threshold=correlation_threshold,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
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
