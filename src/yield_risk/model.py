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


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Run stratified k-fold cross-validation and return per-fold scores.

    Uses ``StratifiedKFold`` with shuffle enabled. The ``random_seed``
    parameter controls the shuffle so results are reproducible.

    Args:
        pipeline: sklearn Pipeline (fitted or unfitted; cloned internally).
        X: Feature matrix.
        y: Labels (0/1).
        cv_folds: Number of stratified folds.
        random_seed: Random state for the ``StratifiedKFold`` shuffle.

    Returns:
        Dict with keys "test_roc_auc" and "test_f1", each a numpy array
        of length cv_folds.

    Raises:
        ValueError: If any validation fold contains zero positive examples.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    _check_positive_per_fold(y, cv)
    results = cross_validate(
        pipeline, X, y, cv=cv, scoring=["roc_auc", "f1"]
    )
    return cast(dict[str, np.ndarray], results)


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


def selection_score(result: SearchResult, std_penalty: float = 1.0) -> float:
    """Compute the std-penalized CV PR-AUC score used to rank model families.

    The score is a lower-confidence bound on cross-validated PR-AUC::

        score = cv_pr_auc_mean - std_penalty * cv_pr_auc_std

    Penalizing the mean by the per-fold standard deviation rewards families
    whose performance is *consistent* across the stratified folds and
    discounts families whose mean is propped up by a single lucky fold.
    Subtracting the std collapses the advantage of high-variance families
    whose mean is inflated by one unusually favorable fold assignment.

    Args:
        result: A single family's search result.
        std_penalty: How many standard deviations to subtract from the mean.
            ``1.0`` corresponds to a one-sigma lower bound; larger values
            penalize instability more aggressively.

    Returns:
        The std-penalized score (may be negative).
    """
    return result.cv_pr_auc_mean - std_penalty * result.cv_pr_auc_std


def select_best(results: list[SearchResult], std_penalty: float = 1.0) -> str:
    """Return the family name with the best std-penalized cross-validated PR-AUC.

    Selection uses a lower-confidence bound, ``cv_pr_auc_mean -
    std_penalty * cv_pr_auc_std`` (see :func:`selection_score`), rather than
    the raw mean. Ranking on the raw mean can select a family whose high average
    is driven by a single lucky fold while every other fold is mediocre.
    Subtracting the per-fold standard deviation favors the family with the most
    consistent per-fold PR-AUC across the stratified folds, which empirically
    tracks held-out generalization far better.

    Ties break toward the family appearing first in *results* (which callers
    pass in registry order).

    Args:
        results: Per-family search results.
        std_penalty: Standard-deviation penalty forwarded to
            :func:`selection_score`. Defaults to ``1.0`` (one-sigma lower bound).

    Returns:
        The winning family's name.

    Raises:
        ValueError: If *results* is empty.
    """
    if not results:
        raise ValueError("select_best requires at least one SearchResult.")
    return max(results, key=lambda r: selection_score(r, std_penalty)).name


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
