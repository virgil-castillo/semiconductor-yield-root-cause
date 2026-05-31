"""Tests for baseline model construction, training, and cross-validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from yield_risk.model import (
    MODEL_REGISTRY,
    SearchResult,
    build_baseline_pipeline,
    build_pipeline,
    cross_validate_model,
    run_search,
    train_model,
)


@pytest.fixture()
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """100-row binary dataset: 85 negatives, 15 positives."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(0, 1, (100, 5)),
        columns=[f"f{i}" for i in range(5)],
    )
    y = pd.Series([0] * 85 + [1] * 15)
    return X, y


class TestBuildBaselinePipeline:
    def test_has_scaler_step(self) -> None:
        pipeline = build_baseline_pipeline(random_seed=42)
        assert isinstance(pipeline.named_steps["scaler"], StandardScaler)

    def test_has_classifier_step(self) -> None:
        pipeline = build_baseline_pipeline(random_seed=42)
        assert isinstance(
            pipeline.named_steps["classifier"], LogisticRegression
        )

    def test_classifier_uses_balanced_class_weight(self) -> None:
        pipeline = build_baseline_pipeline(random_seed=42)
        clf = pipeline.named_steps["classifier"]
        assert clf.class_weight == "balanced"

    def test_classifier_uses_given_seed(self) -> None:
        pipeline = build_baseline_pipeline(random_seed=99)
        clf = pipeline.named_steps["classifier"]
        assert clf.random_state == 99


class TestTrainModel:
    def test_pipeline_is_fitted_after_training(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = synthetic_data
        pipeline = build_baseline_pipeline(random_seed=42)
        train_model(pipeline, X, y)
        assert hasattr(pipeline.named_steps["classifier"], "coef_")

    def test_predict_proba_returns_valid_probabilities(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = synthetic_data
        pipeline = build_baseline_pipeline(random_seed=42)
        train_model(pipeline, X, y)
        proba = pipeline.predict_proba(X)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0


class TestCrossValidateModel:
    def test_returns_expected_keys(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = synthetic_data
        pipeline = build_baseline_pipeline(random_seed=42)
        results = cross_validate_model(pipeline, X, y, cv_folds=3, random_seed=42)
        assert "test_roc_auc" in results
        assert "test_f1" in results

    def test_per_fold_scores_length(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = synthetic_data
        pipeline = build_baseline_pipeline(random_seed=42)
        results = cross_validate_model(pipeline, X, y, cv_folds=3, random_seed=42)
        assert len(results["test_roc_auc"]) == 3
        assert len(results["test_f1"]) == 3


class TestModelRegistry:
    def test_registry_has_four_families(self) -> None:
        assert set(MODEL_REGISTRY) == {
            "dummy",
            "logistic_regression",
            "random_forest",
            "xgboost",
        }

    def test_logistic_regression_pipeline_has_scaler(self) -> None:
        pipeline = build_pipeline("logistic_regression", random_seed=42)
        assert "scaler" in pipeline.named_steps

    def test_random_forest_pipeline_has_no_scaler(self) -> None:
        pipeline = build_pipeline("random_forest", random_seed=42)
        assert "scaler" not in pipeline.named_steps

    def test_dummy_pipeline_has_no_scaler(self) -> None:
        pipeline = build_pipeline("dummy", random_seed=42)
        assert "scaler" not in pipeline.named_steps

    def test_every_pipeline_has_classifier_step(self) -> None:
        for name in MODEL_REGISTRY:
            assert "classifier" in build_pipeline(name, random_seed=42).named_steps


_TINY_MODEL_CFG: dict[str, object] = {
    "models": {
        "logistic_regression": {
            "C": [0.1, 1.0],
            "max_iter": 1000,
            "class_weight": "balanced",
        },
        "random_forest": {"n_estimators": [10, 20], "max_depth": [3, 5]},
        "xgboost": {"n_estimators": [10, 20], "max_depth": [2, 3]},
    },
    "search": {
        "n_iter": 20,
        "scoring": "average_precision",
        "cv_folds": 3,
        "n_jobs": 1,
        "refit": True,
    },
}


class TestRunSearch:
    def test_returns_fitted_estimator(
        self, synthetic_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_data
        result = run_search(
            "random_forest", X, y, _TINY_MODEL_CFG, cv_folds=3,
            random_seed=42, n_jobs=1,
        )
        assert isinstance(result, SearchResult)
        proba = result.estimator.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_cv_score_is_a_probability(
        self, synthetic_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_data
        result = run_search(
            "random_forest", X, y, _TINY_MODEL_CFG, cv_folds=3,
            random_seed=42, n_jobs=1,
        )
        assert 0.0 <= result.cv_pr_auc_mean <= 1.0

    def test_best_params_are_classifier_prefixed(
        self, synthetic_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_data
        result = run_search(
            "random_forest", X, y, _TINY_MODEL_CFG, cv_folds=3,
            random_seed=42, n_jobs=1,
        )
        assert all(k.startswith("classifier__") for k in result.best_params)

    def test_small_grid_does_not_raise_when_n_iter_exceeds_combos(
        self, synthetic_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        # logistic_regression grid has only 2 combos but n_iter is 20.
        X, y = synthetic_data
        result = run_search(
            "logistic_regression", X, y, _TINY_MODEL_CFG, cv_folds=3,
            random_seed=42, n_jobs=1,
        )
        assert result.name == "logistic_regression"

    def test_dummy_has_empty_best_params(
        self, synthetic_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_data
        result = run_search(
            "dummy", X, y, _TINY_MODEL_CFG, cv_folds=3,
            random_seed=42, n_jobs=1,
        )
        assert result.best_params == {}
        assert hasattr(result.estimator.named_steps["classifier"], "predict_proba")
