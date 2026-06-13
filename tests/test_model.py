"""Tests for baseline model construction, training, and cross-validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from yield_risk.model import (
    MODEL_REGISTRY,
    SearchResult,
    build_baseline_pipeline,
    build_pipeline,
    cross_validate_model,
    model_feature_names,
    run_search,
    select_best,
    train_model,
)
from yield_risk.preprocess import SecomPreprocessor

# Small thresholds that preserve all 5 synthetic columns:
#  missing=0.6 → none dropped (no NaNs at all)
#  variance=0.0 → none dropped (all columns have positive variance)
#  correlation=0.99 → none dropped (random data has low correlation)
_THRESH = dict(
    missing_threshold=0.6,
    variance_threshold=0.0,
    correlation_threshold=0.99,
)


@pytest.fixture()
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """100-row binary dataset with positives spread across the time axis.

    Positives are distributed every 6-7 rows so that every TimeSeriesSplit
    validation fold contains at least one positive example.
    Roughly 15 positives / 85 negatives (15 % prevalence).
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(0, 1, (100, 5)),
        columns=[f"f{i}" for i in range(5)],
    )
    # Spread 15 positives evenly: indices 0, 6, 13, 20, 26, 33, 40, 46, 53,
    # 60, 66, 73, 80, 86, 93 — guaranteed coverage across the full range.
    labels = [0] * 100
    for pos_idx in range(0, 100, 7):
        labels[pos_idx] = 1
    y = pd.Series(labels)
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
        pipeline = build_pipeline(
            "logistic_regression", random_seed=42, **_THRESH
        )
        results = cross_validate_model(pipeline, X, y, cv_folds=3, random_seed=42)
        assert "test_roc_auc" in results
        assert "test_f1" in results

    def test_per_fold_scores_length(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = synthetic_data
        pipeline = build_pipeline(
            "logistic_regression", random_seed=42, **_THRESH
        )
        results = cross_validate_model(pipeline, X, y, cv_folds=3, random_seed=42)
        assert len(results["test_roc_auc"]) == 3
        assert len(results["test_f1"]) == 3

    def test_uses_forward_chaining_splits(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Every train index must be strictly earlier than every val index."""
        X, y = synthetic_data
        cv = TimeSeriesSplit(n_splits=3)
        for train_idx, val_idx in cv.split(X):
            assert int(train_idx.max()) < int(val_idx.min())

    def test_zero_positive_fold_raises_value_error(self) -> None:
        """cross_validate_model raises ValueError when a fold has no positives."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(0, 1, (30, 5)), columns=[f"f{i}" for i in range(5)])
        # All positives in the FIRST 10 rows — later CV folds will have zero positives.
        labels = [1] * 10 + [0] * 20
        y = pd.Series(labels)
        pipeline = build_pipeline("random_forest", random_seed=42, **_THRESH)
        with pytest.raises(ValueError, match="[Ff]old"):
            cross_validate_model(pipeline, X, y, cv_folds=3, random_seed=42)


class TestModelRegistry:
    def test_registry_has_four_families(self) -> None:
        assert set(MODEL_REGISTRY) == {
            "dummy",
            "logistic_regression",
            "random_forest",
            "xgboost",
        }

    def test_every_pipeline_has_preprocess_step(self) -> None:
        """Every family pipeline must include a SecomPreprocessor 'preprocess' step."""
        for name in MODEL_REGISTRY:
            pipe = build_pipeline(name, random_seed=42, **_THRESH)
            assert "preprocess" in pipe.named_steps, (
                f"{name!r} pipeline missing 'preprocess' step"
            )
            assert isinstance(pipe.named_steps["preprocess"], SecomPreprocessor)

    def test_logistic_regression_pipeline_has_scaler(self) -> None:
        pipeline = build_pipeline("logistic_regression", random_seed=42, **_THRESH)
        assert "scaler" in pipeline.named_steps

    def test_random_forest_pipeline_has_no_scaler(self) -> None:
        pipeline = build_pipeline("random_forest", random_seed=42, **_THRESH)
        assert "scaler" not in pipeline.named_steps

    def test_dummy_pipeline_has_no_scaler(self) -> None:
        pipeline = build_pipeline("dummy", random_seed=42, **_THRESH)
        assert "scaler" not in pipeline.named_steps

    def test_every_pipeline_has_classifier_step(self) -> None:
        for name in MODEL_REGISTRY:
            assert "classifier" in build_pipeline(
                name, random_seed=42, **_THRESH
            ).named_steps

    def test_step_order_preprocess_before_scaler_before_classifier(self) -> None:
        """logistic_regression: preprocess → scaler → classifier (positional order)."""
        pipe = build_pipeline("logistic_regression", random_seed=42, **_THRESH)
        step_names = [s for s, _ in pipe.steps]
        assert step_names.index("preprocess") < step_names.index("scaler")
        assert step_names.index("scaler") < step_names.index("classifier")

    def test_step_order_preprocess_before_classifier_no_scaler(self) -> None:
        """random_forest: preprocess → classifier (no scaler)."""
        pipe = build_pipeline("random_forest", random_seed=42, **_THRESH)
        step_names = [s for s, _ in pipe.steps]
        assert step_names.index("preprocess") < step_names.index("classifier")
        assert "scaler" not in step_names


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
            random_seed=42, n_jobs=1, **_THRESH,
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
            random_seed=42, n_jobs=1, **_THRESH,
        )
        assert 0.0 <= result.cv_pr_auc_mean <= 1.0

    def test_best_params_are_classifier_prefixed(
        self, synthetic_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_data
        result = run_search(
            "random_forest", X, y, _TINY_MODEL_CFG, cv_folds=3,
            random_seed=42, n_jobs=1, **_THRESH,
        )
        assert all(k.startswith("classifier__") for k in result.best_params)

    def test_small_grid_does_not_raise_when_n_iter_exceeds_combos(
        self, synthetic_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        # logistic_regression grid has only 2 combos but n_iter is 20.
        X, y = synthetic_data
        result = run_search(
            "logistic_regression", X, y, _TINY_MODEL_CFG, cv_folds=3,
            random_seed=42, n_jobs=1, **_THRESH,
        )
        assert result.name == "logistic_regression"

    def test_dummy_has_empty_best_params(
        self, synthetic_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_data
        result = run_search(
            "dummy", X, y, _TINY_MODEL_CFG, cv_folds=3,
            random_seed=42, n_jobs=1, **_THRESH,
        )
        assert result.best_params == {}
        assert hasattr(result.estimator.named_steps["classifier"], "predict_proba")

    def test_zero_positive_fold_raises_value_error(self) -> None:
        """run_search raises ValueError when a CV fold has no positives."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(0, 1, (30, 5)), columns=[f"f{i}" for i in range(5)])
        # All positives at the start — later folds will have zero positives.
        labels = [1] * 10 + [0] * 20
        y = pd.Series(labels)
        with pytest.raises(ValueError, match="[Ff]old"):
            run_search(
                "random_forest", X, y, _TINY_MODEL_CFG, cv_folds=3,
                random_seed=42, n_jobs=1, **_THRESH,
            )


class TestModelFeatureNames:
    def test_length_equals_classifier_n_features_in(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """model_feature_names length equals fitted classifier's n_features_in_."""
        X, y = synthetic_data
        pipe = build_pipeline("random_forest", random_seed=42, **_THRESH)
        pipe.fit(X, y)
        names = model_feature_names(pipe)
        assert len(names) == pipe.named_steps["classifier"].n_features_in_

    def test_returns_list_of_strings(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = synthetic_data
        pipe = build_pipeline("random_forest", random_seed=42, **_THRESH)
        pipe.fit(X, y)
        names = model_feature_names(pipe)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_names_match_preprocessor_get_feature_names_out(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = synthetic_data
        pipe = build_pipeline("random_forest", random_seed=42, **_THRESH)
        pipe.fit(X, y)
        names = model_feature_names(pipe)
        assert names == pipe.named_steps["preprocess"].get_feature_names_out()


class TestSelectBest:
    def test_returns_family_with_highest_mean(self) -> None:
        results = [
            SearchResult("a", build_pipeline("dummy", 42, **_THRESH), 0.20, 0.0, {}),
            SearchResult("b", build_pipeline("dummy", 42, **_THRESH), 0.55, 0.0, {}),
            SearchResult("c", build_pipeline("dummy", 42, **_THRESH), 0.40, 0.0, {}),
        ]
        assert select_best(results) == "b"

    def test_ties_break_toward_first(self) -> None:
        results = [
            SearchResult(
                "first", build_pipeline("dummy", 42, **_THRESH), 0.50, 0.0, {}
            ),
            SearchResult(
                "second", build_pipeline("dummy", 42, **_THRESH), 0.50, 0.0, {}
            ),
        ]
        assert select_best(results) == "first"
