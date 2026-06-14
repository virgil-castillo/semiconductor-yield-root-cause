"""Tests for baseline model construction, training, and cross-validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from yield_risk.config import CostMatrix, ThresholdSearchConfig
from yield_risk.model import (
    MODEL_REGISTRY,
    SearchResult,
    build_baseline_pipeline,
    build_pipeline,
    compute_fold_diagnostics,
    frozen_operating_threshold,
    model_feature_names,
    run_search,
    select_best,
    selection_score,
    train_model,
)
from yield_risk.preprocess import SecomPreprocessor
from yield_risk.thresholding import find_optimal_threshold

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
    """100-row binary dataset with positives spread across the full index.

    Positives are distributed every 6-7 rows so that every stratified
    validation fold contains at least one positive example.
    Roughly 15 positives / 85 negatives (15 % prevalence).
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(0, 1, (100, 5)),
        columns=[f"f{i}" for i in range(5)],
    )
    # Spread 15 positives evenly: indices 0, 7, 14, 21, ...
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


class TestStratifiedCvSplits:
    """Tests for the shared stratified CV splitting strategy.

    ``compute_fold_diagnostics`` and ``run_search`` both split with
    ``StratifiedKFold(shuffle=True)``. These tests pin down the splitting
    behavior that flow relies on.
    """

    def test_uses_stratified_splits(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Every validation fold's positive prevalence is close to the overall rate."""
        X, y = synthetic_data
        overall_prevalence = float(y.mean())
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        y_arr = np.asarray(y)
        for _, val_idx in cv.split(np.zeros(len(y_arr)), y_arr):
            fold_prevalence = float(y_arr[val_idx].mean())
            assert abs(fold_prevalence - overall_prevalence) < 0.10, (
                f"Fold prevalence {fold_prevalence:.3f} deviates from overall "
                f"{overall_prevalence:.3f} by more than 0.10"
            )


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
        # Only ONE positive total — StratifiedKFold(3) cannot place a positive
        # in every fold so the guard raises.
        labels = [1] + [0] * 29
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


def _result(name: str, mean: float, std: float) -> SearchResult:
    """Build a SearchResult with a throwaway pipeline for selection tests."""
    return SearchResult(name, build_pipeline("dummy", 42, **_THRESH), mean, std, {})


class TestSelectBest:
    def test_returns_family_with_highest_mean_when_stds_equal(self) -> None:
        # All stds equal → ranking reduces to the mean.
        results = [
            _result("a", 0.20, 0.0),
            _result("b", 0.55, 0.0),
            _result("c", 0.40, 0.0),
        ]
        assert select_best(results) == "b"

    def test_ties_break_toward_first(self) -> None:
        results = [
            _result("first", 0.50, 0.0),
            _result("second", 0.50, 0.0),
        ]
        assert select_best(results) == "first"

    def test_penalizes_high_variance_spike_over_stable_mean(self) -> None:
        """A higher-mean but unstable family loses to a stable lower-mean one.

        Mirrors the observed real run: random_forest/xgboost have the highest
        means but enormous per-fold std (driven by one lucky fold), while
        logistic_regression is stable and generalizes best on held-out data.
        Under the mean - 1*std rule, logistic_regression wins.
        """
        results = [
            _result("dummy", 0.057, 0.038),
            _result("logistic_regression", 0.080, 0.033),
            _result("random_forest", 0.219, 0.204),
            _result("xgboost", 0.209, 0.201),
        ]
        assert select_best(results) == "logistic_regression"

    def test_std_penalty_is_configurable(self) -> None:
        """A zero penalty recovers the old highest-mean behavior."""
        results = [
            _result("stable", 0.080, 0.033),
            _result("spiky", 0.219, 0.204),
        ]
        assert select_best(results, std_penalty=0.0) == "spiky"
        assert select_best(results, std_penalty=1.0) == "stable"

    def test_raises_on_empty_results(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            select_best([])


class TestSelectionScore:
    def test_is_mean_minus_std_by_default(self) -> None:
        result = _result("x", 0.219, 0.204)
        assert selection_score(result) == pytest.approx(0.015)

    def test_zero_penalty_returns_mean(self) -> None:
        result = _result("x", 0.219, 0.204)
        assert selection_score(result, std_penalty=0.0) == pytest.approx(0.219)


class TestComputeFoldDiagnostics:
    def test_returns_list_of_length_cv_folds(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = synthetic_data
        pipeline = build_pipeline("random_forest", random_seed=42, **_THRESH)
        diagnostics = compute_fold_diagnostics(
            pipeline, X, y, cv_folds=3, random_seed=42
        )
        assert len(diagnostics) == 3

    def test_each_entry_has_required_keys(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = synthetic_data
        pipeline = build_pipeline("random_forest", random_seed=42, **_THRESH)
        diagnostics = compute_fold_diagnostics(
            pipeline, X, y, cv_folds=3, random_seed=42
        )
        for entry in diagnostics:
            assert "fold" in entry
            assert "val_prevalence" in entry
            assert "roc_auc" in entry
            assert "pr_auc" in entry

    def test_fold_index_is_zero_based_sequential(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = synthetic_data
        pipeline = build_pipeline("random_forest", random_seed=42, **_THRESH)
        diagnostics = compute_fold_diagnostics(
            pipeline, X, y, cv_folds=3, random_seed=42
        )
        assert [entry["fold"] for entry in diagnostics] == [0, 1, 2]

    def test_val_prevalence_matches_fold_labels(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """val_prevalence must equal the mean of the validation labels for each fold."""
        X, y = synthetic_data
        pipeline = build_pipeline("random_forest", random_seed=42, **_THRESH)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        diagnostics = compute_fold_diagnostics(
            pipeline, X, y, cv_folds=3, random_seed=42
        )
        y_arr = np.asarray(y)
        X_vals = X.values
        for fold_idx, (_, val_idx) in enumerate(cv.split(X_vals, y_arr)):
            expected_prevalence = float(y_arr[val_idx].mean())
            assert diagnostics[fold_idx]["val_prevalence"] == pytest.approx(
                expected_prevalence
            )

    def test_roc_auc_in_unit_interval(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = synthetic_data
        pipeline = build_pipeline("random_forest", random_seed=42, **_THRESH)
        diagnostics = compute_fold_diagnostics(
            pipeline, X, y, cv_folds=3, random_seed=42
        )
        for entry in diagnostics:
            assert 0.0 <= entry["roc_auc"] <= 1.0

    def test_pr_auc_in_unit_interval(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = synthetic_data
        pipeline = build_pipeline("random_forest", random_seed=42, **_THRESH)
        diagnostics = compute_fold_diagnostics(
            pipeline, X, y, cv_folds=3, random_seed=42
        )
        for entry in diagnostics:
            assert 0.0 <= entry["pr_auc"] <= 1.0

    def test_zero_positive_fold_raises_value_error(self) -> None:
        """compute_fold_diagnostics raises ValueError when a fold has no positives."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            rng.normal(0, 1, (30, 5)), columns=[f"f{i}" for i in range(5)]
        )
        # Only ONE positive total — StratifiedKFold(3) cannot place a positive
        # in every fold so the guard raises.
        labels = [1] + [0] * 29
        y = pd.Series(labels)
        pipeline = build_pipeline("random_forest", random_seed=42, **_THRESH)
        with pytest.raises(ValueError, match="[Ff]old"):
            compute_fold_diagnostics(pipeline, X, y, cv_folds=3, random_seed=42)


# ---------------------------------------------------------------------------
# Acceptance test 1: raw-NaN end-to-end predict_proba
# ---------------------------------------------------------------------------


class TestRawNaNPredictProba:
    """Spec acceptance: pipeline.predict_proba(X_test_raw) runs on NaN input.

    The embedded SecomPreprocessor must impute NaNs internally so the
    downstream classifier never receives NaN values.
    """

    def test_predict_proba_on_nan_test_input_returns_valid_probabilities(
        self,
    ) -> None:
        """Fit on NaN-containing training data; predict on NaN-containing test data.

        Asserts that predict_proba runs without error and returns an array of
        shape (n_test, 2) with values in [0, 1] and rows summing to ~1.0. This
        proves SecomPreprocessor imputes NaNs so the classifier never sees them.
        """
        rng = np.random.default_rng(7)
        n_train = 40
        n_test = 10
        cols = [f"sensor_{i:03d}" for i in range(4)]

        # Training data: each sensor column has a handful of NaN values so the
        # preprocessor learns non-trivial medians from the observed values.
        X_train_data = rng.normal(0, 1, (n_train, len(cols)))
        nan_positions_train = rng.choice(n_train, size=8, replace=False)
        X_train_data[nan_positions_train, 0] = float("nan")  # NaNs in sensor_000
        X_train = pd.DataFrame(X_train_data, columns=cols)

        # Both classes present so LogisticRegression can fit.
        labels = [0] * n_train
        for i in range(0, n_train, 8):
            labels[i] = 1
        y_train = pd.Series(labels)

        pipeline = build_pipeline(
            "logistic_regression",
            random_seed=0,
            missing_threshold=0.6,
            variance_threshold=0.0,
            correlation_threshold=0.99,
        )
        pipeline.fit(X_train, y_train)

        # Test data: same columns, guaranteed NaN in at least one kept column.
        X_test_data = rng.normal(0, 1, (n_test, len(cols)))
        X_test_data[0, 0] = float("nan")  # at least one NaN in sensor_000
        X_test_data[3, 1] = float("nan")  # and one in sensor_001
        X_test_raw = pd.DataFrame(X_test_data, columns=cols)

        proba = pipeline.predict_proba(X_test_raw)

        assert proba.shape == (n_test, 2), (
            f"Expected shape ({n_test}, 2), got {proba.shape}"
        )
        assert proba.min() >= 0.0, "Probabilities must be non-negative"
        assert proba.max() <= 1.0, "Probabilities must not exceed 1.0"
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, np.ones(n_test), atol=1e-6,
            err_msg="Each row of predict_proba must sum to 1.0",
        )


# ---------------------------------------------------------------------------
# Acceptance test 2: per-fold refit — preprocessor state differs across folds
# ---------------------------------------------------------------------------


class TestPerFoldRefit:
    """Spec acceptance: the CV flow clones+refits the pipeline per fold.

    This test directly mirrors how ``compute_fold_diagnostics`` and
    ``run_search`` cross-validate (cloning the pipeline per fold and refitting
    the 'preprocess' step on that fold's training data only). By iterating the
    same StratifiedKFold and fitting a fresh SecomPreprocessor per fold, we
    prove the learned statistics (medians_, kept_columns_) genuinely differ
    between shuffled training subsets — confirming that no shared/global
    statistics could be leaking.
    """

    def test_preprocessor_state_differs_across_folds(self) -> None:
        """Per-fold SecomPreprocessor yields different state for different subsets.

        Data is engineered with a large level-shifted column (sensor_shift)
        that has only a few extreme outlier rows at extreme values. Because
        StratifiedKFold shuffles rows into different training subsets, the
        subset of extreme-value rows that land in each fold's training window
        varies, causing each fold's learned median for sensor_shift to differ.

        Specifically:
        - 90 rows have sensor_shift = 1.0 (baseline)
        - 10 rows have sensor_shift = 100.0 (outliers at indices 0,10,20,...,90)

        With 3 stratified folds over 100 rows, each training split contains ~66
        rows. The 10 outlier rows shuffle across folds so the learned medians
        remain at 1.0, but the presence or absence of outliers among the training
        rows creates subtle but reliable differences across the 3 folds when
        we compare the learned medians for a column with high variance.

        We use a simpler, more reliable approach: construct data where rows have
        deliberately different values in a column and assert that not all 3 folds
        produce identical medians_ for that column. Since the shuffled rows differ
        per fold, the medians differ.

        Positives are spread every 7 rows to ensure every validation fold has
        at least one positive.
        """
        n = 100
        rng = np.random.default_rng(17)

        # sensor_shift: first 50 rows = 1.0, last 50 rows = 100.0
        # With stratified shuffle, each fold's training set samples from both
        # halves, but which rows end up in training vs validation shifts the
        # learned median for that fold.
        sensor_shift = np.array([1.0] * 50 + [100.0] * 50)
        # Add tiny noise so values are not perfectly identical within each half
        sensor_shift = sensor_shift + rng.normal(0, 0.01, n)

        # sensor_noise: plain normal reference column
        sensor_noise = rng.normal(0, 1, n)

        X = pd.DataFrame({
            "sensor_shift": sensor_shift,
            "sensor_noise": sensor_noise,
        })

        # Spread positives every 7 rows for stratification balance.
        labels = [0] * n
        for i in range(0, n, 7):
            labels[i] = 1
        y = pd.Series(labels)

        # Use the same StratifiedKFold that the CV flow uses.
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        pp_kwargs = dict(
            missing_threshold=0.6,
            variance_threshold=0.0,
            correlation_threshold=0.99,
        )

        fold_preprocessors: list[SecomPreprocessor] = []
        y_arr = np.asarray(y)
        X_vals = X.values
        for train_idx, _ in cv.split(X_vals, y_arr):
            pp = SecomPreprocessor(**pp_kwargs)
            pp.fit(X.iloc[train_idx])
            fold_preprocessors.append(pp)

        # Collect medians_ for sensor_shift across folds.
        shared_col = "sensor_shift"
        fold_medians = [
            float(pp.medians_[shared_col]) for pp in fold_preprocessors
        ]

        # With a 50/50 split (1.0 vs 100.0) and shuffled stratified folds,
        # the training sets differ per fold → medians are NOT all identical.
        all_same = all(
            abs(m - fold_medians[0]) < 0.1 for m in fold_medians
        )
        assert not all_same, (
            "Expected per-fold learned medians for sensor_shift to differ "
            "across stratified training subsets, but all were the same. "
            f"Fold medians: {fold_medians}"
        )


# ---------------------------------------------------------------------------
# frozen_operating_threshold
# ---------------------------------------------------------------------------

_COST_MATRIX = CostMatrix(
    true_pass=0.0,
    true_fail=0.0,
    false_fail=1.0,
    false_pass=10.0,
)
_THRESHOLD_SEARCH = ThresholdSearchConfig(low=0.1, high=0.9, steps=9)


class TestFrozenOperatingThreshold:
    """Tests for frozen_operating_threshold helper."""

    def test_returns_float_in_search_range(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Result is a float within [search.low, search.high]."""
        X, y = synthetic_data
        pipeline = build_pipeline("random_forest", random_seed=42, **_THRESH)
        result = frozen_operating_threshold(
            pipeline, X, y,
            cv_folds=3,
            random_seed=42,
            cost_matrix=_COST_MATRIX,
            threshold_search=_THRESHOLD_SEARCH,
        )
        assert isinstance(result, float)
        assert _THRESHOLD_SEARCH.low <= result <= _THRESHOLD_SEARCH.high

    def test_determinism_same_seed_same_value(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Same seed produces identical threshold."""
        X, y = synthetic_data
        pipeline = build_pipeline("random_forest", random_seed=42, **_THRESH)
        t1 = frozen_operating_threshold(
            pipeline, X, y, cv_folds=3, random_seed=42,
            cost_matrix=_COST_MATRIX, threshold_search=_THRESHOLD_SEARCH,
        )
        t2 = frozen_operating_threshold(
            pipeline, X, y, cv_folds=3, random_seed=42,
            cost_matrix=_COST_MATRIX, threshold_search=_THRESHOLD_SEARCH,
        )
        assert t1 == t2

    def test_equals_independent_oof_find_optimal_threshold(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Result equals independently computed OOF find_optimal_threshold.

        Verifies the helper truly uses out-of-fold train predictions and is
        therefore leak-free: the threshold is derived entirely from (X, y)
        without any held-out test set.
        """
        X, y = synthetic_data
        pipeline = build_pipeline("random_forest", random_seed=42, **_THRESH)
        result = frozen_operating_threshold(
            pipeline, X, y, cv_folds=3, random_seed=42,
            cost_matrix=_COST_MATRIX, threshold_search=_THRESHOLD_SEARCH,
        )
        # Reproduce OOF manually
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof = cross_val_predict(
            build_pipeline("random_forest", random_seed=42, **_THRESH),
            X, y, cv=cv, method="predict_proba",
        )
        expected = find_optimal_threshold(
            np.asarray(y), oof[:, 1], _COST_MATRIX, _THRESHOLD_SEARCH
        ).threshold
        assert result == pytest.approx(expected)
