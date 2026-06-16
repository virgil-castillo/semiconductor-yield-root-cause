"""Tests for feature importance extraction and visualization."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yield_risk.importance import (
    extract_feature_importance,
    extract_lr_coefficients,
    plot_top_features,
)
from yield_risk.model import build_pipeline, model_feature_names


@pytest.fixture()
def fitted_pipeline() -> Pipeline:
    """Three-feature pipeline fitted on small synthetic data."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(0, 1, (50, 3)), columns=["f0", "f1", "f2"]
    )
    y = pd.Series([0] * 42 + [1] * 8)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", random_state=42),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


class TestExtractLrCoefficients:
    def test_returns_all_features(self, fitted_pipeline: Pipeline) -> None:
        feature_names = ["f0", "f1", "f2"]
        result = extract_lr_coefficients(fitted_pipeline, feature_names)
        assert len(result) == len(feature_names)

    def test_sorted_by_abs_coefficient_descending(
        self, fitted_pipeline: Pipeline
    ) -> None:
        feature_names = ["f0", "f1", "f2"]
        result = extract_lr_coefficients(fitted_pipeline, feature_names)
        diffs = result["abs_coefficient"].diff().dropna()
        assert (diffs <= 0).all()

    def test_has_required_columns(self, fitted_pipeline: Pipeline) -> None:
        feature_names = ["f0", "f1", "f2"]
        result = extract_lr_coefficients(fitted_pipeline, feature_names)
        assert set(result.columns) == {"feature", "coefficient", "abs_coefficient"}


@pytest.fixture()
def fitted_tree_pipeline() -> Pipeline:
    """Three-feature RandomForest pipeline fitted on small synthetic data."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(0, 1, (50, 3)), columns=["f0", "f1", "f2"]
    )
    y = pd.Series([0] * 42 + [1] * 8)
    pipeline = Pipeline(
        [
            (
                "classifier",
                RandomForestClassifier(n_estimators=10, random_state=0),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture()
def fitted_dummy_pipeline() -> Pipeline:
    """Three-feature DummyClassifier pipeline.

    Has neither coef_ nor feature_importances_, so it exercises the TypeError path.
    """
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        rng.normal(0, 1, (50, 3)), columns=["f0", "f1", "f2"]
    )
    y = pd.Series([0] * 42 + [1] * 8)
    pipeline = Pipeline(
        [
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


class TestExtractFeatureImportance:
    def test_tree_pipeline_returns_all_features(
        self, fitted_tree_pipeline: Pipeline
    ) -> None:
        feature_names = ["f0", "f1", "f2"]
        result = extract_feature_importance(fitted_tree_pipeline, feature_names)
        assert len(result) == len(feature_names)

    def test_tree_pipeline_has_required_columns(
        self, fitted_tree_pipeline: Pipeline
    ) -> None:
        feature_names = ["f0", "f1", "f2"]
        result = extract_feature_importance(fitted_tree_pipeline, feature_names)
        assert set(result.columns) == {"feature", "importance", "abs_importance"}

    def test_tree_pipeline_sorted_descending(
        self, fitted_tree_pipeline: Pipeline
    ) -> None:
        feature_names = ["f0", "f1", "f2"]
        result = extract_feature_importance(fitted_tree_pipeline, feature_names)
        diffs = result["abs_importance"].diff().dropna()
        assert (diffs <= 0).all()

    def test_tree_pipeline_importance_non_negative(
        self, fitted_tree_pipeline: Pipeline
    ) -> None:
        feature_names = ["f0", "f1", "f2"]
        result = extract_feature_importance(fitted_tree_pipeline, feature_names)
        assert (result["importance"] >= 0).all()

    def test_linear_pipeline_returns_unified_schema(
        self, fitted_pipeline: Pipeline
    ) -> None:
        feature_names = ["f0", "f1", "f2"]
        result = extract_feature_importance(fitted_pipeline, feature_names)
        assert set(result.columns) == {"feature", "importance", "abs_importance"}

    def test_linear_pipeline_sorted_descending(
        self, fitted_pipeline: Pipeline
    ) -> None:
        feature_names = ["f0", "f1", "f2"]
        result = extract_feature_importance(fitted_pipeline, feature_names)
        diffs = result["abs_importance"].diff().dropna()
        assert (diffs <= 0).all()

    def test_unsupported_classifier_raises_type_error(
        self, fitted_dummy_pipeline: Pipeline
    ) -> None:
        feature_names = ["f0", "f1", "f2"]
        with pytest.raises(TypeError, match="DummyClassifier"):
            extract_feature_importance(fitted_dummy_pipeline, feature_names)


class TestPlotTopFeatures:
    def test_creates_file(
        self, fitted_pipeline: Pipeline, tmp_path: Path
    ) -> None:
        feature_names = ["f0", "f1", "f2"]
        importance_df = extract_lr_coefficients(fitted_pipeline, feature_names)
        out = tmp_path / "importance.png"
        plot_top_features(importance_df, top_n=2, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0


class TestModelFeatureNamesAlignment:
    """Regression tests: label alignment when SecomPreprocessor drops columns."""

    @pytest.fixture()
    def pipeline_with_dropped_col(self) -> tuple[Pipeline, list[str]]:
        """Fit a logistic-regression pipeline where one column is dropped.

        ``sensor_const`` is all-zero (CV 0), so it is dropped by the
        CV filter when ``cv_threshold`` is slightly above 0.
        Returns the fitted pipeline and the full list of raw sensor column names.
        """
        rng = np.random.default_rng(0)
        n = 80
        X = pd.DataFrame(
            {
                "sensor_a": rng.normal(0, 1, n),
                "sensor_b": rng.normal(0, 1, n),
                "sensor_c": rng.normal(0, 1, n),
                "sensor_const": np.zeros(n),  # CV = 0 — will be dropped
            }
        )
        y = pd.Series([0] * 70 + [1] * 10)
        pipeline = build_pipeline(
            "logistic_regression",
            random_seed=42,
            missing_threshold=1.0,
            cv_threshold=1e-9,
            correlation_threshold=1.0,
        )
        pipeline.fit(X, y)
        full_sensor_cols = list(X.columns)
        return pipeline, full_sensor_cols

    def test_model_feature_names_length_matches_classifier(
        self, pipeline_with_dropped_col: tuple[Pipeline, list[str]]
    ) -> None:
        """model_feature_names length must equal classifier.n_features_in_."""
        pipeline, _ = pipeline_with_dropped_col
        kept = model_feature_names(pipeline)
        clf = pipeline.named_steps["classifier"]
        assert len(kept) == clf.n_features_in_

    def test_dropped_column_not_in_model_feature_names(
        self, pipeline_with_dropped_col: tuple[Pipeline, list[str]]
    ) -> None:
        """The dropped low-CV column must not appear in model_feature_names."""
        pipeline, _ = pipeline_with_dropped_col
        kept = model_feature_names(pipeline)
        assert "sensor_const" not in kept

    def test_extract_feature_importance_with_kept_names_returns_correct_rows(
        self, pipeline_with_dropped_col: tuple[Pipeline, list[str]]
    ) -> None:
        """extract_feature_importance with kept names yields one row per feature."""
        pipeline, _ = pipeline_with_dropped_col
        kept = model_feature_names(pipeline)
        result = extract_feature_importance(pipeline, kept)
        assert len(result) == len(kept)
        assert set(result["feature"]) == set(kept)

    def test_extract_feature_importance_with_full_cols_raises(
        self, pipeline_with_dropped_col: tuple[Pipeline, list[str]]
    ) -> None:
        """Passing full raw column list causes a length-mismatch error (the bug)."""
        pipeline, full_sensor_cols = pipeline_with_dropped_col
        with pytest.raises((ValueError, IndexError)):
            extract_feature_importance(pipeline, full_sensor_cols)
