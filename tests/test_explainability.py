"""Tests for SHAP explainability module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yield_risk.explainability import (
    ShapExplanations,
    compute_shap_values,
    global_feature_importance,
    load_shap_values,
    local_explanation,
    save_shap_values,
)
from yield_risk.model import model_feature_names
from yield_risk.preprocess import SecomPreprocessor


@pytest.fixture()
def tiny_pipeline() -> Pipeline:
    """3-feature LogisticRegression pipeline fitted on 40 samples."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(40, 3)),
        columns=["sensor_000", "sensor_001", "sensor_002"],
    )
    y = pd.Series([0] * 34 + [1] * 6)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=200, random_state=0, class_weight="balanced"
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


@pytest.fixture()
def background_data() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        rng.normal(size=(20, 3)),
        columns=["sensor_000", "sensor_001", "sensor_002"],
    )


@pytest.fixture()
def explain_data() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        rng.normal(size=(10, 3)),
        columns=["sensor_000", "sensor_001", "sensor_002"],
    )


@pytest.fixture()
def tree_pipeline() -> Pipeline:
    """3-feature RandomForestClassifier pipeline fitted on 50 samples."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(3)
    X = pd.DataFrame(
        rng.normal(size=(50, 3)),
        columns=["sensor_000", "sensor_001", "sensor_002"],
    )
    y = pd.Series([0] * 42 + [1] * 8)
    pipe = Pipeline(
        [
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=10, random_state=0, class_weight="balanced"
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


@pytest.fixture()
def explanations(
    tiny_pipeline: Pipeline,
    background_data: pd.DataFrame,
    explain_data: pd.DataFrame,
) -> ShapExplanations:
    return compute_shap_values(tiny_pipeline, background_data, explain_data)


class TestComputeShapValues:
    def test_returns_shap_explanations(
        self,
        tiny_pipeline: Pipeline,
        background_data: pd.DataFrame,
        explain_data: pd.DataFrame,
    ) -> None:
        result = compute_shap_values(tiny_pipeline, background_data, explain_data)
        assert isinstance(result, ShapExplanations)

    def test_shap_values_shape(self, explanations: ShapExplanations) -> None:
        # 10 samples, 3 features
        assert explanations.shap_values.shape == (10, 3)

    def test_feature_names_match_columns(
        self,
        explanations: ShapExplanations,
        explain_data: pd.DataFrame,
    ) -> None:
        assert explanations.feature_names == list(explain_data.columns)

    def test_base_value_is_float(self, explanations: ShapExplanations) -> None:
        assert isinstance(explanations.base_value, float)

    def test_unsupported_classifier_raises(
        self,
        background_data: pd.DataFrame,
        explain_data: pd.DataFrame,
    ) -> None:
        from sklearn.neighbors import KNeighborsClassifier

        pipe = Pipeline([("clf", KNeighborsClassifier())])
        X_vals = background_data.to_numpy()
        y = np.array([0] * 17 + [1] * 3)
        pipe.fit(X_vals, y)
        # Re-wrap with DataFrame columns
        bg = background_data
        ex = explain_data
        with pytest.raises(NotImplementedError):
            compute_shap_values(pipe, bg, ex)


class TestGlobalFeatureImportance:
    def test_returns_dataframe(self, explanations: ShapExplanations) -> None:
        result = global_feature_importance(explanations)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, explanations: ShapExplanations) -> None:
        result = global_feature_importance(explanations)
        assert "feature" in result.columns
        assert "mean_abs_shap" in result.columns

    def test_sorted_descending(self, explanations: ShapExplanations) -> None:
        result = global_feature_importance(explanations)
        scores = result["mean_abs_shap"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_row_count_matches_features(self, explanations: ShapExplanations) -> None:
        result = global_feature_importance(explanations)
        assert len(result) == len(explanations.feature_names)

    def test_values_non_negative(self, explanations: ShapExplanations) -> None:
        result = global_feature_importance(explanations)
        assert (result["mean_abs_shap"] >= 0).all()


class TestLocalExplanation:
    def test_returns_dataframe(self, explanations: ShapExplanations) -> None:
        result = local_explanation(explanations, row_idx=0)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, explanations: ShapExplanations) -> None:
        result = local_explanation(explanations, row_idx=0)
        assert "feature" in result.columns
        assert "shap_value" in result.columns

    def test_row_count_matches_features(self, explanations: ShapExplanations) -> None:
        result = local_explanation(explanations, row_idx=0)
        assert len(result) == len(explanations.feature_names)

    def test_sorted_by_abs_shap_descending(
        self, explanations: ShapExplanations
    ) -> None:
        result = local_explanation(explanations, row_idx=0)
        abs_vals = result["shap_value"].abs().tolist()
        assert abs_vals == sorted(abs_vals, reverse=True)


class TestSaveLoadShapValues:
    def test_roundtrip_preserves_shap_values(
        self, tmp_path: Path, explanations: ShapExplanations
    ) -> None:
        out = tmp_path / "shap.npz"
        save_shap_values(explanations, out)
        loaded = load_shap_values(out)
        np.testing.assert_array_almost_equal(
            loaded.shap_values, explanations.shap_values
        )

    def test_roundtrip_preserves_feature_names(
        self, tmp_path: Path, explanations: ShapExplanations
    ) -> None:
        out = tmp_path / "shap.npz"
        save_shap_values(explanations, out)
        loaded = load_shap_values(out)
        assert loaded.feature_names == explanations.feature_names

    def test_roundtrip_preserves_base_value(
        self, tmp_path: Path, explanations: ShapExplanations
    ) -> None:
        out = tmp_path / "shap.npz"
        save_shap_values(explanations, out)
        loaded = load_shap_values(out)
        assert loaded.base_value == pytest.approx(explanations.base_value)

    def test_file_created(
        self, tmp_path: Path, explanations: ShapExplanations
    ) -> None:
        out = tmp_path / "shap.npz"
        save_shap_values(explanations, out)
        assert out.exists()


class TestTreeExplainerDispatch:
    def test_tree_pipeline_returns_shap_explanations(
        self,
        tree_pipeline: Pipeline,
        background_data: pd.DataFrame,
        explain_data: pd.DataFrame,
    ) -> None:
        result = compute_shap_values(tree_pipeline, background_data, explain_data)
        assert isinstance(result, ShapExplanations)

    def test_tree_shap_values_shape(
        self,
        tree_pipeline: Pipeline,
        background_data: pd.DataFrame,
        explain_data: pd.DataFrame,
    ) -> None:
        result = compute_shap_values(tree_pipeline, background_data, explain_data)
        # Must be 2D: (n_explain_samples, n_features)
        assert result.shap_values.ndim == 2
        assert result.shap_values.shape == (10, 3)

    def test_tree_global_importance_works(
        self,
        tree_pipeline: Pipeline,
        background_data: pd.DataFrame,
        explain_data: pd.DataFrame,
    ) -> None:
        result = compute_shap_values(tree_pipeline, background_data, explain_data)
        importance = global_feature_importance(result)
        assert "mean_abs_shap" in importance.columns
        assert (importance["mean_abs_shap"] >= 0).all()

    def test_tree_local_explanation_works(
        self,
        tree_pipeline: Pipeline,
        background_data: pd.DataFrame,
        explain_data: pd.DataFrame,
    ) -> None:
        result = compute_shap_values(tree_pipeline, background_data, explain_data)
        local = local_explanation(result, row_idx=0)
        assert "shap_value" in local.columns
        abs_vals = local["shap_value"].abs().tolist()
        assert abs_vals == sorted(abs_vals, reverse=True)


class TestPreprocessPipelineFeatureNames:
    """SHAP feature names use post-selection names when preprocess drops columns."""

    def _build_pipeline_with_dropped_col(
        self,
    ) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
        """Return a fitted pipeline, background df, and explain df.

        sensor_000, sensor_001, sensor_002 are normal random features.
        sensor_constant is all-zeros — CV = 0, dropped by SecomPreprocessor.
        """
        rng = np.random.default_rng(42)
        n_train = 60

        X_train = pd.DataFrame(
            {
                "sensor_000": rng.normal(size=n_train),
                "sensor_001": rng.normal(size=n_train),
                "sensor_002": rng.normal(size=n_train),
                "sensor_constant": np.zeros(n_train),  # CV = 0 — dropped
            }
        )
        y_train = pd.Series([0] * 52 + [1] * 8)

        X_bg = pd.DataFrame(
            {
                "sensor_000": rng.normal(size=20),
                "sensor_001": rng.normal(size=20),
                "sensor_002": rng.normal(size=20),
                "sensor_constant": np.zeros(20),
            }
        )
        X_ex = pd.DataFrame(
            {
                "sensor_000": rng.normal(size=10),
                "sensor_001": rng.normal(size=10),
                "sensor_002": rng.normal(size=10),
                "sensor_constant": np.zeros(10),
            }
        )

        # cv_threshold=0.01: sensor_constant (CV=0) is below this → dropped
        pipeline = Pipeline(
            [
                (
                    "preprocess",
                    SecomPreprocessor(
                        missing_threshold=1.0,
                        cv_threshold=0.01,
                        correlation_threshold=1.0,
                    ),
                ),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=500, random_state=0, class_weight="balanced"
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        return pipeline, X_bg, X_ex

    def test_feature_names_match_model_feature_names(self) -> None:
        pipeline, X_bg, X_ex = self._build_pipeline_with_dropped_col()
        result = compute_shap_values(pipeline, X_bg, X_ex)
        assert result.feature_names == model_feature_names(pipeline)

    def test_feature_names_length_matches_shap_width(self) -> None:
        pipeline, X_bg, X_ex = self._build_pipeline_with_dropped_col()
        result = compute_shap_values(pipeline, X_bg, X_ex)
        assert len(result.feature_names) == result.shap_values.shape[1]

    def test_dropped_column_excluded_from_feature_names(self) -> None:
        pipeline, X_bg, X_ex = self._build_pipeline_with_dropped_col()
        result = compute_shap_values(pipeline, X_bg, X_ex)
        assert "sensor_constant" not in result.feature_names
