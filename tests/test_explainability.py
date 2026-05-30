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
