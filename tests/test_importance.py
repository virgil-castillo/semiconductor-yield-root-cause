"""Tests for feature importance extraction and visualization."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yield_risk.importance import extract_lr_coefficients, plot_top_features


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
