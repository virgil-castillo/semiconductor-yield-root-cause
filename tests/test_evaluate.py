"""Tests for classification metrics, reports, and plot functions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from yield_risk.evaluate import (
    ClassificationMetrics,
    compute_metrics,
    format_report,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)

_Y_TRUE = np.array([0, 0, 0, 1, 1])
_Y_PROB_PERFECT = np.array([0.1, 0.2, 0.1, 0.9, 0.8])


class TestComputeMetrics:
    def test_perfect_predictions_roc_auc_is_one(self) -> None:
        metrics = compute_metrics(_Y_TRUE, _Y_PROB_PERFECT)
        assert metrics.roc_auc == pytest.approx(1.0)

    def test_perfect_predictions_f1_is_one(self) -> None:
        metrics = compute_metrics(_Y_TRUE, _Y_PROB_PERFECT)
        assert metrics.f1 == pytest.approx(1.0)

    def test_confusion_matrix_is_2x2(self) -> None:
        metrics = compute_metrics(_Y_TRUE, _Y_PROB_PERFECT)
        cm = metrics.confusion_matrix
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2

    def test_all_fields_populated(self) -> None:
        metrics = compute_metrics(_Y_TRUE, _Y_PROB_PERFECT)
        assert metrics.roc_auc is not None
        assert metrics.pr_auc is not None
        assert metrics.precision is not None
        assert metrics.recall is not None
        assert metrics.f1 is not None
        assert metrics.confusion_matrix is not None

    def test_threshold_changes_predictions(self) -> None:
        y_true = np.array([0, 1])
        y_prob = np.array([0.4, 0.6])
        m1 = compute_metrics(y_true, y_prob, threshold=0.5)
        m2 = compute_metrics(y_true, y_prob, threshold=0.7)
        assert m1.confusion_matrix != m2.confusion_matrix


class TestFormatReport:
    def test_contains_all_metric_names(self) -> None:
        metrics = ClassificationMetrics(
            roc_auc=0.9,
            pr_auc=0.8,
            precision=0.7,
            recall=0.6,
            f1=0.65,
            confusion_matrix=[[10, 1], [2, 5]],
        )
        report = format_report(metrics)
        assert "ROC AUC" in report
        assert "PR AUC" in report
        assert "Precision" in report
        assert "Recall" in report
        assert "F1" in report

    def test_returns_string(self) -> None:
        metrics = ClassificationMetrics(
            roc_auc=0.9,
            pr_auc=0.8,
            precision=0.7,
            recall=0.6,
            f1=0.65,
            confusion_matrix=[[10, 1], [2, 5]],
        )
        assert isinstance(format_report(metrics), str)


class TestPlotFunctions:
    def test_plot_confusion_matrix_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "cm.png"
        plot_confusion_matrix([[3, 0], [0, 2]], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_roc_curve_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "roc.png"
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        plot_roc_curve(y_true, y_prob, 1.0, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_precision_recall_curve_creates_file(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "pr.png"
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        plot_precision_recall_curve(y_true, y_prob, 1.0, out)
        assert out.exists()
        assert out.stat().st_size > 0
