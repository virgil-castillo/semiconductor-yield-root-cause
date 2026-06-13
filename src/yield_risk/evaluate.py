"""Classification metrics, formatted reports, and diagnostic plots."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from yield_risk.config import CostMatrix
from yield_risk.thresholding import expected_cost_at_threshold


@dataclass
class ClassificationMetrics:
    """Container for binary classification evaluation metrics.

    Attributes:
        roc_auc: Area under the ROC curve.
        pr_auc: Area under the precision-recall curve (average precision).
        precision: Precision at the classification threshold.
        recall: Recall at the classification threshold.
        f1: F1 score at the classification threshold.
        confusion_matrix: 2x2 matrix as [[TN, FP], [FN, TP]].
    """

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute classification metrics from true labels and predicted probabilities.

    Args:
        y_true: Ground truth binary labels (0 or 1), shape (n_samples,).
        y_prob: Predicted probabilities for the positive class, shape (n_samples,).
        threshold: Decision threshold for converting probabilities to binary.

    Returns:
        Populated ClassificationMetrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_matrix: CostMatrix,
) -> tuple[ClassificationMetrics, float]:
    """Apply a frozen threshold and return metrics plus expected cost.

    Computes ``ClassificationMetrics`` via :func:`compute_metrics` and expected
    cost via :func:`~yield_risk.thresholding.expected_cost_at_threshold`, both
    at *threshold*.  This helper never calls ``find_optimal_threshold`` — it
    only applies an already-decided threshold to test data.

    Args:
        y_true: Ground-truth binary labels (0 or 1), shape (n_samples,).
        y_prob: Predicted probabilities for the positive class, shape
            (n_samples,).
        threshold: Decision threshold to apply (frozen, derived from training
            data).
        cost_matrix: Per-outcome costs used to compute expected cost.

    Returns:
        Tuple of ``(ClassificationMetrics, expected_cost)`` where
        ``expected_cost`` is a non-negative float.
    """
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    cost = expected_cost_at_threshold(y_true, y_prob, threshold, cost_matrix)
    return metrics, cost


def format_report(metrics: ClassificationMetrics, model_name: str = "Model") -> str:
    """Format metrics as a human-readable multi-line text report.

    Args:
        metrics: Computed classification metrics.
        model_name: Name shown in the report title.

    Returns:
        Formatted string with all metric values and confusion matrix.
    """
    cm = metrics.confusion_matrix
    return (
        f"=== {model_name} Evaluation ===\n"
        f"ROC AUC:    {metrics.roc_auc:.3f}\n"
        f"PR AUC:     {metrics.pr_auc:.3f}\n"
        f"Precision:  {metrics.precision:.3f}\n"
        f"Recall:     {metrics.recall:.3f}\n"
        f"F1 Score:   {metrics.f1:.3f}\n"
        "\n"
        "Confusion Matrix:\n"
        "              Pred Pass  Pred Fail\n"
        f"Actual Pass  {cm[0][0]:9d}  {cm[0][1]:9d}\n"
        f"Actual Fail  {cm[1][0]:9d}  {cm[1][1]:9d}"
    )


def save_metrics(metrics: ClassificationMetrics, output_path: Path) -> None:
    """Serialize metrics to a JSON file.

    Uses dataclasses.asdict for conversion. confusion_matrix is stored
    as a nested list of ints.

    Args:
        metrics: Computed classification metrics.
        output_path: Path to write the JSON file.
    """
    output_path.write_text(json.dumps(asdict(metrics), indent=2))


def plot_confusion_matrix(
    cm: list[list[int]],
    output_path: Path,
) -> None:
    """Save confusion matrix heatmap as PNG.

    Labels: x-axis "Predicted" (Pass/Fail), y-axis "Actual" (Pass/Fail).
    Annotate each cell with the count value.

    Args:
        cm: 2x2 confusion matrix [[TN, FP], [FN, TP]].
        output_path: File path for the saved PNG.
    """
    cm_array = np.array(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm_array, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pass", "Fail"])
    ax.set_yticklabels(["Pass", "Fail"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=12)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    output_path: Path,
) -> None:
    """Save ROC curve plot as PNG.

    Includes diagonal reference line. Legend shows AUC value.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities for positive class.
        roc_auc: Pre-computed AUC (displayed in legend).
        output_path: File path for the saved PNG.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    _, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    output_path: Path,
) -> None:
    """Save precision-recall curve plot as PNG.

    Legend shows AP value.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities for positive class.
        pr_auc: Pre-computed average precision (displayed in legend).
        output_path: File path for the saved PNG.
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    _, ax = plt.subplots()
    ax.plot(recall_vals, precision_vals, label=f"AP = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
