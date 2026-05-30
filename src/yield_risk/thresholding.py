"""Cost-sensitive decision threshold optimisation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from yield_risk.config import CostMatrix, ThresholdSearchConfig


@dataclass
class ThresholdResult:
    """Result of cost-sensitive threshold optimisation.

    Attributes:
        threshold: Decision threshold that minimises expected cost.
        expected_cost: Total expected cost at the optimal threshold.
    """

    threshold: float
    expected_cost: float


def expected_cost_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_matrix: CostMatrix,
) -> float:
    """Compute total expected cost for binary predictions at a given threshold.

    Converts *y_prob* to binary predictions using *threshold*, computes the
    confusion matrix, and sums per-cell costs from *cost_matrix*.

    Args:
        y_true: Ground-truth binary labels (0=pass, 1=fail), shape (n,).
        y_prob: Predicted probabilities for the positive class, shape (n,).
        threshold: Decision boundary. Samples with ``y_prob >= threshold``
            are predicted positive (fail).
        cost_matrix: Per-outcome costs (true_pass, true_fail, false_fail,
            false_pass).

    Returns:
        Total cost as a float (sum of count × unit_cost for each cell).
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(
        int(tn) * cost_matrix.true_pass
        + int(fp) * cost_matrix.false_fail
        + int(fn) * cost_matrix.false_pass
        + int(tp) * cost_matrix.true_fail
    )


def threshold_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_matrix: CostMatrix,
    search: ThresholdSearchConfig,
) -> pd.DataFrame:
    """Compute expected cost at each candidate threshold in the search grid.

    Evaluates ``search.steps`` evenly-spaced thresholds from ``search.low``
    to ``search.high`` (inclusive).

    Args:
        y_true: Ground-truth binary labels (0=pass, 1=fail), shape (n,).
        y_prob: Predicted probabilities for the positive class, shape (n,).
        cost_matrix: Per-outcome cost assignments.
        search: Grid parameters — low, high, and number of steps.

    Returns:
        DataFrame with columns ``threshold`` (float) and ``expected_cost``
        (float), one row per candidate threshold, ordered from low to high.
    """
    thresholds = np.linspace(search.low, search.high, search.steps)
    costs = [
        expected_cost_at_threshold(y_true, y_prob, float(t), cost_matrix)
        for t in thresholds
    ]
    return pd.DataFrame({"threshold": thresholds.tolist(), "expected_cost": costs})


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_matrix: CostMatrix,
    search: ThresholdSearchConfig,
) -> ThresholdResult:
    """Find the decision threshold that minimises expected cost.

    Evaluates all candidate thresholds via :func:`threshold_cost_curve` and
    returns the one with the lowest expected cost. Ties are broken by choosing
    the first (lowest) threshold in the search grid.

    Args:
        y_true: Ground-truth binary labels (0=pass, 1=fail), shape (n,).
        y_prob: Predicted probabilities for the positive class, shape (n,).
        cost_matrix: Per-outcome cost assignments.
        search: Grid parameters — low, high, and number of steps.

    Returns:
        ThresholdResult containing the optimal threshold and its expected cost.
    """
    curve = threshold_cost_curve(y_true, y_prob, cost_matrix, search)
    best_row = curve.loc[curve["expected_cost"].idxmin()]
    return ThresholdResult(
        threshold=float(best_row["threshold"]),
        expected_cost=float(best_row["expected_cost"]),
    )
