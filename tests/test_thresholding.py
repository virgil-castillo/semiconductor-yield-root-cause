"""Tests for cost-sensitive threshold optimizer."""
from __future__ import annotations

import numpy as np
import pytest

from yield_risk.config import CostMatrix, ThresholdSearchConfig
from yield_risk.thresholding import (
    ThresholdResult,
    expected_cost_at_threshold,
    find_optimal_threshold,
    threshold_cost_curve,
)

_COST = CostMatrix(true_pass=0.0, true_fail=0.0, false_fail=1.0, false_pass=10.0)
_SEARCH = ThresholdSearchConfig(low=0.01, high=0.99, steps=99)

# Perfect classifier: true labels match high/low probs exactly
_Y_TRUE = np.array([0, 0, 0, 1, 1], dtype=int)
_Y_PROB_PERFECT = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
_Y_PROB_ZEROS = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # all predicted pass


class TestExpectedCostAtThreshold:
    def test_perfect_classifier_zero_cost(self) -> None:
        cost = expected_cost_at_threshold(_Y_TRUE, _Y_PROB_PERFECT, 0.5, _COST)
        assert cost == pytest.approx(0.0)

    def test_all_predicted_pass_costs_fn_only(self) -> None:
        # y_prob_zeros < 0.5, so all predicted pass
        # 2 true positives become false negatives: 2 * 10.0 = 20.0
        cost = expected_cost_at_threshold(_Y_TRUE, _Y_PROB_ZEROS, 0.5, _COST)
        assert cost == pytest.approx(20.0)

    def test_cost_increases_with_fn(self) -> None:
        # lower threshold → more FP not fewer FN at low probs
        c_low = expected_cost_at_threshold(_Y_TRUE, _Y_PROB_ZEROS, 0.05, _COST)
        c_high = expected_cost_at_threshold(_Y_TRUE, _Y_PROB_ZEROS, 0.5, _COST)
        # at threshold=0.05, all predicted fail: FP=3 (3*1), TP=2 (0); cost=3
        # at threshold=0.5, all predicted pass: FN=2 (2*10); cost=20
        assert c_low < c_high

    def test_returns_float(self) -> None:
        cost = expected_cost_at_threshold(_Y_TRUE, _Y_PROB_PERFECT, 0.5, _COST)
        assert isinstance(cost, float)


class TestThresholdCostCurve:
    def test_returns_dataframe_with_expected_columns(self) -> None:
        df = threshold_cost_curve(_Y_TRUE, _Y_PROB_PERFECT, _COST, _SEARCH)
        assert "threshold" in df.columns
        assert "expected_cost" in df.columns

    def test_row_count_matches_steps(self) -> None:
        df = threshold_cost_curve(_Y_TRUE, _Y_PROB_PERFECT, _COST, _SEARCH)
        assert len(df) == _SEARCH.steps

    def test_thresholds_in_range(self) -> None:
        df = threshold_cost_curve(_Y_TRUE, _Y_PROB_PERFECT, _COST, _SEARCH)
        assert df["threshold"].min() >= _SEARCH.low
        assert df["threshold"].max() <= _SEARCH.high

    def test_costs_are_non_negative(self) -> None:
        df = threshold_cost_curve(_Y_TRUE, _Y_PROB_PERFECT, _COST, _SEARCH)
        assert (df["expected_cost"] >= 0).all()


class TestFindOptimalThreshold:
    def test_returns_threshold_result(self) -> None:
        result = find_optimal_threshold(_Y_TRUE, _Y_PROB_PERFECT, _COST, _SEARCH)
        assert isinstance(result, ThresholdResult)

    def test_optimal_threshold_in_range(self) -> None:
        result = find_optimal_threshold(_Y_TRUE, _Y_PROB_PERFECT, _COST, _SEARCH)
        assert _SEARCH.low <= result.threshold <= _SEARCH.high

    def test_perfect_classifier_low_cost(self) -> None:
        result = find_optimal_threshold(_Y_TRUE, _Y_PROB_PERFECT, _COST, _SEARCH)
        assert result.expected_cost == pytest.approx(0.0)

    def test_high_fn_cost_prefers_low_threshold(self) -> None:
        # With false_pass=100 >> false_fail=1, optimizer must catch all failures
        # → optimal threshold should be below 0.5 to flag borderline cases
        high_fn_cost = CostMatrix(
            true_pass=0.0, true_fail=0.0, false_fail=1.0, false_pass=100.0
        )
        ambiguous_prob = np.array([0.1, 0.1, 0.1, 0.4, 0.9])
        result_high = find_optimal_threshold(
            _Y_TRUE, ambiguous_prob, high_fn_cost, _SEARCH
        )
        result_base = find_optimal_threshold(_Y_TRUE, ambiguous_prob, _COST, _SEARCH)
        assert result_high.threshold <= result_base.threshold
