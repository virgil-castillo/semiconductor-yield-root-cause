"""Tests for the GRU sequence model core (model/dataset/preprocessing/loss)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tests.conftest import make_synthetic
from yield_risk.sequence_models import (
    RawSensorCleaner,
    SequenceSensorPipeline,
    build_gru,
    build_timestep_weights,
    compute_pos_weight,
    predict_logits,
)

RAW_COLS = [f"sensor_{i}" for i in range(8)]
MISSING_THRESH = 0.5
VAR_THRESH = 1e-6
CORR_THRESH = 0.95


class TestForwardShapes:
    """Forward-pass shape and guard behavior."""

    def test_standard_mode_shape(self) -> None:
        model = build_gru(n_sensors=6, emb_dim=4, hidden_size=8)
        x = torch.randn(4, 6)
        ids = torch.randint(0, 6, (4, 6))
        out = model(x, ids)
        assert out.shape == (4,)
        assert out.dtype == torch.float32

    def test_early_mode_shape(self) -> None:
        model = build_gru(n_sensors=6, emb_dim=4, hidden_size=8, early_prediction=True)
        x = torch.randn(4, 6)
        ids = torch.randint(0, 6, (4, 6))
        out = model(x, ids)
        assert out.shape == (4, 6)

    def test_shorter_window(self) -> None:
        model = build_gru(n_sensors=6, emb_dim=4, hidden_size=8)
        x = torch.randn(4, 3)
        ids = torch.randint(0, 6, (4, 3))
        assert model(x, ids).shape == (4,)

    def test_predict_logits_early_final_timestep(self) -> None:
        model = build_gru(n_sensors=6, emb_dim=4, hidden_size=8, early_prediction=True)
        x = torch.randn(4, 5)
        ids = torch.randint(0, 6, (4, 5))
        assert predict_logits(model, x, ids).shape == (4,)

    def test_wrong_ndim_x(self) -> None:
        model = build_gru(n_sensors=6)
        with pytest.raises(ValueError, match="2-D"):
            model(torch.randn(4), torch.randint(0, 6, (4, 6)))

    def test_wrong_ndim_sensor_ids(self) -> None:
        model = build_gru(n_sensors=6)
        with pytest.raises(ValueError, match="sensor_ids as a 2-D"):
            model(torch.randn(4, 6), torch.randint(0, 6, (4,)))

    def test_mismatched_shapes(self) -> None:
        model = build_gru(n_sensors=6)
        with pytest.raises(ValueError, match="same shape"):
            model(torch.randn(4, 6), torch.randint(0, 6, (4, 5)))

    def test_empty_window(self) -> None:
        model = build_gru(n_sensors=6)
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            model(torch.randn(4, 0), torch.randint(0, 6, (4, 0)))

    def test_out_of_range_sensor_id(self) -> None:
        model = build_gru(n_sensors=6)
        ids = torch.full((4, 6), 99, dtype=torch.int64)
        with pytest.raises(ValueError, match=r"outside \[0, n_sensors\)"):
            model(torch.randn(4, 6), ids)


class TestRawSensorCleaner:
    """Raw cleaning strict-threshold semantics and leakage-freedom."""

    def _fit(self) -> RawSensorCleaner:
        x, _ = make_synthetic()
        return RawSensorCleaner.fit(
            x, RAW_COLS, MISSING_THRESH, VAR_THRESH, CORR_THRESH
        )

    def test_drops_match_strict_semantics(self) -> None:
        cleaner = self._fit()
        assert "sensor_0" in cleaner.dropped_high_missing
        assert "sensor_1" in cleaner.dropped_low_variance
        assert "sensor_3" in cleaner.dropped_high_correlation
        assert "sensor_2" in cleaner.sensor_cols
        assert "sensor_3" not in cleaner.sensor_cols

    def test_full_transform_finite_and_ordered(self) -> None:
        cleaner = self._fit()
        x, _ = make_synthetic()
        out = cleaner.transform_full(x, RAW_COLS)
        assert out.shape == (40, len(cleaner.sensor_cols))
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))

    def test_window_transform_ignores_dropped(self) -> None:
        cleaner = self._fit()
        x, _ = make_synthetic()
        window_cols = ["sensor_1", "sensor_2", "sensor_4"]
        x_win = x[:, [1, 2, 4]]
        out, retained = cleaner.transform_window(x_win, window_cols)
        assert retained == ["sensor_2", "sensor_4"]
        assert out.shape == (40, 2)
        assert np.all(np.isfinite(out))

    def test_unknown_window_sensor(self) -> None:
        cleaner = self._fit()
        with pytest.raises(ValueError, match="Unknown window sensor column: sensor_99"):
            cleaner.transform_window(np.zeros((2, 1), dtype=np.float32), ["sensor_99"])

    def test_window_width_mismatch(self) -> None:
        cleaner = self._fit()
        with pytest.raises(ValueError, match="Expected 2 window columns, got 1"):
            cleaner.transform_window(
                np.zeros((2, 1), dtype=np.float32), ["sensor_2", "sensor_4"]
            )

    def test_all_dropped_window(self) -> None:
        cleaner = self._fit()
        with pytest.raises(ValueError, match="no retained sensor columns"):
            cleaner.transform_window(
                np.zeros((2, 2), dtype=np.float32), ["sensor_0", "sensor_1"]
            )

    def test_infinite_value_in_window(self) -> None:
        cleaner = self._fit()
        x = np.full((2, 1), np.inf, dtype=np.float32)
        with pytest.raises(ValueError, match="infinite values"):
            cleaner.transform_window(x, ["sensor_2"])

    def test_empty_window(self) -> None:
        cleaner = self._fit()
        with pytest.raises(ValueError, match="Window sensor columns must be non-empty"):
            cleaner.transform_window(np.zeros((2, 0), dtype=np.float32), [])

    def test_infinite_value_on_fit(self) -> None:
        x, _ = make_synthetic()
        x[0, 5] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            RawSensorCleaner.fit(x, RAW_COLS, MISSING_THRESH, VAR_THRESH, CORR_THRESH)

    def test_no_leakage(self) -> None:
        x, _ = make_synthetic()
        train = x[:20].copy()
        val = x[20:].copy()
        cleaner = RawSensorCleaner.fit(
            train, RAW_COLS, MISSING_THRESH, VAR_THRESH, CORR_THRESH
        )
        medians_before = cleaner.medians.copy()
        drops_before = list(cleaner.dropped_high_missing)
        # Mutate val and transform it — fitted stats must not change.
        val[:, 4] = 999.0
        cleaner.transform_full(val, RAW_COLS)
        assert np.array_equal(cleaner.medians, medians_before)
        assert cleaner.dropped_high_missing == drops_before


class TestSequenceSensorPipeline:
    """Scaler composition, ID mapping, round-trip, and leakage-freedom."""

    def _fit(self) -> SequenceSensorPipeline:
        x, _ = make_synthetic()
        return SequenceSensorPipeline.fit(
            x, RAW_COLS, MISSING_THRESH, VAR_THRESH, CORR_THRESH
        )

    def test_full_transform_normalized_and_ids(self) -> None:
        pipe = self._fit()
        x, _ = make_synthetic()
        x_norm, ids = pipe.transform_full(x, RAW_COLS)
        n_sensors = len(pipe.sensor_cols)
        assert x_norm.shape == (40, n_sensors)
        assert ids.shape == (40, n_sensors)
        assert ids.dtype == np.int64
        assert np.array_equal(ids[0], np.arange(n_sensors))
        # Standardized columns are approximately zero-mean.
        assert np.allclose(x_norm.mean(axis=0), 0.0, atol=1e-4)

    def test_window_transform_ids(self) -> None:
        pipe = self._fit()
        x, _ = make_synthetic()
        window_cols = ["sensor_2", "sensor_4"]
        x_win = x[:, [2, 4]]
        x_norm, ids = pipe.transform_window(x_win, window_cols)
        expected = [pipe.sensor_cols.index(c) for c in window_cols]
        assert np.array_equal(ids[0], np.array(expected))
        assert x_norm.shape == (40, 2)

    def test_to_from_dict_roundtrip(self) -> None:
        pipe = self._fit()
        d = pipe.to_dict()
        restored = SequenceSensorPipeline.from_dict(d)
        assert restored.raw_sensor_cols == pipe.raw_sensor_cols
        assert restored.sensor_cols == pipe.sensor_cols
        assert np.allclose(restored.mean, pipe.mean)
        assert np.allclose(restored.scale, pipe.scale)
        assert np.allclose(restored.cleaner.medians, pipe.cleaner.medians)
        assert restored.cleaner.dropped_high_correlation == (
            pipe.cleaner.dropped_high_correlation
        )
        # to_dict uses plain lists.
        assert isinstance(d["mean"], list)
        assert isinstance(d["medians"], list)

    def test_scale_zero_guard(self) -> None:
        pipe = self._fit()
        assert np.all(pipe.scale != 0.0)

    def test_no_leakage(self) -> None:
        x, _ = make_synthetic()
        train = x[:20].copy()
        val = x[20:].copy()
        pipe = SequenceSensorPipeline.fit(
            train, RAW_COLS, MISSING_THRESH, VAR_THRESH, CORR_THRESH
        )
        mean_before = pipe.mean.copy()
        scale_before = pipe.scale.copy()
        medians_before = pipe.cleaner.medians.copy()
        val[:, 5] = -500.0
        pipe.transform_full(val, RAW_COLS)
        assert np.array_equal(pipe.mean, mean_before)
        assert np.array_equal(pipe.scale, scale_before)
        assert np.array_equal(pipe.cleaner.medians, medians_before)


class TestPosWeight:
    """pos_weight computation and degenerate-label raises."""

    def test_ratio(self) -> None:
        y = np.array([0] * 32 + [1] * 8, dtype=np.int64)
        assert compute_pos_weight(y) == 4.0

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="y_train is empty"):
            compute_pos_weight(np.array([], dtype=np.int64))

    def test_no_positive(self) -> None:
        with pytest.raises(ValueError, match="no positive samples"):
            compute_pos_weight(np.zeros(10, dtype=np.int64))

    def test_no_negative(self) -> None:
        with pytest.raises(ValueError, match="no negative samples"):
            compute_pos_weight(np.ones(10, dtype=np.int64))


class TestTimestepWeights:
    """Timestep-weighting schemes sum to 1 with correct shape."""

    def test_none_uniform(self) -> None:
        w = build_timestep_weights(5, "none", "cpu")
        assert torch.allclose(w, torch.full((5,), 0.2))
        assert math.isclose(float(w.sum()), 1.0, abs_tol=1e-6)

    def test_linear_increasing(self) -> None:
        w = build_timestep_weights(6, "linear", "cpu")
        assert float(w[-1]) > float(w[0])
        assert torch.all(w[1:] > w[:-1])
        assert math.isclose(float(w.sum()), 1.0, abs_tol=1e-6)

    def test_sqrt_increasing(self) -> None:
        w = build_timestep_weights(6, "sqrt", "cpu")
        assert torch.all(w[1:] > w[:-1])
        assert math.isclose(float(w.sum()), 1.0, abs_tol=1e-6)

    def test_unknown_scheme(self) -> None:
        with pytest.raises(ValueError, match="Unknown timestep_weighting scheme"):
            build_timestep_weights(4, "bogus", "cpu")

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            build_timestep_weights(0, "none", "cpu")


class TestEarlyPredictionLoss:
    """Early-prediction loss target broadcast and weighting behavior."""

    def _setup(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        b, w = 4, 5
        logits = torch.randn(b, w)
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        targets = y.unsqueeze(1).expand(b, w)
        pos_weight = torch.tensor([2.0])
        elem = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        return elem, y, pos_weight

    def _loss(self, elem: torch.Tensor, scheme: str) -> float:
        w = build_timestep_weights(elem.shape[1], scheme, "cpu")
        per_sample = (elem * w.unsqueeze(0)).sum(dim=1)
        return float(per_sample.mean())

    def test_target_broadcast_shape(self) -> None:
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        targets = y.unsqueeze(1).expand(4, 5)
        assert targets.shape == (4, 5)
        assert torch.all(targets[:, 0] == y)

    def test_scheme_changes_loss(self) -> None:
        elem, _, _ = self._setup()
        assert not math.isclose(
            self._loss(elem, "none"), self._loss(elem, "linear"), abs_tol=1e-6
        )

    def test_none_equals_mean_reduction(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(4, 5)
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        targets = y.unsqueeze(1).expand(4, 5)
        pos_weight = torch.tensor([2.0])
        elem = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        mean_red = float(
            F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction="mean"
            )
        )
        assert math.isclose(self._loss(elem, "none"), mean_red, abs_tol=1e-6)
