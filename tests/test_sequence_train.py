"""Tests for the GRU orchestration module (config/data/train/eval/baseline)."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import make_synthetic
from yield_risk.sequence_models import SequenceSensorPipeline, build_gru
from yield_risk.sequence_train import (
    LoadedCheckpoint,
    PreparedData,
    SequenceMetrics,
    TrainConfig,
    WindowArrays,
    compute_sequence_metrics,
    evaluate,
    load_checkpoint,
    load_sequence_config,
    load_tabular_baseline,
    prepare_data,
    resolve_device,
    save_checkpoint,
    train,
    write_baseline_comparison,
)

RAW_COLS = [f"sensor_{i}" for i in range(8)]
MISSING_THRESH = 0.5
VAR_THRESH = 1e-6
CORR_THRESH = 0.95

# validate_secom enforces the real SECOM schema shape.
SECOM_ROWS = 1567
SECOM_SENSORS = 590


def _fit_pipeline() -> SequenceSensorPipeline:
    x, _ = make_synthetic()
    return SequenceSensorPipeline.fit(
        x, RAW_COLS, MISSING_THRESH, VAR_THRESH, CORR_THRESH
    )


def _prepared_from_synthetic(
    n_rows: int = 80,
    n_sensors: int = 8,
    n_pos: int = 20,
    window_sizes: tuple[int, ...] | None = None,
) -> PreparedData:
    """Build a PreparedData directly from synthetic arrays (no files)."""
    x, y = make_synthetic(n_rows=n_rows, n_sensors=n_sensors, n_pos=n_pos)
    raw_cols = [f"sensor_{i}" for i in range(n_sensors)]
    n_te = n_rows // 4
    x_train, x_test = x[n_te:], x[:n_te]
    y_train, y_test = y[n_te:], y[:n_te]
    pipe = SequenceSensorPipeline.fit(
        x_train, raw_cols, MISSING_THRESH, VAR_THRESH, CORR_THRESH
    )
    sensor_cols = pipe.sensor_cols
    n = len(sensor_cols)
    x_norm, ids = pipe.transform_full(x_train, raw_cols)
    if window_sizes is None:
        train_windows = [
            WindowArrays(x_norm, ids, y_train.astype(np.int64), list(sensor_cols))
        ]
    else:
        train_windows = []
        for k in sorted({min(w, n) for w in window_sizes}):
            train_windows.append(
                WindowArrays(
                    x_norm[:, :k].copy(),
                    ids[:, :k].copy(),
                    y_train.astype(np.int64),
                    list(sensor_cols[:k]),
                )
            )
    n_pos_tr = int((y_train == 1).sum())
    n_neg_tr = int((y_train == 0).sum())
    return PreparedData(
        train_windows=train_windows,
        val_windows=[],
        x_test_raw=x_test,
        y_test=y_test.astype(np.int64),
        raw_sensor_cols=raw_cols,
        sensor_cols=list(sensor_cols),
        preprocessor=pipe,
        pos_weight=float(n_neg_tr) / float(n_pos_tr),
    )


def _write_synthetic_secom(raw_dir: Path, seed: int = 0) -> None:
    """Write full-schema synthetic secom.data / secom_labels.data files."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((SECOM_ROWS, SECOM_SENSORS)).astype(np.float64)

    n_pos = 110
    y = -np.ones(SECOM_ROWS, dtype=int)
    pos_idx = rng.choice(SECOM_ROWS, size=n_pos, replace=False)
    y[pos_idx] = 1

    # Sensor 0: high-missing column (drop).
    x[:, 0] = np.nan
    x[:5, 0] = 1.0
    # Sensor 1: constant (low-variance drop).
    x[:, 1] = 3.0
    # Sensor 3 duplicates sensor 2 (high-correlation drop of later col 3).
    x[:, 2] = rng.standard_normal(SECOM_ROWS)
    x[:, 3] = x[:, 2]
    # Sensor 4: a few ordinary NaNs for imputation.
    x[10, 4] = np.nan
    x[20, 4] = np.nan
    # Sensor 5: signal correlated with the label.
    x[:, 5] = (y == 1).astype(np.float64) * 2.0 + rng.standard_normal(SECOM_ROWS) * 0.5

    lines = []
    for row in x:
        cells = ["NaN" if math.isnan(v) else repr(float(v)) for v in row]
        lines.append(" ".join(cells))
    (raw_dir / "secom.data").write_text("\n".join(lines) + "\n")

    label_lines = [
        f"{int(lbl)} 01/01/2020 00:00:00" for lbl in y
    ]
    (raw_dir / "secom_labels.data").write_text("\n".join(label_lines) + "\n")


class TestLoadConfig:
    """Config loading, overlay, validation, and CLI-merge semantics."""

    def test_missing_file_defaults(self, tmp_path: Path) -> None:
        cfg = load_sequence_config(tmp_path / "nope.yaml")
        assert cfg == TrainConfig()

    def test_overlay_values(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text(
            "emb_dim: 8\nhidden_size: 32\nepochs: 5\n"
            "window_sizes: [2, 4]\ntimestep_weighting: linear\n"
        )
        cfg = load_sequence_config(p)
        assert cfg.emb_dim == 8
        assert cfg.hidden_size == 32
        assert cfg.epochs == 5
        assert cfg.window_sizes == (2, 4)
        assert cfg.timestep_weighting == "linear"
        # Untouched fields keep defaults.
        assert cfg.lr == TrainConfig().lr

    def test_window_sizes_null(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text("window_sizes: null\n")
        assert load_sequence_config(p).window_sizes is None

    def test_unknown_key(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text("bogus_key: 1\n")
        with pytest.raises(ValueError, match="Unknown sequence config keys"):
            load_sequence_config(p)

    def test_bad_timestep_weighting(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text("timestep_weighting: cubic\n")
        with pytest.raises(ValueError, match="timestep_weighting"):
            load_sequence_config(p)

    def test_malformed_window_sizes(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text("window_sizes: [0, -1]\n")
        with pytest.raises(ValueError, match="window_sizes"):
            load_sequence_config(p)

    def test_empty_window_sizes(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text("window_sizes: []\n")
        with pytest.raises(ValueError, match="window_sizes"):
            load_sequence_config(p)

    def test_uncoercible_value(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text("epochs: not_an_int\n")
        with pytest.raises(ValueError, match="epochs"):
            load_sequence_config(p)

    def test_cli_override_merge(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text("emb_dim: 8\nepochs: 5\n")
        cfg = load_sequence_config(p)
        overrides = {k: v for k, v in {"epochs": 10}.items() if v is not None}
        merged = dataclasses.replace(cfg, **overrides)
        assert merged.epochs == 10
        assert merged.emb_dim == 8


class TestPrepareData:
    """Leakage-free preparation over synthetic SECOM-like files."""

    def test_split_shapes_and_blocks(self, tmp_path: Path) -> None:
        _write_synthetic_secom(tmp_path)
        prepared = prepare_data(
            tmp_path,
            test_size=0.2,
            val_size=0.1,
            random_seed=0,
            missing_threshold=0.5,
            variance_threshold=1e-6,
            correlation_threshold=0.95,
            window_sizes=None,
        )
        assert prepared.x_test_raw.shape[1] == SECOM_SENSORS
        assert abs(prepared.x_test_raw.shape[0] - 0.2 * SECOM_ROWS) <= 1
        assert len(prepared.train_windows) == 1
        block = prepared.train_windows[0]
        assert block.x.shape[1] == len(prepared.sensor_cols)
        assert block.x.shape[0] == prepared.train_windows[0].y.shape[0]
        assert prepared.pos_weight > 1.0

    def test_drops_match_leakage_rule(self, tmp_path: Path) -> None:
        _write_synthetic_secom(tmp_path)
        prepared = prepare_data(
            tmp_path,
            test_size=0.2,
            val_size=0.1,
            random_seed=0,
            missing_threshold=0.5,
            variance_threshold=1e-6,
            correlation_threshold=0.95,
        )
        cleaner = prepared.preprocessor.cleaner
        assert "sensor_000" in cleaner.dropped_high_missing
        assert "sensor_001" in cleaner.dropped_low_variance
        assert "sensor_003" in cleaner.dropped_high_correlation
        # Preprocessor stats fit only on train sub-split → medians length matches.
        assert len(cleaner.medians) == len(prepared.sensor_cols)

    def test_window_sizes_blocks(self, tmp_path: Path) -> None:
        _write_synthetic_secom(tmp_path)
        prepared = prepare_data(
            tmp_path,
            test_size=0.2,
            val_size=0.1,
            random_seed=0,
            missing_threshold=0.5,
            variance_threshold=1e-6,
            correlation_threshold=0.95,
            window_sizes=(2, 4),
        )
        assert len(prepared.train_windows) == 2
        assert prepared.train_windows[0].x.shape[1] == 2
        assert prepared.train_windows[1].x.shape[1] == 4

    def test_missing_raw_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            prepare_data(
                tmp_path,
                test_size=0.2,
                val_size=0.1,
                random_seed=0,
                missing_threshold=0.5,
                variance_threshold=1e-6,
                correlation_threshold=0.95,
            )

    def test_thresholds_drop_every_sensor(self, tmp_path: Path) -> None:
        _write_synthetic_secom(tmp_path)
        with pytest.raises(ValueError, match="No sensor columns remain"):
            prepare_data(
                tmp_path,
                test_size=0.2,
                val_size=0.1,
                random_seed=0,
                missing_threshold=0.5,
                variance_threshold=1e9,
                correlation_threshold=0.95,
            )


class TestCheckpointRoundTrip:
    """Bit-exact save/load round-trip and schema validation."""

    @pytest.mark.parametrize("early", [False, True])
    def test_roundtrip_bit_exact(self, tmp_path: Path, early: bool) -> None:
        data = _prepared_from_synthetic()
        config = TrainConfig(
            emb_dim=4,
            hidden_size=8,
            epochs=1,
            batch_size=16,
            early_prediction=early,
            window_sizes=None,
        )
        model, _ = train(data, config)
        ckpt_path = tmp_path / "ckpt.pt"
        save_checkpoint(
            ckpt_path,
            model,
            data.preprocessor,
            data.sensor_cols,
            data.pos_weight,
            config.timestep_weighting,
            config.seed,
            config.epochs,
            config.lr,
            config.batch_size,
            "cpu",
        )

        n = len(data.sensor_cols)
        x = torch.randn(3, n)
        ids = torch.from_numpy(np.tile(np.arange(n), (3, 1)))
        from yield_risk.sequence_models import predict_logits

        model.eval()
        with torch.no_grad():
            before = predict_logits(model, x, ids)

        loaded = load_checkpoint(ckpt_path, "cpu")
        with torch.no_grad():
            after = predict_logits(loaded.model, x, ids)
        assert torch.equal(before, after)

    def test_ckpt_keys_and_types(self, tmp_path: Path) -> None:
        data = _prepared_from_synthetic()
        config = TrainConfig(emb_dim=4, hidden_size=8, epochs=0, window_sizes=None)
        model, _ = train(data, config)
        ckpt_path = tmp_path / "ckpt.pt"
        save_checkpoint(
            ckpt_path,
            model,
            data.preprocessor,
            data.sensor_cols,
            data.pos_weight,
            "none",
            42,
            0,
            1e-3,
            32,
            "cpu",
        )
        raw = torch.load(ckpt_path, weights_only=False)
        assert raw["format_version"] == 1
        assert isinstance(raw["model_state_dict"], dict)
        assert isinstance(raw["n_sensors"], int)
        assert isinstance(raw["dropout"], float)
        assert isinstance(raw["early_prediction"], bool)
        assert isinstance(raw["preprocessor"], dict)
        assert isinstance(raw["sensor_cols"], list)
        assert isinstance(raw["created_at"], str)
        prep = raw["preprocessor"]
        assert len(prep["medians"]) == len(data.sensor_cols)
        assert len(prep["mean"]) == len(data.sensor_cols)
        assert len(prep["scale"]) == len(data.sensor_cols)
        assert len(prep["raw_sensor_cols"]) == len(data.raw_sensor_cols)

    def test_loaded_preprocessor_handles_short_window(self, tmp_path: Path) -> None:
        data = _prepared_from_synthetic()
        config = TrainConfig(emb_dim=4, hidden_size=8, epochs=0, window_sizes=None)
        model, _ = train(data, config)
        ckpt_path = tmp_path / "ckpt.pt"
        save_checkpoint(
            ckpt_path, model, data.preprocessor, data.sensor_cols,
            data.pos_weight, "none", 42, 0, 1e-3, 32, "cpu",
        )
        loaded = load_checkpoint(ckpt_path)
        retained = loaded.sensor_cols[0]
        x_win = np.zeros((2, 1), dtype=np.float32)
        x_norm, ids = loaded.preprocessor.transform_window(x_win, [retained])
        assert x_norm.shape == (2, 1)
        assert ids.shape == (2, 1)

    def test_save_length_mismatch(self, tmp_path: Path) -> None:
        data = _prepared_from_synthetic()
        model = build_gru(n_sensors=len(data.sensor_cols), emb_dim=4, hidden_size=8)
        with pytest.raises(ValueError, match="sensor_cols length"):
            save_checkpoint(
                tmp_path / "c.pt", model, data.preprocessor,
                data.sensor_cols + ["extra"], data.pos_weight,
                "none", 42, 0, 1e-3, 32, "cpu",
            )

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nope.pt")

    def test_missing_key(self, tmp_path: Path) -> None:
        ckpt_path = tmp_path / "bad.pt"
        torch.save({"format_version": 1}, ckpt_path)
        with pytest.raises(KeyError, match="Checkpoint missing required key"):
            load_checkpoint(ckpt_path)

    def test_bad_format_version(self, tmp_path: Path) -> None:
        data = _prepared_from_synthetic()
        config = TrainConfig(emb_dim=4, hidden_size=8, epochs=0, window_sizes=None)
        model, _ = train(data, config)
        ckpt_path = tmp_path / "ckpt.pt"
        save_checkpoint(
            ckpt_path, model, data.preprocessor, data.sensor_cols,
            data.pos_weight, "none", 42, 0, 1e-3, 32, "cpu",
        )
        raw = torch.load(ckpt_path, weights_only=False)
        raw["format_version"] = 2
        torch.save(raw, ckpt_path)
        with pytest.raises(ValueError, match="Unsupported checkpoint format_version"):
            load_checkpoint(ckpt_path)


def _trained_checkpoint(tmp_path: Path, early: bool = False) -> LoadedCheckpoint:
    data = _prepared_from_synthetic()
    config = TrainConfig(
        emb_dim=4, hidden_size=8, epochs=1, batch_size=16,
        early_prediction=early, window_sizes=None,
    )
    model, _ = train(data, config)
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path, model, data.preprocessor, data.sensor_cols,
        data.pos_weight, config.timestep_weighting, config.seed,
        config.epochs, config.lr, config.batch_size, "cpu",
    )
    return load_checkpoint(ckpt_path)


class TestEvaluate:
    """End-to-end evaluation over full rows and current windows."""

    def test_full_row_end_to_end(self, tmp_path: Path) -> None:
        loaded = _trained_checkpoint(tmp_path)
        data = _prepared_from_synthetic()
        y_prob, metrics = evaluate(
            loaded,
            data.x_test_raw,
            data.y_test,
            loaded.preprocessor.raw_sensor_cols,
        )
        assert y_prob.shape == (data.x_test_raw.shape[0],)
        assert y_prob.dtype == np.float64
        assert isinstance(metrics, SequenceMetrics)

    def test_short_window_end_to_end(self, tmp_path: Path) -> None:
        loaded = _trained_checkpoint(tmp_path)
        data = _prepared_from_synthetic()
        retained = loaded.sensor_cols
        cols = retained[:2]
        idx = [loaded.preprocessor.raw_sensor_cols.index(c) for c in cols]
        x_win = data.x_test_raw[:, idx]
        _, metrics = evaluate(loaded, x_win, data.y_test, cols)
        assert isinstance(metrics, SequenceMetrics)


class TestSensorMismatch:
    """Window-sensor and raw-schema mismatch errors."""

    def test_unknown_window_sensor(self, tmp_path: Path) -> None:
        loaded = _trained_checkpoint(tmp_path)
        with pytest.raises(ValueError, match="Unknown window sensor column"):
            evaluate(
                loaded,
                np.zeros((2, 1), dtype=np.float32),
                np.array([0, 1]),
                ["sensor_99"],
            )

    def test_all_dropped_window(self, tmp_path: Path) -> None:
        loaded = _trained_checkpoint(tmp_path)
        dropped = (
            loaded.preprocessor.cleaner.dropped_high_missing
            + loaded.preprocessor.cleaner.dropped_low_variance
            + loaded.preprocessor.cleaner.dropped_high_correlation
        )
        with pytest.raises(ValueError, match="no retained sensor columns"):
            evaluate(
                loaded,
                np.zeros((2, 1), dtype=np.float32),
                np.array([0, 1]),
                [dropped[0]],
            )


class TestResolveDevice:
    """Device resolution semantics."""

    def test_cpu(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_auto_without_cuda(self) -> None:
        if not torch.cuda.is_available():
            assert resolve_device("auto") == "cpu"

    def test_cuda_without_cuda(self) -> None:
        if not torch.cuda.is_available():
            with pytest.raises(ValueError, match="CUDA requested but not available"):
                resolve_device("cuda")


class TestBaselineComparison:
    """Baseline loading, fallback, and comparison output."""

    def test_comparison_primary(self, tmp_path: Path) -> None:
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "model_comparison.json").write_text(
            json.dumps(
                [
                    {"model": "rf", "test_pr_auc": 0.2, "test_roc_auc": 0.75,
                     "test_recall": 0.5, "test_precision": 0.17, "selected": True},
                    {"model": "lr", "test_pr_auc": 0.1, "test_roc_auc": 0.6,
                     "test_recall": 0.4, "test_precision": 0.1, "selected": False},
                ]
            )
        )
        baseline = load_tabular_baseline(reports, tmp_path / "models")
        assert baseline is not None
        assert baseline.source == "model_comparison.json"
        assert baseline.model == "rf"
        assert baseline.test_pr_auc == 0.2

        seq = compute_sequence_metrics(
            np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])
        )
        write_baseline_comparison(seq, baseline, reports)
        payload = json.loads(
            (reports / "sequence_model_comparison.json").read_text()
        )
        assert isinstance(payload["delta_pr_auc"], float)
        assert payload["verdict"] in {
            "sequence_better", "baseline_better", "tie",
        }
        assert (reports / "sequence_model_comparison.csv").exists()

    def test_metadata_fallback(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        (models / "model_metadata.json").write_text(
            json.dumps(
                {
                    "model_version": "rf-abc",
                    "metrics": {"pr_auc": 0.19, "roc_auc": 0.75,
                                "recall": 0.57, "precision": 0.17},
                }
            )
        )
        baseline = load_tabular_baseline(tmp_path / "reports", models)
        assert baseline is not None
        assert baseline.source == "model_metadata.json"
        assert baseline.model == "rf-abc"
        assert baseline.test_pr_auc == 0.19

    def test_both_absent(self, tmp_path: Path) -> None:
        reports = tmp_path / "reports"
        reports.mkdir()
        baseline = load_tabular_baseline(reports, tmp_path / "models")
        assert baseline is None
        seq = compute_sequence_metrics(
            np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])
        )
        write_baseline_comparison(seq, None, reports)
        payload = json.loads(
            (reports / "sequence_model_comparison.json").read_text()
        )
        assert payload["baseline"] is None
        assert payload["verdict"] == "no_baseline"
        assert payload["delta_pr_auc"] is None

    def test_malformed_json(self, tmp_path: Path) -> None:
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "model_comparison.json").write_text("{not valid json")
        baseline = load_tabular_baseline(reports, tmp_path / "models")
        assert baseline is None


class TestSmoke:
    """End-to-end smoke on synthetic arrays for both modes."""

    @pytest.mark.parametrize("early", [False, True])
    def test_full_pipeline(self, tmp_path: Path, early: bool) -> None:
        data = _prepared_from_synthetic()
        config = TrainConfig(
            emb_dim=4, hidden_size=8, epochs=2, batch_size=16,
            early_prediction=early, window_sizes=None,
        )
        model, history = train(data, config)
        assert len(history) == 2
        ckpt_path = tmp_path / "ckpt.pt"
        save_checkpoint(
            ckpt_path, model, data.preprocessor, data.sensor_cols,
            data.pos_weight, config.timestep_weighting, config.seed,
            config.epochs, config.lr, config.batch_size, "cpu",
        )
        loaded = load_checkpoint(ckpt_path)
        y_prob, metrics = evaluate(
            loaded, data.x_test_raw, data.y_test,
            loaded.preprocessor.raw_sensor_cols,
        )
        assert y_prob.shape[0] == data.x_test_raw.shape[0]
        write_baseline_comparison(metrics, None, tmp_path)
        assert (tmp_path / "sequence_model_comparison.json").exists()


class TestDeterminism:
    """Identical config + data → bit-identical weights."""

    def test_bit_identical(self) -> None:
        data1 = _prepared_from_synthetic()
        data2 = _prepared_from_synthetic()
        config = TrainConfig(
            emb_dim=4, hidden_size=8, epochs=2, batch_size=16, window_sizes=None
        )
        model1, _ = train(data1, config)
        model2, _ = train(data2, config)
        sd1 = model1.state_dict()
        sd2 = model2.state_dict()
        assert sd1.keys() == sd2.keys()
        for key in sd1:
            assert torch.equal(sd1[key], sd2[key])


class TestEdgeCases:
    """Zero-epoch and empty-val edge cases."""

    def test_zero_epochs(self, tmp_path: Path) -> None:
        data = _prepared_from_synthetic()
        config = TrainConfig(emb_dim=4, hidden_size=8, epochs=0, window_sizes=None)
        model, history = train(data, config)
        assert history == []
        ckpt_path = tmp_path / "ckpt.pt"
        save_checkpoint(
            ckpt_path, model, data.preprocessor, data.sensor_cols,
            data.pos_weight, "none", config.seed, 0, config.lr,
            config.batch_size, "cpu",
        )
        raw = torch.load(ckpt_path, weights_only=False)
        assert raw["epochs"] == 0

    def test_empty_val_metrics_nan(self) -> None:
        data = _prepared_from_synthetic()
        assert data.val_windows == []
        config = TrainConfig(
            emb_dim=4, hidden_size=8, epochs=1, batch_size=16, window_sizes=None
        )
        _, history = train(data, config)
        assert math.isnan(history[0]["val_loss"])
        assert math.isnan(history[0]["val_pr_auc"])
