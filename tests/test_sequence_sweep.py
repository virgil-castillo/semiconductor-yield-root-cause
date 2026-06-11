"""Tests for the GRU hyperparameter sweep grid generator."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from yield_risk.sequence_sweep import TrialResult, TrialSpec
from yield_risk.sequence_train import (
    PreparedData,
    TrainConfig,
    WindowArrays,
)

# ---------------------------------------------------------------------------
# Helpers shared by run_trial tests
# ---------------------------------------------------------------------------

def _make_two_class_prepared(tmp_path: Path) -> PreparedData:  # noqa: ARG001
    """Build a tiny PreparedData with two-class val_windows."""
    from tests.conftest import make_synthetic
    from yield_risk.sequence_models import SequenceSensorPipeline

    n_rows = 60
    n_sensors = 8
    x, y = make_synthetic(n_rows=n_rows, n_sensors=n_sensors, n_pos=15, seed=0)
    raw_cols = [f"sensor_{i}" for i in range(n_sensors)]

    # Train/val split: first 40 train, last 20 val (both two-class)
    x_train, x_val = x[:40], x[40:]
    y_train, y_val = y[:40], y[40:]

    # Ensure val has at least one of each class
    y_val[0] = 0
    y_val[1] = 1

    pipe = SequenceSensorPipeline.fit(
        x_train, raw_cols, 0.5, 1e-6, 0.95
    )
    sensor_cols = pipe.sensor_cols
    x_norm_tr, ids_tr = pipe.transform_full(x_train, raw_cols)
    x_norm_val, ids_val = pipe.transform_full(x_val, raw_cols)

    train_windows = [
        WindowArrays(x_norm_tr, ids_tr, y_train.astype(np.int64), list(sensor_cols))
    ]
    val_windows = [
        WindowArrays(x_norm_val, ids_val, y_val.astype(np.int64), list(sensor_cols))
    ]

    n_pos_tr = int((y_train == 1).sum())
    n_neg_tr = int((y_train == 0).sum())
    pos_weight = float(n_neg_tr) / float(max(n_pos_tr, 1))

    return PreparedData(
        train_windows=train_windows,
        val_windows=val_windows,
        x_test_raw=x_val,
        y_test=y_val.astype(np.int64),
        raw_sensor_cols=raw_cols,
        sensor_cols=list(sensor_cols),
        preprocessor=pipe,
        pos_weight=pos_weight,
    )


def _make_tiny_spec() -> TrialSpec:
    config = TrainConfig(
        emb_dim=8,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        lr=0.001,
        batch_size=32,
        epochs=1,
        seed=0,
        device="cpu",
        num_workers=0,
    )
    return TrialSpec(trial_id="trial_000", config=config)


# ---------------------------------------------------------------------------
# run_trial: success path (two-class val)
# ---------------------------------------------------------------------------


def test_run_trial_success_path(tmp_path: Path) -> None:
    """run_trial with two-class val produces ok status with all artifacts."""
    from yield_risk.sequence_sweep import run_trial

    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    spec = _make_tiny_spec()
    data = _make_two_class_prepared(tmp_path)

    result = run_trial(spec, data, models_dir, reports_dir, device="cpu")

    # Status
    assert result.status == "ok"
    assert result.error is None

    # Checkpoint file exists
    ckpt_path = models_dir / "sequence_sweep" / spec.trial_id / "sequence_gru.pt"
    assert ckpt_path.exists(), f"checkpoint not found at {ckpt_path}"
    assert result.checkpoint_path == str(ckpt_path)

    # History JSON exists and val_pr_auc matches final epoch
    hist_path = (
        reports_dir / "sequence_sweep" / spec.trial_id / "sequence_train_history.json"
    )
    assert hist_path.exists(), f"history JSON not found at {hist_path}"
    history = json.loads(hist_path.read_text())
    assert len(history) == 1  # epochs=1
    assert result.val_pr_auc == history[-1]["val_pr_auc"]

    # n_params is a positive int
    assert isinstance(result.n_params, int)
    assert result.n_params > 0

    # val_roc_auc is a float (two-class val)
    assert isinstance(result.val_roc_auc, float)

    # Swept params copied from spec.config
    assert result.emb_dim == spec.config.emb_dim
    assert result.hidden_size == spec.config.hidden_size
    assert result.num_layers == spec.config.num_layers
    assert result.dropout == spec.config.dropout
    assert result.lr == spec.config.lr
    assert result.batch_size == spec.config.batch_size


# ---------------------------------------------------------------------------
# run_trial: empty-val path
# ---------------------------------------------------------------------------


def test_run_trial_empty_val(tmp_path: Path) -> None:
    """run_trial with empty val_windows gives ok status, None roc_auc, NaN pr_auc."""
    from tests.conftest import make_synthetic
    from yield_risk.sequence_models import SequenceSensorPipeline
    from yield_risk.sequence_sweep import run_trial

    n_sensors = 8
    x, y = make_synthetic(n_rows=40, n_sensors=n_sensors, n_pos=10, seed=1)
    raw_cols = [f"sensor_{i}" for i in range(n_sensors)]
    pipe = SequenceSensorPipeline.fit(x, raw_cols, 0.5, 1e-6, 0.95)
    sensor_cols = pipe.sensor_cols
    x_norm, ids = pipe.transform_full(x, raw_cols)
    train_windows = [
        WindowArrays(x_norm, ids, y.astype(np.int64), list(sensor_cols))
    ]
    n_pos_tr = int((y == 1).sum())
    n_neg_tr = int((y == 0).sum())
    data = PreparedData(
        train_windows=train_windows,
        val_windows=[],
        x_test_raw=x[:5],
        y_test=y[:5].astype(np.int64),
        raw_sensor_cols=raw_cols,
        sensor_cols=list(sensor_cols),
        preprocessor=pipe,
        pos_weight=float(n_neg_tr) / float(max(n_pos_tr, 1)),
    )

    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    spec = _make_tiny_spec()

    result = run_trial(spec, data, models_dir, reports_dir, device="cpu")

    assert result.status == "ok"
    assert result.val_roc_auc is None
    assert math.isnan(result.val_pr_auc)

    ckpt_path = models_dir / "sequence_sweep" / spec.trial_id / "sequence_gru.pt"
    assert ckpt_path.exists()


# ---------------------------------------------------------------------------
# run_trial: failure path (monkeypatched train raises)
# ---------------------------------------------------------------------------


def test_run_trial_failure_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_trial records a failed trial when train raises; no checkpoint written."""
    from yield_risk.sequence_sweep import run_trial

    def _boom(
        data: PreparedData, config: TrainConfig
    ) -> object:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("yield_risk.sequence_sweep.train", _boom)

    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    spec = _make_tiny_spec()
    data = _make_two_class_prepared(tmp_path)

    result = run_trial(spec, data, models_dir, reports_dir, device="cpu")

    assert result.status == "failed"
    assert result.error is not None and len(result.error) > 0
    assert result.checkpoint_path is None

    ckpt_path = models_dir / "sequence_sweep" / spec.trial_id / "sequence_gru.pt"
    assert not ckpt_path.exists(), "checkpoint should not exist on failure"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_grid() -> list:  # type: ignore[type-arg]
    from yield_risk.sequence_sweep import build_sweep_grid
    base = TrainConfig()
    return build_sweep_grid(base)


# ---------------------------------------------------------------------------
# Grid size
# ---------------------------------------------------------------------------

def test_build_sweep_grid_returns_108_specs() -> None:
    """Grid must contain exactly 108 TrialSpec objects."""
    grid = _get_grid()
    assert len(grid) == 108


# ---------------------------------------------------------------------------
# trial_id format and uniqueness
# ---------------------------------------------------------------------------

def test_trial_ids_are_unique() -> None:
    """Every trial_id in the grid must be distinct."""
    grid = _get_grid()
    ids = [spec.trial_id for spec in grid]
    assert len(ids) == len(set(ids))


def test_trial_ids_follow_zero_padded_format() -> None:
    """trial_ids must match 'trial_NNN' with zero-padded three digits."""
    grid = _get_grid()
    pattern = re.compile(r"^trial_\d{3}$")
    for spec in grid:
        assert pattern.match(spec.trial_id), (
            f"trial_id {spec.trial_id!r} does not match 'trial_NNN'"
        )


def test_trial_ids_are_in_ascending_order() -> None:
    """trial_ids must be in strict ascending enumeration order."""
    grid = _get_grid()
    ids = [spec.trial_id for spec in grid]
    assert ids == sorted(ids)
    # Also verify they start at trial_000 and end at trial_107
    assert ids[0] == "trial_000"
    assert ids[-1] == "trial_107"


# ---------------------------------------------------------------------------
# (num_layers, dropout) coupling
# ---------------------------------------------------------------------------

def test_num_layers_1_always_has_dropout_0() -> None:
    """Every spec with num_layers==1 must have dropout==0.0."""
    grid = _get_grid()
    for spec in grid:
        if spec.config.num_layers == 1:
            assert spec.config.dropout == 0.0, (
                f"Expected dropout 0.0 for num_layers=1, got {spec.config.dropout}"
            )


def test_num_layers_2_always_has_nonzero_dropout() -> None:
    """Every spec with num_layers==2 must have dropout in {0.1, 0.3}."""
    grid = _get_grid()
    for spec in grid:
        if spec.config.num_layers == 2:
            assert spec.config.dropout in (0.1, 0.3), (
                f"Expected dropout in {{0.1, 0.3}} for num_layers=2, "
                f"got {spec.config.dropout}"
            )


def test_no_invalid_layer_dropout_combos() -> None:
    """No (num_layers=1, dropout!=0.0) or (num_layers=2, dropout=0.0) combos."""
    grid = _get_grid()
    combos = {(spec.config.num_layers, spec.config.dropout) for spec in grid}
    assert (1, 0.1) not in combos
    assert (1, 0.3) not in combos
    assert (2, 0.0) not in combos


# ---------------------------------------------------------------------------
# Swept values coverage
# ---------------------------------------------------------------------------

def test_all_emb_dim_values_appear() -> None:
    """The set of emb_dim values across all specs must be exactly {8, 16, 32}."""
    grid = _get_grid()
    assert {spec.config.emb_dim for spec in grid} == {8, 16, 32}


def test_all_hidden_size_values_appear() -> None:
    """The set of hidden_size values across all specs must be {32, 64, 128}."""
    grid = _get_grid()
    assert {spec.config.hidden_size for spec in grid} == {32, 64, 128}


def test_all_num_layers_values_appear() -> None:
    """Both num_layers values (1 and 2) must appear in the grid."""
    grid = _get_grid()
    assert {spec.config.num_layers for spec in grid} == {1, 2}


def test_all_dropout_values_appear() -> None:
    """All three dropout values (0.0, 0.1, 0.3) must appear in the grid."""
    grid = _get_grid()
    assert {spec.config.dropout for spec in grid} == {0.0, 0.1, 0.3}


def test_all_lr_values_appear() -> None:
    """Both lr values (0.001, 0.0003) must appear in the grid."""
    grid = _get_grid()
    assert {spec.config.lr for spec in grid} == {0.001, 0.0003}


def test_all_batch_size_values_appear() -> None:
    """Both batch_size values (32, 64) must appear in the grid."""
    grid = _get_grid()
    assert {spec.config.batch_size for spec in grid} == {32, 64}


# ---------------------------------------------------------------------------
# Held-fixed fields are inherited from base
# ---------------------------------------------------------------------------

def test_held_fixed_fields_inherited_from_base() -> None:
    """Non-swept fields (epochs, seed, window_sizes) must be inherited from base."""
    from yield_risk.sequence_sweep import build_sweep_grid

    base = TrainConfig(
        epochs=99,
        seed=7,
        val_size=0.25,
        window_sizes=(16, 32),
    )
    grid = build_sweep_grid(base)
    # Sample a few non-consecutive specs to avoid only testing one combo
    for idx in (0, 17, 53, 107):
        spec = grid[idx]
        assert spec.config.epochs == 99, (
            f"spec[{idx}].config.epochs expected 99, got {spec.config.epochs}"
        )
        assert spec.config.seed == 7, (
            f"spec[{idx}].config.seed expected 7, got {spec.config.seed}"
        )
        assert spec.config.val_size == 0.25, (
            f"spec[{idx}].config.val_size expected 0.25, got {spec.config.val_size}"
        )
        assert spec.config.window_sizes == (16, 32), (
            f"spec[{idx}].config.window_sizes expected (16, 32), "
            f"got {spec.config.window_sizes}"
        )


# ---------------------------------------------------------------------------
# TrialSpec is frozen (immutable)
# ---------------------------------------------------------------------------

def test_trial_spec_is_frozen() -> None:
    """TrialSpec must be a frozen dataclass — mutation raises FrozenInstanceError."""
    from yield_risk.sequence_sweep import build_sweep_grid
    grid = build_sweep_grid(TrainConfig())
    spec = grid[0]
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        spec.trial_id = "hacked"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TrialResult + rank_trials + select_best
# ---------------------------------------------------------------------------


def _make_ok(
    trial_id: str,
    val_pr_auc: float,
    val_loss: float,
    n_params: int | None = 100,
    val_roc_auc: float | None = None,
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        emb_dim=8,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        lr=0.001,
        batch_size=32,
        val_loss=val_loss,
        val_pr_auc=val_pr_auc,
        val_roc_auc=val_roc_auc,
        n_params=n_params,
        checkpoint_path="/tmp/ckpt",
        status="ok",
        error=None,
    )


def _make_failed(trial_id: str) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        emb_dim=8,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        lr=0.001,
        batch_size=32,
        val_loss=math.nan,
        val_pr_auc=math.nan,
        val_roc_auc=None,
        n_params=None,
        checkpoint_path=None,
        status="failed",
        error="RuntimeError: something went wrong",
    )


def test_rank_trials_higher_pr_auc_ranks_first() -> None:
    """The trial with higher val_pr_auc must rank first."""
    from yield_risk.sequence_sweep import rank_trials

    low = _make_ok("trial_001", val_pr_auc=0.70, val_loss=0.5)
    high = _make_ok("trial_002", val_pr_auc=0.90, val_loss=0.5)
    ranked = rank_trials([low, high])
    assert ranked[0].trial_id == "trial_002"
    assert ranked[1].trial_id == "trial_001"


def test_rank_trials_equal_pr_auc_lower_loss_wins() -> None:
    """When val_pr_auc is equal, the trial with lower val_loss ranks first."""
    from yield_risk.sequence_sweep import rank_trials

    worse_loss = _make_ok("trial_001", val_pr_auc=0.80, val_loss=0.9)
    better_loss = _make_ok("trial_002", val_pr_auc=0.80, val_loss=0.3)
    ranked = rank_trials([worse_loss, better_loss])
    assert ranked[0].trial_id == "trial_002"
    assert ranked[1].trial_id == "trial_001"


def test_rank_trials_equal_pr_auc_equal_loss_smaller_params_wins() -> None:
    """When val_pr_auc and val_loss are equal, smaller n_params ranks first."""
    from yield_risk.sequence_sweep import rank_trials

    large = _make_ok("trial_001", val_pr_auc=0.80, val_loss=0.5, n_params=500)
    small = _make_ok("trial_002", val_pr_auc=0.80, val_loss=0.5, n_params=100)
    ranked = rank_trials([large, small])
    assert ranked[0].trial_id == "trial_002"
    assert ranked[1].trial_id == "trial_001"


def test_rank_trials_failed_always_below_successful() -> None:
    """A failed trial ranks below every successful trial, even one with low PR-AUC."""
    from yield_risk.sequence_sweep import rank_trials

    low_auc_ok = _make_ok("trial_001", val_pr_auc=0.10, val_loss=9.9)
    failed = _make_failed("trial_002")
    ranked = rank_trials([failed, low_auc_ok])
    assert ranked[0].trial_id == "trial_001"
    assert ranked[1].trial_id == "trial_002"


def test_rank_trials_nan_pr_auc_successful_below_real_but_above_failed() -> None:
    """NaN val_pr_auc ok-trial ranks below real-valued ok-trials but above failed."""
    from yield_risk.sequence_sweep import rank_trials

    real_ok = _make_ok("trial_001", val_pr_auc=0.50, val_loss=0.5)
    nan_ok = _make_ok("trial_002", val_pr_auc=math.nan, val_loss=0.5)
    failed = _make_failed("trial_003")
    ranked = rank_trials([failed, nan_ok, real_ok])
    assert ranked[0].trial_id == "trial_001"
    assert ranked[1].trial_id == "trial_002"
    assert ranked[2].trial_id == "trial_003"


def test_select_best_returns_top_trial() -> None:
    """select_best returns the highest-ranked successful trial."""
    from yield_risk.sequence_sweep import select_best

    low = _make_ok("trial_001", val_pr_auc=0.70, val_loss=0.5)
    high = _make_ok("trial_002", val_pr_auc=0.90, val_loss=0.5)
    best = select_best([low, high])
    assert best is not None
    assert best.trial_id == "trial_002"


def test_select_best_returns_none_when_all_failed() -> None:
    """select_best returns None when every trial failed."""
    from yield_risk.sequence_sweep import select_best

    results = [_make_failed("trial_000"), _make_failed("trial_001")]
    assert select_best(results) is None


def test_select_best_returns_none_on_empty_list() -> None:
    """select_best returns None for an empty results list."""
    from yield_risk.sequence_sweep import select_best

    assert select_best([]) is None


def test_rank_trials_does_not_mutate_input() -> None:
    """rank_trials must return a new list and leave the input list unchanged."""
    from yield_risk.sequence_sweep import rank_trials

    a = _make_ok("trial_001", val_pr_auc=0.90, val_loss=0.5)
    b = _make_ok("trial_002", val_pr_auc=0.70, val_loss=0.5)
    original = [a, b]
    original_copy = list(original)
    result = rank_trials(original)
    assert original == original_copy, "rank_trials mutated the input list"
    assert result is not original, "rank_trials must return a new list"
