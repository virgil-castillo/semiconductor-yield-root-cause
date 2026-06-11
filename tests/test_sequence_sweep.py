"""Tests for the GRU hyperparameter sweep grid generator."""
from __future__ import annotations

import math
import re

import pytest

from yield_risk.sequence_sweep import TrialResult
from yield_risk.sequence_train import TrainConfig  # noqa: E402

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
