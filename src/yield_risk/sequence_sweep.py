"""GRU hyperparameter sweep grid generator.

This module owns the sweep parameter grid definition, the ``TrialSpec``
container, the ``TrialResult`` container, and ranking/selection helpers.
It does NOT run trials; that is a later task.

Grid enumeration order (outermost to innermost loop):
    1. emb_dim       — [8, 16, 32]
    2. hidden_size   — [32, 64, 128]
    3. (num_layers, dropout) — [(1, 0.0), (2, 0.1), (2, 0.3)]
    4. lr            — [0.001, 0.0003]
    5. batch_size    — [32, 64]

Total: 3 × 3 × 3 × 2 × 2 = 108 trials.

``trial_id`` values are zero-padded three-digit strings (``trial_000`` …
``trial_107``) assigned in the enumeration order above, making them stable
and deterministic across runs.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from yield_risk.sequence_train import TrainConfig

# ---------------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------------

_EMB_DIMS: list[int] = [8, 16, 32]
_HIDDEN_SIZES: list[int] = [32, 64, 128]
# (num_layers, dropout) coupled combos — NOT an independent cross product.
_LAYER_DROPOUT_COMBOS: list[tuple[int, float]] = [(1, 0.0), (2, 0.1), (2, 0.3)]
_LRS: list[float] = [0.001, 0.0003]
_BATCH_SIZES: list[int] = [32, 64]


# ---------------------------------------------------------------------------
# TrialSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialSpec:
    """An immutable container pairing a trial identifier with its config.

    Attributes:
        trial_id: Zero-padded deterministic identifier (e.g. ``"trial_000"``).
        config: The full ``TrainConfig`` for this trial, derived from a base
            config via ``dataclasses.replace``.
    """

    trial_id: str
    config: TrainConfig


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------


def build_sweep_grid(base: TrainConfig) -> list[TrialSpec]:
    """Return the 108-trial hyperparameter grid as a list of ``TrialSpec``.

    The grid is the cross product of:

    * ``emb_dim`` in ``[8, 16, 32]``
    * ``hidden_size`` in ``[32, 64, 128]``
    * ``(num_layers, dropout)`` in ``[(1, 0.0), (2, 0.1), (2, 0.3)]``
    * ``lr`` in ``[0.001, 0.0003]``
    * ``batch_size`` in ``[32, 64]``

    yielding 3 × 3 × 3 × 2 × 2 = 108 specs.

    Only the six swept fields are overridden on ``base``; all other fields
    (``epochs``, ``seed``, ``val_size``, ``window_sizes``, ``device``,
    ``num_workers``) are inherited unchanged.

    Args:
        base: The base ``TrainConfig`` whose non-swept fields are preserved
            on every generated trial config.

    Returns:
        A list of 108 ``TrialSpec`` objects in deterministic enumeration
        order (see module docstring).
    """
    specs: list[TrialSpec] = []
    i = 0
    for emb_dim in _EMB_DIMS:
        for hidden_size in _HIDDEN_SIZES:
            for num_layers, dropout in _LAYER_DROPOUT_COMBOS:
                for lr in _LRS:
                    for batch_size in _BATCH_SIZES:
                        config = dataclasses.replace(
                            base,
                            emb_dim=emb_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            lr=lr,
                            batch_size=batch_size,
                        )
                        specs.append(
                            TrialSpec(trial_id=f"trial_{i:03d}", config=config)
                        )
                        i += 1
    return specs


# ---------------------------------------------------------------------------
# TrialResult
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Container for the outcome of a single sweep trial.

    Attributes:
        trial_id: Identifier matching the corresponding ``TrialSpec``.
        emb_dim: Embedding dimension used in this trial.
        hidden_size: GRU hidden size used in this trial.
        num_layers: Number of GRU layers used in this trial.
        dropout: Dropout rate used in this trial.
        lr: Learning rate used in this trial.
        batch_size: Batch size used in this trial.
        val_loss: Validation loss; NaN when unavailable.
        val_pr_auc: Validation PR-AUC; NaN when unavailable.
        val_roc_auc: Validation ROC-AUC; None when not available.
        n_params: Total trainable parameter count; None for failed trials.
        checkpoint_path: Path to saved checkpoint; None for failed trials.
        status: ``"ok"`` for a successful trial, ``"failed"`` otherwise.
        error: Error message string for failed trials; None on success.
    """

    trial_id: str
    emb_dim: int
    hidden_size: int
    num_layers: int
    dropout: float
    lr: float
    batch_size: int
    val_loss: float
    val_pr_auc: float
    val_roc_auc: float | None
    n_params: int | None
    checkpoint_path: str | None
    status: str
    error: str | None


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


def rank_trials(results: list[TrialResult]) -> list[TrialResult]:
    """Return a new list of trials sorted by the sweep selection rule.

    Selection rule (descending priority):

    1. Successful trials (``status == "ok"``) rank above failed ones.
    2. Among successful trials: primary sort key is ``val_pr_auc`` descending
       (higher is better); NaN treated as worst.
    3. Tie-break 1: ``val_loss`` ascending (lower is better); NaN treated as
       worst.
    4. Tie-break 2: ``n_params`` ascending (smaller is better); None treated
       as worst.
    5. Failed trials are sorted among themselves by ``trial_id`` ascending for
       a deterministic, stable total ordering.

    The input list is not mutated.

    Args:
        results: The list of ``TrialResult`` objects to rank.

    Returns:
        A new list containing the same objects in ranked order.
    """
    _BIG = float("inf")

    def _sort_key(r: TrialResult) -> tuple[int, float, float, float, str]:
        if r.status != "ok":
            # Failed trials go last; break ties by trial_id ascending.
            return (1, _BIG, _BIG, _BIG, r.trial_id)
        # Negate pr_auc so that higher values sort first (ascending sort).
        pr_auc = r.val_pr_auc if not math.isnan(r.val_pr_auc) else -_BIG
        neg_pr_auc = -pr_auc
        loss = r.val_loss if not math.isnan(r.val_loss) else _BIG
        n_params = float(r.n_params) if r.n_params is not None else _BIG
        return (0, neg_pr_auc, loss, n_params, r.trial_id)

    return sorted(results, key=_sort_key)


def select_best(results: list[TrialResult]) -> TrialResult | None:
    """Return the top-ranked successful trial, or ``None`` if none exist.

    Args:
        results: The list of ``TrialResult`` objects to evaluate.

    Returns:
        The best ``TrialResult`` by the sweep selection rule, or ``None``
        when *results* is empty or every trial has ``status == "failed"``.
    """
    ranked = rank_trials(results)
    if not ranked or ranked[0].status != "ok":
        return None
    return ranked[0]
