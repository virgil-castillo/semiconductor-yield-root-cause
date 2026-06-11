"""GRU hyperparameter sweep grid generator.

This module owns the sweep parameter grid definition and the ``TrialSpec``
container.  It does NOT run trials; that is a later task.

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
