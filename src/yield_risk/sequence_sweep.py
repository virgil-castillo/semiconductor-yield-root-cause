"""GRU hyperparameter sweep grid generator.

This module owns the sweep parameter grid definition, the ``TrialSpec``
container, the ``TrialResult`` container, ranking/selection helpers, and the
single-trial runner ``run_trial``.

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

import csv
import dataclasses
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from yield_risk.sequence_models import (
    SecomGRU,
    SecomSequenceDataset,
    make_loader,
    predict_logits,
)
from yield_risk.sequence_train import (
    PreparedData,
    TrainConfig,
    _json_safe,
    compute_sequence_metrics,
    save_checkpoint,
    train,
)

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


def _val_roc_auc(
    model: SecomGRU,
    data: PreparedData,
    config: TrainConfig,
    device: str,
) -> float | None:
    """Score all val_windows and return ROC-AUC, or None if unavailable.

    Args:
        model: The trained GRU in eval() mode.
        data: Prepared data with val window blocks.
        config: Hyperparameters (batch_size, num_workers, seed).
        device: Compute device.

    Returns:
        ROC-AUC as a float, or ``None`` when val is empty, single-class, or
        the computed value is NaN.
    """
    if not data.val_windows:
        return None

    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for block in data.val_windows:
            loader = make_loader(
                SecomSequenceDataset(block.x, block.sensor_ids, block.y),
                batch_size=config.batch_size,
                shuffle=False,
                seed=config.seed,
                num_workers=config.num_workers,
            )
            for x_b, sensor_ids_b, y_b in loader:
                x_b = x_b.to(device)
                sensor_ids_b = sensor_ids_b.to(device)
                logits = predict_logits(model, x_b, sensor_ids_b)
                probs.append(
                    torch.sigmoid(logits).cpu().numpy().astype(np.float64)
                )
                labels.append(y_b.numpy().astype(np.int64))

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)

    if len(np.unique(y_true)) < 2:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = compute_sequence_metrics(y_true, y_prob)

    roc = metrics.roc_auc
    if math.isnan(roc):
        return None
    return float(roc)


_SWEEP_CSV_COLUMNS: list[str] = [
    "trial_id",
    "emb_dim",
    "hidden_size",
    "num_layers",
    "dropout",
    "lr",
    "batch_size",
    "val_loss",
    "val_pr_auc",
    "val_roc_auc",
    "n_params",
    "checkpoint_path",
    "status",
    "error",
]


def write_sweep_results(ranked: list[TrialResult], reports_dir: Path) -> None:
    """Write ranked sweep results as JSON and CSV into ``reports_dir``.

    Both files are written in the order of ``ranked`` — the caller is
    responsible for sorting (typically the output of ``rank_trials``).

    Args:
        ranked: Ordered list of ``TrialResult`` objects to serialize.
        reports_dir: Directory for the output files; created if absent.

    Returns:
        None.  Side-effects: writes ``sequence_sweep_results.json`` and
        ``sequence_sweep_results.csv`` under ``reports_dir``.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    def _cell(value: object) -> object:
        """Return empty string for None or NaN, otherwise the value.

        Args:
            value: A scalar value.

        Returns:
            ``""`` if ``value`` is ``None`` or a ``nan`` float, else ``value``.
        """
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return value

    # --- JSON ---
    payload: list[dict[str, object]] = []
    for r in ranked:
        payload.append(
            {
                "trial_id": r.trial_id,
                "emb_dim": r.emb_dim,
                "hidden_size": r.hidden_size,
                "num_layers": r.num_layers,
                "dropout": r.dropout,
                "lr": r.lr,
                "batch_size": r.batch_size,
                "val_loss": _json_safe(r.val_loss),
                "val_pr_auc": _json_safe(r.val_pr_auc),
                "val_roc_auc": r.val_roc_auc,
                "n_params": r.n_params,
                "checkpoint_path": r.checkpoint_path,
                "status": r.status,
                "error": r.error,
            }
        )
    (reports_dir / "sequence_sweep_results.json").write_text(
        json.dumps(payload, indent=2)
    )

    # --- CSV ---
    csv_path = reports_dir / "sequence_sweep_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_SWEEP_CSV_COLUMNS)
        for r in ranked:
            writer.writerow(
                [
                    _cell(r.trial_id),
                    _cell(r.emb_dim),
                    _cell(r.hidden_size),
                    _cell(r.num_layers),
                    _cell(r.dropout),
                    _cell(r.lr),
                    _cell(r.batch_size),
                    _cell(r.val_loss),
                    _cell(r.val_pr_auc),
                    _cell(r.val_roc_auc),
                    _cell(r.n_params),
                    _cell(r.checkpoint_path),
                    _cell(r.status),
                    _cell(r.error),
                ]
            )


def run_trial(
    spec: TrialSpec,
    data: PreparedData,
    models_dir: Path,
    reports_dir: Path,
    device: str = "cpu",
) -> TrialResult:
    """Run one sweep trial; return a ``TrialResult``.

    Creates output directories, trains the model, saves the checkpoint and
    training history, then computes validation metrics. Any exception is caught
    and recorded as a failed trial rather than propagating.

    Args:
        spec: The trial specification with trial_id and config.
        data: Prepared leakage-free training data.
        models_dir: Root models directory; checkpoint written under
            ``models_dir/sequence_sweep/<trial_id>/sequence_gru.pt``.
        reports_dir: Root reports directory; history written under
            ``reports_dir/sequence_sweep/<trial_id>/sequence_train_history.json``.
        device: Compute device string.

    Returns:
        A ``TrialResult`` with ``status="ok"`` on success or
        ``status="failed"`` on any exception.
    """
    trial_models_dir = models_dir / "sequence_sweep" / spec.trial_id
    trial_reports_dir = reports_dir / "sequence_sweep" / spec.trial_id
    trial_models_dir.mkdir(parents=True, exist_ok=True)
    trial_reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = spec.config

    try:
        model, history = train(data, cfg)

        checkpoint_path = trial_models_dir / "sequence_gru.pt"
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            preprocessor=data.preprocessor,
            sensor_cols=data.sensor_cols,
            pos_weight=data.pos_weight,
            random_seed=cfg.seed,
            epochs=cfg.epochs,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            device=cfg.device,
        )

        history_path = trial_reports_dir / "sequence_train_history.json"
        history_path.write_text(json.dumps(history, indent=2))

        if history:
            last = history[-1]
            val_loss = float(last["val_loss"])
            val_pr_auc = float(last["val_pr_auc"])
        else:
            val_loss = math.nan
            val_pr_auc = math.nan

        roc_auc = _val_roc_auc(model, data, cfg, device)
        n_params = sum(p.numel() for p in model.parameters())

        return TrialResult(
            trial_id=spec.trial_id,
            emb_dim=cfg.emb_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            val_loss=val_loss,
            val_pr_auc=val_pr_auc,
            val_roc_auc=roc_auc,
            n_params=n_params,
            checkpoint_path=str(checkpoint_path),
            status="ok",
            error=None,
        )

    except Exception as exc:  # noqa: BLE001
        return TrialResult(
            trial_id=spec.trial_id,
            emb_dim=cfg.emb_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            val_loss=math.nan,
            val_pr_auc=math.nan,
            val_roc_auc=None,
            n_params=None,
            checkpoint_path=None,
            status="failed",
            error=str(exc),
        )
