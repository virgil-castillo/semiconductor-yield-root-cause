"""Orchestration for the experimental GRU sequence model.

This module owns config loading, leakage-free data preparation, the fixed-epoch
training loop, checkpoint save/load, the sequence metric set, and the tabular
baseline comparison. It is isolated from the tabular pipeline and writes only to
namespaced ``sequence_*`` artifacts.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import functional as F

from yield_risk.data import load_secom
from yield_risk.evaluate import compute_metrics
from yield_risk.sequence_models import (
    SecomGRU,
    SecomSequenceDataset,
    SequenceSensorPipeline,
    _as_str_list,
    build_gru,
    build_timestep_weights,
    compute_pos_weight,
    make_loader,
    predict_logits,
)
from yield_risk.validation import validate_secom

_VALID_WEIGHTING = {"none", "linear", "sqrt"}


# --------------------------------------------------------------------------- #
# §10.0 Config
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """In-memory carrier of all GRU hyperparameters and their defaults.

    Attributes:
        emb_dim: Sensor-ID embedding dimension.
        hidden_size: GRU hidden size.
        num_layers: Number of stacked GRU layers.
        dropout: Inter-layer dropout in ``[0.0, 1.0)``.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        epochs: Number of training epochs (fixed; no early stopping).
        early_prediction: Whether to train in per-timestep mode.
        timestep_weighting: Early-mode weighting scheme.
        seed: Global RNG seed.
        device: Requested compute device.
        num_workers: DataLoader worker count.
        val_size: Validation carve-out fraction of the train rows.
        window_sizes: Prefix window sizes for training augmentation, or ``None``
            for the full-row baseline.
    """

    emb_dim: int = 16
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 30
    early_prediction: bool = False
    timestep_weighting: str = "none"
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 0
    val_size: float = 0.10
    window_sizes: tuple[int, ...] | None = (64, 128, 256, 512)


def _coerce_window_sizes(value: object) -> tuple[int, ...] | None:
    """Coerce a config ``window_sizes`` value to a validated tuple or ``None``.

    Args:
        value: Raw YAML value (``None``, list, or tuple).

    Returns:
        A non-empty tuple of positive ints, or ``None``.

    Raises:
        ValueError: If the value is malformed (not a list/tuple of positive
            ints, or empty).
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("Could not coerce config field 'window_sizes'")
    if len(value) == 0:
        raise ValueError("Could not coerce config field 'window_sizes'")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("Could not coerce config field 'window_sizes'")
        if item < 1:
            raise ValueError("Could not coerce config field 'window_sizes'")
        out.append(int(item))
    return tuple(out)


def _coerce_scalar(field_name: str, field_type: type, value: object) -> object:
    """Coerce a scalar config value to its dataclass field type.

    Args:
        field_name: The TrainConfig field being coerced.
        field_type: The annotated field type (``int``/``float``/``bool``/``str``).
        value: Raw YAML value.

    Returns:
        The coerced value.

    Raises:
        ValueError: If the value cannot be coerced to the field type.
    """
    try:
        if field_type is bool:
            if not isinstance(value, bool):
                raise ValueError
            return value
        if field_type is int:
            if isinstance(value, bool):
                raise ValueError
            return int(value)  # type: ignore[call-overload]
        if field_type is float:
            if isinstance(value, bool):
                raise ValueError
            return float(value)  # type: ignore[arg-type]
        return str(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not coerce config field '{field_name}'"
        ) from exc


def load_sequence_config(
    path: Path | str = "configs/sequence_config.yaml",
) -> TrainConfig:
    """Load ``TrainConfig`` from YAML, falling back to dataclass defaults.

    If the file does not exist, returns ``TrainConfig()`` with all defaults so
    the experiment is runnable out of the box. If it exists, parses YAML to a
    dict and constructs ``TrainConfig(**known_keys)``.

    Args:
        path: Path to the sequence config YAML.

    Returns:
        A ``TrainConfig`` with file values overlaid on defaults.

    Raises:
        ValueError: If the YAML contains keys not present on ``TrainConfig``, a
            value cannot be coerced to its field type, ``timestep_weighting``
            is invalid, or ``window_sizes`` is malformed.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        return TrainConfig()
    with cfg_path.open() as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return TrainConfig()
    if not isinstance(raw, dict):
        raise ValueError("Sequence config must be a YAML mapping")

    field_types = {f.name: f.type for f in fields(TrainConfig)}
    known = set(field_types)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown sequence config keys: {sorted(unknown)}")

    kwargs: dict[str, object] = {}
    for key, value in raw.items():
        if key == "window_sizes":
            kwargs[key] = _coerce_window_sizes(value)
        elif key == "timestep_weighting":
            scheme = str(value)
            if scheme not in _VALID_WEIGHTING:
                raise ValueError(
                    "Could not coerce config field 'timestep_weighting'"
                )
            kwargs[key] = scheme
        else:
            ftype = _scalar_field_type(field_types[key])
            kwargs[key] = _coerce_scalar(key, ftype, value)
    return TrainConfig(**kwargs)  # type: ignore[arg-type]


def _scalar_field_type(annotation: object) -> type:
    """Map a dataclass field annotation to its concrete scalar type.

    Args:
        annotation: The field annotation (possibly a string under
            ``from __future__ import annotations``).

    Returns:
        One of ``int``, ``float``, ``bool``, or ``str``.
    """
    name = annotation if isinstance(annotation, str) else getattr(
        annotation, "__name__", str(annotation)
    )
    if name == "bool":
        return bool
    if name == "int":
        return int
    if name == "float":
        return float
    return str


# --------------------------------------------------------------------------- #
# §10.1 Data preparation
# --------------------------------------------------------------------------- #
@dataclass
class WindowArrays:
    """A single fixed-width window block of normalized examples.

    Attributes:
        x: Normalized values, shape ``(n_examples, W)`` float32.
        sensor_ids: Sensor IDs, shape ``(n_examples, W)`` int64.
        y: Labels, shape ``(n_examples,)`` int64.
        window_sensor_cols: Retained model sensor names for this block.
    """

    x: np.ndarray
    sensor_ids: np.ndarray
    y: np.ndarray
    window_sensor_cols: list[str]


@dataclass
class PreparedData:
    """Leakage-controlled training/eval inputs for the GRU.

    Attributes:
        train_windows: Per-size train window blocks.
        val_windows: Per-size val window blocks (may be empty).
        x_test_raw: RAW held-out test matrix, shape ``(n_te, n_raw_sensors)``.
        y_test: Held-out test labels, shape ``(n_te,)`` int64.
        raw_sensor_cols: Raw sensor names in fit-time order.
        sensor_cols: Retained model sensor names.
        preprocessor: The train-only fitted preprocessing pipeline.
        pos_weight: Train-only ``n_neg / n_pos``.
    """

    train_windows: list[WindowArrays]
    val_windows: list[WindowArrays]
    x_test_raw: np.ndarray
    y_test: np.ndarray
    raw_sensor_cols: list[str]
    sensor_cols: list[str]
    preprocessor: SequenceSensorPipeline
    pos_weight: float


def _stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified row split, falling back to an empty second split if infeasible.

    Args:
        x: Raw matrix, shape ``(n_rows, n_cols)``.
        y: Labels, shape ``(n_rows,)``.
        test_size: Fraction held out in the second split.
        random_seed: RNG seed.

    Returns:
        ``(x_a, x_b, y_a, y_b)`` where the ``b`` arrays are empty if a stratified
        split is infeasible or ``test_size <= 0``.
    """
    n_rows = x.shape[0]
    feasible = (
        test_size > 0.0
        and int((y == 1).sum()) >= 2
        and int((y == 0).sum()) >= 2
        and round(test_size * n_rows) >= 1
        and round((1.0 - test_size) * n_rows) >= 1
    )
    if not feasible:
        empty_x = np.empty((0, x.shape[1]), dtype=x.dtype)
        empty_y = np.empty((0,), dtype=y.dtype)
        return x, empty_x, y, empty_y
    x_a, x_b, y_a, y_b = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )
    return x_a, x_b, y_a, y_b


def _build_window_blocks(
    preprocessor: SequenceSensorPipeline,
    x_raw: np.ndarray,
    y: np.ndarray,
    raw_sensor_cols: list[str],
    window_sizes: tuple[int, ...] | None,
) -> list[WindowArrays]:
    """Build window blocks for a split using the fitted preprocessor.

    Args:
        preprocessor: Fitted preprocessing pipeline.
        x_raw: RAW matrix for this split, shape ``(n_rows, n_raw_sensors)``.
        y: Labels, shape ``(n_rows,)``.
        raw_sensor_cols: Raw sensor names aligned to ``x_raw``.
        window_sizes: Prefix sizes, or ``None`` for the full-row block.

    Returns:
        A list of ``WindowArrays`` blocks. Empty if ``x_raw`` has no rows.
    """
    if x_raw.shape[0] == 0:
        return []
    sensor_cols = preprocessor.sensor_cols
    n_sensors = len(sensor_cols)

    if window_sizes is None:
        x_norm, ids = preprocessor.transform_full(x_raw, raw_sensor_cols)
        return [
            WindowArrays(
                x=x_norm,
                sensor_ids=ids,
                y=y.astype(np.int64),
                window_sensor_cols=list(sensor_cols),
            )
        ]

    clipped = sorted({min(int(w), n_sensors) for w in window_sizes})
    x_full, ids_full = preprocessor.transform_full(x_raw, raw_sensor_cols)
    blocks: list[WindowArrays] = []
    for k in clipped:
        blocks.append(
            WindowArrays(
                x=x_full[:, :k].copy(),
                sensor_ids=ids_full[:, :k].copy(),
                y=y.astype(np.int64),
                window_sensor_cols=list(sensor_cols[:k]),
            )
        )
    return blocks


def prepare_data(
    raw_dir: Path,
    test_size: float,
    val_size: float,
    random_seed: int,
    missing_threshold: float,
    variance_threshold: float,
    correlation_threshold: float,
    window_sizes: tuple[int, ...] | None = None,
) -> PreparedData:
    """Load raw SECOM data, split rows, fit preprocessing, and build windows.

    The pinned leakage-control order fits the preprocessing pipeline ONLY on the
    train sub-split produced after the validation carve-out (§10.1).

    Args:
        raw_dir: Directory with raw SECOM files.
        test_size: Held-out test fraction of all rows.
        val_size: Validation fraction of the post-test train rows.
        random_seed: Global RNG seed for both splits.
        missing_threshold: Strict upper bound on missing fraction.
        variance_threshold: Strict lower bound on post-impute variance.
        correlation_threshold: Strict upper bound on absolute correlation.
        window_sizes: Prefix sizes, or ``None`` for the full-row baseline.

    Returns:
        A populated ``PreparedData``.

    Raises:
        FileNotFoundError: If raw SECOM files are missing.
        ValueError: For an invalid raw schema, no retained sensors, infinite
            values, failed imputation, or undefined pos_weight.
    """
    raw_df = load_secom(raw_dir)
    validate_secom(raw_df)

    raw_sensor_cols = [c for c in raw_df.columns if str(c).startswith("sensor_")]
    if not raw_sensor_cols:
        raise ValueError("No raw sensor_ columns found")

    x_raw = raw_df[raw_sensor_cols].to_numpy(dtype=np.float32)
    y = raw_df["label"].to_numpy(dtype=np.int64)

    x_train_full, x_test, y_train_full, y_test = _stratified_split(
        x_raw, y, test_size, random_seed
    )
    x_train, x_val, y_train, y_val = _stratified_split(
        x_train_full, y_train_full, val_size, random_seed
    )

    preprocessor = SequenceSensorPipeline.fit(
        x_train,
        raw_sensor_cols,
        missing_threshold,
        variance_threshold,
        correlation_threshold,
    )
    sensor_cols = preprocessor.sensor_cols
    pos_weight = compute_pos_weight(y_train)

    train_windows = _build_window_blocks(
        preprocessor, x_train, y_train, raw_sensor_cols, window_sizes
    )
    val_windows = _build_window_blocks(
        preprocessor, x_val, y_val, raw_sensor_cols, window_sizes
    )

    return PreparedData(
        train_windows=train_windows,
        val_windows=val_windows,
        x_test_raw=x_test,
        y_test=y_test,
        raw_sensor_cols=list(raw_sensor_cols),
        sensor_cols=list(sensor_cols),
        preprocessor=preprocessor,
        pos_weight=pos_weight,
    )


# --------------------------------------------------------------------------- #
# §10.2 Training
# --------------------------------------------------------------------------- #
def resolve_device(requested: str) -> str:
    """Resolve a device string.

    Args:
        requested: One of ``"cpu"``, ``"auto"``, or ``"cuda"``.

    Returns:
        ``"cpu"`` or ``"cuda"``.

    Raises:
        ValueError: If ``"cuda"`` is requested but unavailable.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise ValueError("CUDA requested but not available")
    raise ValueError(f"Unknown device: {requested}")


def set_global_determinism(seed: int) -> None:
    """Seed all RNGs and enable deterministic algorithms.

    Args:
        seed: Global RNG seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _block_weights(
    block: WindowArrays, config: TrainConfig, device: str
) -> Tensor | None:
    """Build the sum-to-1 timestep weights for one window block (early mode only).

    Args:
        block: The window block whose fixed width sets the weight length.
        config: Hyperparameters (mode and weighting scheme).
        device: Compute device for the returned tensor.

    Returns:
        A ``(W,)`` float32 weights tensor in early mode, or ``None`` in standard
        mode (where timestep weighting does not apply).
    """
    if not config.early_prediction:
        return None
    return build_timestep_weights(
        block.x.shape[1], config.timestep_weighting, device
    )


def _block_loss(
    model: SecomGRU,
    x: Tensor,
    sensor_ids: Tensor,
    y: Tensor,
    pos_weight_tensor: Tensor,
    early_prediction: bool,
    timestep_weights: Tensor | None,
) -> Tensor:
    """Compute the per-batch loss for either mode (§7).

    Args:
        model: The GRU model.
        x: Normalized values, shape ``(B, W)``.
        sensor_ids: Sensor IDs, shape ``(B, W)``.
        y: Float labels, shape ``(B,)``.
        pos_weight_tensor: ``pos_weight`` tensor, shape ``(1,)``.
        early_prediction: Whether to use early-mode loss.
        timestep_weights: Precomputed sum-to-1 timestep weights, shape ``(W,)``;
            required in early mode, ignored in standard mode.

    Returns:
        A scalar loss tensor.
    """
    if not early_prediction:
        logits = model(x, sensor_ids)
        return F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight_tensor
        )
    if timestep_weights is None:
        raise ValueError("timestep_weights are required in early-prediction mode")
    logits = model(x, sensor_ids)
    window = logits.shape[1]
    targets = y.unsqueeze(1).expand(logits.shape[0], window)
    elem = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight_tensor, reduction="none"
    )
    per_sample = (elem * timestep_weights.unsqueeze(0)).sum(dim=1)
    return per_sample.mean()


def train(
    data: PreparedData, config: TrainConfig
) -> tuple[SecomGRU, list[dict[str, float]]]:
    """Train a ``SecomGRU`` over fixed epochs; return the model and history.

    Args:
        data: Prepared leakage-free training data.
        config: Hyperparameters.

    Returns:
        A tuple ``(model, history)`` where ``model`` is in ``eval()`` mode and
        ``history`` is a per-epoch list of
        ``{"epoch", "train_loss", "val_loss", "val_pr_auc"}`` dicts.

    Raises:
        ValueError: From ``resolve_device`` on unavailable CUDA.
    """
    set_global_determinism(config.seed)
    device = resolve_device(config.device)

    n_sensors = len(data.sensor_cols)
    model = build_gru(
        n_sensors=n_sensors,
        emb_dim=config.emb_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        early_prediction=config.early_prediction,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    pos_weight_tensor = torch.tensor(
        [data.pos_weight], dtype=torch.float32, device=device
    )

    train_loaders = [
        make_loader(
            SecomSequenceDataset(block.x, block.sensor_ids, block.y),
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed + block_idx,
            num_workers=config.num_workers,
        )
        for block_idx, block in enumerate(data.train_windows)
    ]
    # Timestep weights depend only on the (fixed) window width of each block, so
    # build them once per block rather than rebuilding inside the batch loop.
    train_weights = [
        _block_weights(block, config, device) for block in data.train_windows
    ]

    if not data.val_windows:
        print("val split empty; skipping validation", file=sys.stderr)

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for loader, weights in zip(train_loaders, train_weights, strict=True):
            for x, sensor_ids, y in loader:
                x = x.to(device)
                sensor_ids = sensor_ids.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                loss = _block_loss(
                    model,
                    x,
                    sensor_ids,
                    y,
                    pos_weight_tensor,
                    config.early_prediction,
                    weights,
                )
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                total_loss += float(loss.detach())
                n_batches += 1
        train_loss = total_loss / n_batches if n_batches > 0 else math.nan

        val_loss, val_pr_auc = _evaluate_val(
            model, data, config, pos_weight_tensor, device
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
            }
        )

    model.eval()
    return model, history


def _evaluate_val(
    model: SecomGRU,
    data: PreparedData,
    config: TrainConfig,
    pos_weight_tensor: Tensor,
    device: str,
) -> tuple[float, float]:
    """Compute aggregate validation loss and PR-AUC for one epoch.

    Args:
        model: The GRU model.
        data: Prepared data with val window blocks.
        config: Hyperparameters.
        pos_weight_tensor: ``pos_weight`` tensor, shape ``(1,)``.
        device: Compute device.

    Returns:
        ``(val_loss, val_pr_auc)``; both ``nan`` if val is empty or single-class.
    """
    if not data.val_windows:
        return math.nan, math.nan

    model.eval()
    total_loss = 0.0
    n_batches = 0
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for block in data.val_windows:
            weights = _block_weights(block, config, device)
            loader = make_loader(
                SecomSequenceDataset(block.x, block.sensor_ids, block.y),
                batch_size=config.batch_size,
                shuffle=False,
                seed=config.seed,
                num_workers=config.num_workers,
            )
            for x, sensor_ids, y in loader:
                x = x.to(device)
                sensor_ids = sensor_ids.to(device)
                y = y.to(device)
                loss = _block_loss(
                    model,
                    x,
                    sensor_ids,
                    y,
                    pos_weight_tensor,
                    config.early_prediction,
                    weights,
                )
                total_loss += float(loss.detach())
                n_batches += 1
                logits = predict_logits(model, x, sensor_ids)
                probs.append(torch.sigmoid(logits).cpu().numpy().astype(np.float64))
                labels.append(y.cpu().numpy().astype(np.int64))

    val_loss = total_loss / n_batches if n_batches > 0 else math.nan
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    if len(np.unique(y_true)) < 2:
        return val_loss, math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = compute_sequence_metrics(y_true, y_prob)
    return val_loss, metrics.pr_auc


# --------------------------------------------------------------------------- #
# §6 Checkpoint
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: Path,
    model: SecomGRU,
    preprocessor: SequenceSensorPipeline,
    sensor_cols: list[str],
    pos_weight: float,
    timestep_weighting: str,
    random_seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> None:
    """Serialize the model and preprocessing pipeline for leakage-free eval.

    Args:
        path: Output checkpoint path.
        model: Trained ``SecomGRU``.
        preprocessor: Fitted preprocessing pipeline.
        sensor_cols: Ordered retained model sensor names.
        pos_weight: Train-only ``n_neg / n_pos``.
        timestep_weighting: Early-mode weighting scheme (persisted always).
        random_seed: Global seed used for training.
        epochs: Epochs actually trained.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        device: Device string used for training.

    Raises:
        ValueError: If ``len(preprocessor.cleaner.medians)``,
            ``len(preprocessor.mean)``, ``len(preprocessor.scale)``, or
            ``len(sensor_cols)`` do not equal ``model.n_sensors``.
    """
    n = model.n_sensors
    if len(preprocessor.cleaner.medians) != n:
        raise ValueError(
            f"medians length {len(preprocessor.cleaner.medians)} != n_sensors {n}"
        )
    if len(preprocessor.mean) != n:
        raise ValueError(f"mean length {len(preprocessor.mean)} != n_sensors {n}")
    if len(preprocessor.scale) != n:
        raise ValueError(f"scale length {len(preprocessor.scale)} != n_sensors {n}")
    if len(sensor_cols) != n:
        raise ValueError(
            f"sensor_cols length {len(sensor_cols)} != n_sensors {n}"
        )

    ckpt: dict[str, object] = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "n_sensors": int(model.n_sensors),
        "emb_dim": int(model.emb_dim),
        "hidden_size": int(model.hidden_size),
        "num_layers": int(model.num_layers),
        "dropout": float(model.dropout),
        "early_prediction": bool(model.early_prediction),
        "timestep_weighting": str(timestep_weighting),
        "preprocessor": preprocessor.to_dict(),
        "sensor_cols": list(sensor_cols),
        "pos_weight": float(pos_weight),
        "random_seed": int(random_seed),
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }
    torch.save(ckpt, path)


@dataclass
class LoadedCheckpoint:
    """A reconstructed checkpoint ready for inference.

    Attributes:
        model: ``SecomGRU`` in ``eval()`` mode with restored weights.
        preprocessor: The training-fitted preprocessing pipeline.
        sensor_cols: Ordered retained model sensor names.
        pos_weight: Train-only ``n_neg / n_pos``.
        early_prediction: Whether the model was trained in early mode.
        timestep_weighting: Persisted early-mode weighting scheme.
        raw: The full raw checkpoint dict.
    """

    model: SecomGRU
    preprocessor: SequenceSensorPipeline
    sensor_cols: list[str]
    pos_weight: float
    early_prediction: bool
    timestep_weighting: str
    raw: dict[str, object]


_REQUIRED_CKPT_KEYS = (
    "format_version",
    "model_state_dict",
    "n_sensors",
    "emb_dim",
    "hidden_size",
    "num_layers",
    "dropout",
    "early_prediction",
    "timestep_weighting",
    "preprocessor",
    "sensor_cols",
    "pos_weight",
    "random_seed",
    "epochs",
    "lr",
    "batch_size",
    "torch_version",
    "cuda_available",
    "device",
    "created_at",
)


def load_checkpoint(path: Path, device: str = "cpu") -> LoadedCheckpoint:
    """Load a checkpoint, rebuild the model, and restore weights and stats.

    Args:
        path: Checkpoint path.
        device: Device to map tensors to and place the model on.

    Returns:
        A populated ``LoadedCheckpoint`` with the model in ``eval()`` mode.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If a required schema key is missing.
        ValueError: If ``format_version != 1`` or preprocessor metadata is
            invalid.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt: dict[str, object] = torch.load(
        path, map_location=device, weights_only=False
    )
    for key in _REQUIRED_CKPT_KEYS:
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing required key: {key}")

    version = ckpt["format_version"]
    if version != 1:
        raise ValueError(f"Unsupported checkpoint format_version: {version}")

    model = build_gru(
        n_sensors=int(ckpt["n_sensors"]),  # type: ignore[call-overload]
        emb_dim=int(ckpt["emb_dim"]),  # type: ignore[call-overload]
        hidden_size=int(ckpt["hidden_size"]),  # type: ignore[call-overload]
        num_layers=int(ckpt["num_layers"]),  # type: ignore[call-overload]
        dropout=float(ckpt["dropout"]),  # type: ignore[arg-type]
        early_prediction=bool(ckpt["early_prediction"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])  # type: ignore[arg-type]
    model.to(device)
    model.eval()

    preprocessor = SequenceSensorPipeline.from_dict(
        ckpt["preprocessor"]  # type: ignore[arg-type]
    )
    return LoadedCheckpoint(
        model=model,
        preprocessor=preprocessor,
        sensor_cols=_as_str_list(ckpt["sensor_cols"]),
        pos_weight=float(ckpt["pos_weight"]),  # type: ignore[arg-type]
        early_prediction=bool(ckpt["early_prediction"]),
        timestep_weighting=str(ckpt["timestep_weighting"]),
        raw=ckpt,
    )


# --------------------------------------------------------------------------- #
# §11 Metrics
# --------------------------------------------------------------------------- #
@dataclass
class SequenceMetrics:
    """The full GRU metric set with a single-class guard.

    Attributes:
        roc_auc: ROC-AUC; ``nan`` if ``y_true`` is single-class.
        pr_auc: PR-AUC; ``nan`` if ``y_true`` is single-class.
        precision: Precision at the threshold.
        recall: Recall at the threshold.
        f1: F1 at the threshold.
        balanced_accuracy: Balanced accuracy at the threshold.
        confusion_matrix: 2x2 matrix ``[[TN, FP], [FN, TP]]``.
        threshold: Decision threshold used.
        n_pos: Number of positive ground-truth labels.
        n_neg: Number of negative ground-truth labels.
    """

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    confusion_matrix: list[list[int]]
    threshold: float
    n_pos: int
    n_neg: int


def compute_sequence_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> SequenceMetrics:
    """Compute the full GRU metric set with a single-class guard.

    Args:
        y_true: Ground-truth binary labels, shape ``(n,)``.
        y_prob: Predicted positive-class probabilities, shape ``(n,)``.
        threshold: Decision threshold.

    Returns:
        A populated ``SequenceMetrics``. ``roc_auc``/``pr_auc`` are ``nan`` (with
        a warning) when ``y_true`` has fewer than two distinct classes.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    if len(np.unique(y_true)) < 2:
        warnings.warn("y_true single-class; ROC/PR-AUC undefined")
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        base = compute_metrics(y_true, y_prob, threshold)
        roc_auc = base.roc_auc
        pr_auc = base.pr_auc

    return SequenceMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        threshold=float(threshold),
        n_pos=n_pos,
        n_neg=n_neg,
    )


def _json_safe(value: object) -> object:
    """Map ``nan`` floats to ``None`` for JSON serialization.

    Args:
        value: A scalar value.

    Returns:
        ``None`` if ``value`` is a ``nan`` float, otherwise ``value``.
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_sequence_metrics(metrics: SequenceMetrics, path: Path) -> None:
    """Write the metrics dataclass as JSON; ``nan`` serialized as JSON null.

    Args:
        metrics: Computed sequence metrics.
        path: Output JSON path.
    """
    payload = {
        "roc_auc": _json_safe(metrics.roc_auc),
        "pr_auc": _json_safe(metrics.pr_auc),
        "precision": _json_safe(metrics.precision),
        "recall": _json_safe(metrics.recall),
        "f1": _json_safe(metrics.f1),
        "balanced_accuracy": _json_safe(metrics.balanced_accuracy),
        "confusion_matrix": metrics.confusion_matrix,
        "threshold": _json_safe(metrics.threshold),
        "n_pos": metrics.n_pos,
        "n_neg": metrics.n_neg,
    }
    Path(path).write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------- #
# §10.3 Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    checkpoint: LoadedCheckpoint,
    x_window_raw: np.ndarray,
    y_true: np.ndarray,
    window_sensor_cols: list[str],
    threshold: float = 0.5,
    device: str = "cpu",
) -> tuple[np.ndarray, SequenceMetrics]:
    """Compute probabilities and metrics for a raw sensor window.

    Args:
        checkpoint: A loaded checkpoint with the training-fitted preprocessor.
        x_window_raw: RAW window matrix, shape ``(n_rows, raw_window_size)``.
        y_true: Ground-truth labels, shape ``(n_rows,)``.
        window_sensor_cols: Raw names for each supplied column.
        threshold: Decision threshold.
        device: Compute device.

    Returns:
        ``(y_prob, metrics)`` where ``y_prob`` is float64 shape ``(n_rows,)``.

    Raises:
        ValueError: On unknown window sensor columns, a shape mismatch, or a
            full-row raw schema mismatch.
    """
    preprocessor = checkpoint.preprocessor
    if list(window_sensor_cols) == list(preprocessor.raw_sensor_cols):
        x_norm, sensor_ids = preprocessor.transform_full(
            x_window_raw, window_sensor_cols
        )
    elif _is_full_row_attempt(window_sensor_cols, preprocessor.raw_sensor_cols):
        raise ValueError(
            f"Raw sensor columns mismatch: checkpoint has "
            f"{len(preprocessor.raw_sensor_cols)}, eval data has "
            f"{len(window_sensor_cols)}"
        )
    else:
        x_norm, sensor_ids = preprocessor.transform_window(
            x_window_raw, window_sensor_cols
        )

    model = checkpoint.model.to(device)
    model.eval()
    x_t = torch.from_numpy(np.ascontiguousarray(x_norm, dtype=np.float32)).to(device)
    ids_t = torch.from_numpy(
        np.ascontiguousarray(sensor_ids, dtype=np.int64)
    ).to(device)
    with torch.no_grad():
        logits = predict_logits(model, x_t, ids_t)
        y_prob = torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    metrics = compute_sequence_metrics(y_true, y_prob, threshold)
    return y_prob, metrics


def _is_full_row_attempt(
    window_sensor_cols: list[str], raw_sensor_cols: list[str]
) -> bool:
    """Decide whether a window is a full-row attempt that mismatches the schema.

    A full-row attempt is one whose column set differs from the fitted raw
    schema only by membership/order while covering (nearly) the full row, so the
    §8 raw-mismatch error is the correct signal rather than a window error.

    Args:
        window_sensor_cols: Supplied raw window names.
        raw_sensor_cols: Fitted raw schema names.

    Returns:
        ``True`` if the lists have equal length but differ in membership/order.
    """
    return len(window_sensor_cols) == len(raw_sensor_cols) and list(
        window_sensor_cols
    ) != list(raw_sensor_cols)


# --------------------------------------------------------------------------- #
# §12 Baseline comparison
# --------------------------------------------------------------------------- #
@dataclass
class TabularBaseline:
    """The existing best tabular baseline metrics.

    Attributes:
        source: ``"model_comparison.json"`` or ``"model_metadata.json"``.
        model: Baseline model name/version, if known.
        test_pr_auc: Test PR-AUC.
        test_roc_auc: Test ROC-AUC.
        test_recall: Test recall.
        test_precision: Test precision.
    """

    source: str
    model: str | None
    test_pr_auc: float | None
    test_roc_auc: float | None
    test_recall: float | None
    test_precision: float | None


def _as_number(value: object) -> float | None:
    """Coerce a JSON value to a float, returning ``None`` on failure.

    Args:
        value: Candidate value.

    Returns:
        The float value, or ``None`` if not numeric.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load_from_comparison(reports_dir: Path) -> TabularBaseline | None:
    """Load the selected baseline row from ``model_comparison.json``.

    Args:
        reports_dir: Directory holding ``model_comparison.json``.

    Returns:
        A ``TabularBaseline`` or ``None`` if absent/malformed/no-unique-selected.
    """
    path = reports_dir / "model_comparison.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, list):
        print(f"warning: {path} is not a JSON list", file=sys.stderr)
        return None
    selected = [r for r in data if isinstance(r, dict) and r.get("selected") is True]
    if len(selected) != 1:
        print(
            f"warning: {path} has {len(selected)} selected rows (need exactly 1)",
            file=sys.stderr,
        )
        return None
    row = selected[0]
    pr_auc = _as_number(row.get("test_pr_auc"))
    roc_auc = _as_number(row.get("test_roc_auc"))
    recall = _as_number(row.get("test_recall"))
    precision = _as_number(row.get("test_precision"))
    if None in (pr_auc, roc_auc, recall, precision):
        print(f"warning: {path} selected row missing fields", file=sys.stderr)
        return None
    model = row.get("model")
    return TabularBaseline(
        source="model_comparison.json",
        model=str(model) if model is not None else None,
        test_pr_auc=pr_auc,
        test_roc_auc=roc_auc,
        test_recall=recall,
        test_precision=precision,
    )


def _load_from_metadata(models_dir: Path) -> TabularBaseline | None:
    """Load the baseline from ``model_metadata.json`` as a fallback.

    Args:
        models_dir: Directory holding ``model_metadata.json``.

    Returns:
        A ``TabularBaseline`` or ``None`` if absent/malformed/missing-fields.
    """
    path = models_dir / "model_metadata.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"warning: {path} is not a JSON object", file=sys.stderr)
        return None
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        print(f"warning: {path} missing metrics object", file=sys.stderr)
        return None
    pr_auc = _as_number(metrics.get("pr_auc"))
    roc_auc = _as_number(metrics.get("roc_auc"))
    recall = _as_number(metrics.get("recall"))
    precision = _as_number(metrics.get("precision"))
    if None in (pr_auc, roc_auc, recall, precision):
        print(f"warning: {path} metrics missing fields", file=sys.stderr)
        return None
    model = data.get("model_version")
    return TabularBaseline(
        source="model_metadata.json",
        model=str(model) if model is not None else None,
        test_pr_auc=pr_auc,
        test_roc_auc=roc_auc,
        test_recall=recall,
        test_precision=precision,
    )


def load_tabular_baseline(
    reports_dir: Path, models_dir: Path
) -> TabularBaseline | None:
    """Load the existing best tabular baseline, gracefully handling absence.

    Resolution order (first success wins): ``model_comparison.json`` (selected
    row), then ``model_metadata.json``.

    Args:
        reports_dir: Directory holding ``model_comparison.json``.
        models_dir: Directory holding ``model_metadata.json``.

    Returns:
        A ``TabularBaseline`` or ``None`` if neither source is usable. Never
        raises; warns to stderr on any problem.
    """
    primary = _load_from_comparison(reports_dir)
    if primary is not None:
        return primary
    return _load_from_metadata(models_dir)


def _delta(seq: float, base: float | None) -> float | None:
    """Compute ``seq - base`` guarding against ``None``/``nan``.

    Args:
        seq: Sequence-model metric value.
        base: Baseline metric value, or ``None``.

    Returns:
        The delta, or ``None`` if undefined.
    """
    if base is None or math.isnan(seq) or math.isnan(base):
        return None
    return seq - base


def compute_verdict(
    seq_metrics: SequenceMetrics, baseline: TabularBaseline | None
) -> str:
    """Return the PR-AUC comparison verdict (§12).

    Single source of truth for the verdict so the written artifact and any
    printed summary cannot diverge.

    Args:
        seq_metrics: Computed sequence metrics.
        baseline: Loaded tabular baseline, or ``None``.

    Returns:
        One of ``"sequence_better"``, ``"baseline_better"``, ``"tie"``, or
        ``"no_baseline"``.
    """
    if baseline is None or math.isnan(seq_metrics.pr_auc):
        return "no_baseline"
    delta_pr = _delta(seq_metrics.pr_auc, baseline.test_pr_auc)
    if delta_pr is None:
        return "no_baseline"
    if abs(delta_pr) < 1e-6:
        return "tie"
    return "sequence_better" if delta_pr > 0 else "baseline_better"


def write_baseline_comparison(
    seq_metrics: SequenceMetrics,
    baseline: TabularBaseline | None,
    reports_dir: Path,
) -> None:
    """Write ``sequence_model_comparison.{json,csv}`` with delta and verdict.

    Args:
        seq_metrics: Computed sequence metrics.
        baseline: Loaded tabular baseline, or ``None``.
        reports_dir: Directory for the comparison artifacts.
    """
    delta_pr = _delta(seq_metrics.pr_auc, baseline.test_pr_auc if baseline else None)
    delta_roc = _delta(
        seq_metrics.roc_auc, baseline.test_roc_auc if baseline else None
    )
    verdict = compute_verdict(seq_metrics, baseline)

    sequence_block = {
        "pr_auc": _json_safe(seq_metrics.pr_auc),
        "roc_auc": _json_safe(seq_metrics.roc_auc),
        "recall": _json_safe(seq_metrics.recall),
        "precision": _json_safe(seq_metrics.precision),
        "balanced_accuracy": _json_safe(seq_metrics.balanced_accuracy),
        "threshold": _json_safe(seq_metrics.threshold),
        "n_pos": seq_metrics.n_pos,
        "n_neg": seq_metrics.n_neg,
    }
    baseline_block: dict[str, object] | None
    if baseline is None:
        baseline_block = None
    else:
        baseline_block = {
            "source": baseline.source,
            "model": baseline.model,
            "pr_auc": _json_safe(baseline.test_pr_auc),
            "roc_auc": _json_safe(baseline.test_roc_auc),
            "recall": _json_safe(baseline.test_recall),
            "precision": _json_safe(baseline.test_precision),
        }

    payload = {
        "sequence": sequence_block,
        "baseline": baseline_block,
        "delta_pr_auc": _json_safe(delta_pr) if delta_pr is not None else None,
        "delta_roc_auc": _json_safe(delta_roc) if delta_roc is not None else None,
        "verdict": verdict,
    }
    json_path = reports_dir / "sequence_model_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = reports_dir / "sequence_model_comparison.csv"
    _write_comparison_csv(
        csv_path, sequence_block, baseline, delta_pr, delta_roc, verdict
    )


def _write_comparison_csv(
    path: Path,
    sequence_block: dict[str, object],
    baseline: TabularBaseline | None,
    delta_pr: float | None,
    delta_roc: float | None,
    verdict: str,
) -> None:
    """Write the flattened single-row comparison CSV.

    Args:
        path: Output CSV path.
        sequence_block: The sequence metric block.
        baseline: Loaded baseline, or ``None``.
        delta_pr: PR-AUC delta, or ``None``.
        delta_roc: ROC-AUC delta, or ``None``.
        verdict: Comparison verdict.
    """
    header = [
        "sequence_pr_auc",
        "sequence_roc_auc",
        "sequence_recall",
        "sequence_precision",
        "sequence_balanced_accuracy",
        "sequence_threshold",
        "sequence_n_pos",
        "sequence_n_neg",
        "baseline_source",
        "baseline_model",
        "baseline_pr_auc",
        "baseline_roc_auc",
        "baseline_recall",
        "baseline_precision",
        "delta_pr_auc",
        "delta_roc_auc",
        "verdict",
    ]

    def _cell(value: object) -> object:
        return "" if value is None else value

    row = [
        _cell(sequence_block["pr_auc"]),
        _cell(sequence_block["roc_auc"]),
        _cell(sequence_block["recall"]),
        _cell(sequence_block["precision"]),
        _cell(sequence_block["balanced_accuracy"]),
        _cell(sequence_block["threshold"]),
        _cell(sequence_block["n_pos"]),
        _cell(sequence_block["n_neg"]),
        _cell(baseline.source if baseline else None),
        _cell(baseline.model if baseline else None),
        _cell(_json_safe(baseline.test_pr_auc) if baseline else None),
        _cell(_json_safe(baseline.test_roc_auc) if baseline else None),
        _cell(_json_safe(baseline.test_recall) if baseline else None),
        _cell(_json_safe(baseline.test_precision) if baseline else None),
        _cell(delta_pr),
        _cell(delta_roc),
        verdict,
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
