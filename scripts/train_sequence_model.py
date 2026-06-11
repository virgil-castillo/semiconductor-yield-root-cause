"""Train the experimental GRU sequence model on raw SECOM data.

Loads ``TrainConfig`` from a YAML config file (defaulting to
``configs/sequence_config.yaml``), applies any CLI overrides, then runs the
leakage-free data-preparation and fixed-epoch training loop. Writes the
checkpoint and the per-epoch training history JSON.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from yield_risk.config import load_config
from yield_risk.sequence_train import (
    TrainConfig,
    load_sequence_config,
    prepare_data,
    save_checkpoint,
    train,
)


def _parse_window_sizes(value: str) -> tuple[int, ...]:
    """Parse a comma-separated string of positive ints into a tuple.

    Args:
        value: Comma-separated positive integers, e.g. ``"64,128,256"``.

    Returns:
        A non-empty tuple of positive ints.

    Raises:
        argparse.ArgumentTypeError: If any token is not a positive integer.
    """
    tokens = value.split(",")
    result: list[int] = []
    for tok in tokens:
        tok = tok.strip()
        try:
            n = int(tok)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"window-sizes: {tok!r} is not an integer"
            ) from exc
        if n < 1:
            raise argparse.ArgumentTypeError(
                f"window-sizes: each value must be a positive integer, got {n}"
            )
        result.append(n)
    if not result:
        raise argparse.ArgumentTypeError(
            "window-sizes: must provide at least one value"
        )
    return tuple(result)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training script.

    Returns:
        A configured ``ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description="Train the experimental GRU sequence model on raw SECOM data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sequence_config.yaml"),
        help="Path to sequence config YAML (missing -> dataclass defaults).",
    )
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=None,
        metavar="N",
        help="Sensor-ID embedding dimension.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        metavar="N",
        help="GRU hidden size.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        metavar="N",
        help="Number of stacked GRU layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        metavar="F",
        help="Inter-layer dropout in [0.0, 1.0).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="F",
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Number of training epochs (fixed; no early stopping).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Global RNG seed.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Compute device.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=None,
        metavar="F",
        help="Validation carve-out fraction of train rows.",
    )
    parser.add_argument(
        "--window-sizes",
        type=_parse_window_sizes,
        default=None,
        metavar="N[,N...]",
        help="Comma-separated prefix window sizes for training augmentation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Output checkpoint path (default: models_dir/sequence_gru.pt).",
    )
    return parser


def _fmt(value: float) -> str:
    """Format a possibly-``nan`` metric to 4 decimals.

    Args:
        value: A metric value (``nan`` when validation was skipped).

    Returns:
        ``"nan"`` if ``value`` is ``nan``, else the value to 4 decimal places.
    """
    return "nan" if value != value else f"{value:.4f}"


def _apply_overrides(
    config: TrainConfig, args: argparse.Namespace
) -> TrainConfig:
    """Apply non-None CLI values to the loaded config via ``dataclasses.replace``.

    Args:
        config: Loaded ``TrainConfig`` from the YAML file.
        args: Parsed namespace with optional CLI overrides.

    Returns:
        A new ``TrainConfig`` with CLI overrides applied.
    """
    overrides: dict[str, object] = {}
    if args.emb_dim is not None:
        overrides["emb_dim"] = args.emb_dim
    if args.hidden_size is not None:
        overrides["hidden_size"] = args.hidden_size
    if args.num_layers is not None:
        overrides["num_layers"] = args.num_layers
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.val_size is not None:
        overrides["val_size"] = args.val_size
    if args.window_sizes is not None:
        overrides["window_sizes"] = args.window_sizes
    if not overrides:
        return config
    return dataclasses.replace(config, **overrides)  # type: ignore[arg-type]


def main() -> None:
    """Entry point: train the GRU sequence model and save artifacts.

    Raises:
        SystemExit: With code 1 on ``FileNotFoundError``, ``ValueError``, or
            ``KeyError``; exits 0 on success.
    """
    parser = _build_parser()
    args = parser.parse_args()

    try:
        cfg = load_config()
        seq_config = load_sequence_config(args.config)
        seq_config = _apply_overrides(seq_config, args)

        checkpoint_path: Path = (
            args.checkpoint
            if args.checkpoint is not None
            else cfg.paths.models_dir / "sequence_gru.pt"
        )

        print("Preparing data...")
        data = prepare_data(
            raw_dir=cfg.paths.raw_dir,
            test_size=cfg.run.test_size,
            val_size=seq_config.val_size,
            random_seed=seq_config.seed,
            missing_threshold=cfg.run.missing_threshold,
            variance_threshold=cfg.run.variance_threshold,
            correlation_threshold=cfg.run.correlation_threshold,
            window_sizes=seq_config.window_sizes,
        )

        print(
            f"Training GRU: {len(data.sensor_cols)} retained sensors, "
            f"{seq_config.epochs} epochs, device={seq_config.device}"
        )
        model, history = train(data, seq_config)

        cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            preprocessor=data.preprocessor,
            sensor_cols=data.sensor_cols,
            pos_weight=data.pos_weight,
            random_seed=seq_config.seed,
            epochs=seq_config.epochs,
            lr=seq_config.lr,
            batch_size=seq_config.batch_size,
            device=seq_config.device,
        )

        cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        history_path = cfg.paths.reports_dir / "sequence_train_history.json"
        history_path.write_text(json.dumps(history, indent=2))

        if history:
            last = history[-1]
            print(
                f"Final epoch {int(last['epoch'])}: "
                f"train_loss={_fmt(last['train_loss'])}  "
                f"val_loss={_fmt(last['val_loss'])}  "
                f"val_pr_auc={_fmt(last['val_pr_auc'])}"
            )
        else:
            print("No training epochs ran (epochs=0).")

        print(f"Checkpoint written to: {checkpoint_path}")

    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
