"""Run the GRU hyperparameter sweep on raw SECOM data.

Builds the full 108-trial sweep grid from a base ``TrainConfig``, runs every
trial in order, ranks results, writes ``sequence_sweep_results.json`` and
``sequence_sweep_results.csv``, and optionally auto-evaluates the winning
checkpoint with the existing evaluation script.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from yield_risk.config import load_config
from yield_risk.sequence_sweep import build_sweep_grid, run_sweep
from yield_risk.sequence_train import TrainConfig, load_sequence_config, prepare_data


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the sweep script.

    Returns:
        A configured ``ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description="Run the GRU hyperparameter sweep on raw SECOM data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sequence_config.yaml"),
        help="Path to sequence config YAML (missing -> dataclass defaults).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Override number of training epochs for every trial.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Override compute device.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="Skip auto-evaluation of the winning checkpoint after the sweep.",
    )
    return parser


def main() -> None:
    """Entry point: run the full sweep and print a summary.

    Raises:
        SystemExit: With code 1 on ``FileNotFoundError``, ``ValueError``, or
            ``KeyError``; exits 0 on success.
    """
    parser = _build_parser()
    args = parser.parse_args()

    try:
        cfg = load_config()
        base: TrainConfig = load_sequence_config(args.config)

        overrides: dict[str, object] = {}
        if args.epochs is not None:
            overrides["epochs"] = args.epochs
        if args.device is not None:
            overrides["device"] = args.device
        if overrides:
            base = dataclasses.replace(base, **overrides)  # type: ignore[arg-type]

        resolved_device: str = base.device

        print("Preparing data...")
        data = prepare_data(
            raw_dir=cfg.paths.raw_dir,
            test_size=cfg.run.test_size,
            val_size=base.val_size,
            random_seed=base.seed,
            missing_threshold=cfg.run.missing_threshold,
            variance_threshold=cfg.run.variance_threshold,
            correlation_threshold=cfg.run.correlation_threshold,
            window_sizes=base.window_sizes,
        )

        specs = build_sweep_grid(base)
        print(f"Running {len(specs)} trials...")

        cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
        cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)

        ranked, best = run_sweep(
            data,
            specs,
            cfg.paths.models_dir,
            cfg.paths.reports_dir,
            device=resolved_device,
            evaluate_best=not args.no_eval,
        )

        n_total = len(ranked)
        n_failed = sum(1 for r in ranked if r.status == "failed")

        print()
        print(f"Sweep complete: {n_total} trials, {n_failed} failed.")
        if best is None:
            print("No successful trials produced.")
        else:
            print(
                f"Best trial: {best.trial_id}  "
                f"val_pr_auc={best.val_pr_auc:.4f}  "
                f"checkpoint={best.checkpoint_path}"
            )

    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
