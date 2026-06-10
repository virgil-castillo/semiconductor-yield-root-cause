"""Evaluate the experimental GRU sequence model on the held-out test split.

Loads the checkpoint written by ``train_sequence_model.py``, reproduces the
identical held-out test split (same raw stratified split used during training),
scores the test wafers through the checkpoint's preprocessing pipeline, and
writes metrics, figures, and a baseline comparison report.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from yield_risk.config import load_config
from yield_risk.data import load_secom
from yield_risk.evaluate import (
    ClassificationMetrics,
    format_report,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from yield_risk.sequence_train import (
    LoadedCheckpoint,
    SequenceMetrics,
    _stratified_split,
    compute_verdict,
    evaluate,
    load_checkpoint,
    load_tabular_baseline,
    save_sequence_metrics,
    write_baseline_comparison,
)
from yield_risk.validation import validate_secom


def _parse_window_sensors(value: str) -> list[str]:
    """Parse a comma-separated list of raw sensor names.

    Args:
        value: Comma-separated sensor names, e.g. ``"sensor_001,sensor_002"``.

    Returns:
        A non-empty list of stripped sensor name strings.

    Raises:
        argparse.ArgumentTypeError: If the list is empty after splitting.
    """
    names = [s.strip() for s in value.split(",") if s.strip()]
    if not names:
        raise argparse.ArgumentTypeError(
            "--window-sensors: must provide at least one sensor name"
        )
    return names


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluation script.

    Returns:
        A configured ``ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the GRU sequence model on the held-out test split.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: models_dir/sequence_gru.pt).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="F",
        help="Decision threshold for thresholded metrics (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Compute device (default: cpu).",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help=(
            "Metrics JSON output path "
            "(default: reports_dir/sequence_model_metrics.json)."
        ),
    )
    parser.add_argument(
        "--comparison-out",
        type=Path,
        default=None,
        help=(
            "Comparison JSON output path "
            "(default: reports_dir/sequence_model_comparison.json). "
            "The CSV sibling is written automatically alongside it."
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory for figure PNGs (default: cfg.paths.figures_dir).",
    )
    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument(
        "--window-size",
        type=int,
        default=None,
        metavar="N",
        help="Score using the first N raw sensor columns instead of the full row.",
    )
    window_group.add_argument(
        "--window-sensors",
        type=_parse_window_sensors,
        default=None,
        metavar="NAME[,NAME...]",
        help=(
            "Comma-separated raw sensor names for an explicit scoring window. "
            "Mutually exclusive with --window-size."
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        default=False,
        help="Skip figure generation.",
    )
    return parser


def _reproduce_test_split(
    checkpoint: LoadedCheckpoint,
    raw_dir: Path,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Reproduce the held-out test split identical to the training-time split.

    Replicates the exact call sequence from ``prepare_data``: raw stratified
    train/test split with the checkpoint's ``random_seed``, same feasibility
    fallback. Returns the raw test matrix and labels, plus the raw sensor names.

    Args:
        checkpoint: Loaded checkpoint carrying ``random_seed`` and preprocessor.
        raw_dir: Directory containing raw SECOM data files.
        test_size: Held-out test fraction (from ``cfg.run.test_size``).

    Returns:
        ``(x_test_raw, y_test, raw_sensor_cols)`` where ``x_test_raw`` is
        float32 shape ``(n_te, n_raw_sensors)`` and ``y_test`` is int64 shape
        ``(n_te,)``.

    Raises:
        FileNotFoundError: If raw SECOM files are missing.
        ValueError: If raw SECOM schema is invalid or no sensor columns found.
    """
    raw_df = load_secom(raw_dir)
    validate_secom(raw_df)

    raw_sensor_cols = [c for c in raw_df.columns if str(c).startswith("sensor_")]
    if not raw_sensor_cols:
        raise ValueError("No raw sensor_ columns found")

    x_raw = raw_df[raw_sensor_cols].to_numpy(dtype=np.float32)
    y = raw_df["label"].to_numpy(dtype=np.int64)

    random_seed = int(checkpoint.raw["random_seed"])  # type: ignore[call-overload]
    _x_train, x_test, _y_train, y_test = _stratified_split(
        x_raw, y, test_size, random_seed
    )
    return x_test, y_test, raw_sensor_cols


def _choose_window(
    checkpoint: LoadedCheckpoint,
    raw_sensor_cols: list[str],
    x_test_raw: np.ndarray,
    window_size: int | None,
    window_sensors: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """Select the scoring window matrix and column names.

    Args:
        checkpoint: Loaded checkpoint with the fitted preprocessor.
        raw_sensor_cols: Full ordered list of raw sensor column names.
        x_test_raw: Raw test matrix, shape ``(n_te, n_raw_sensors)``.
        window_size: When not ``None``, use the first N raw sensor columns.
        window_sensors: When not ``None``, use these explicit raw sensor names.

    Returns:
        ``(x_window_raw, window_sensor_cols)`` ready for ``evaluate``.

    Raises:
        ValueError: If ``--window-size`` exceeds the raw sensor count.
    """
    if window_sensors is not None:
        # Build x_window_raw aligned to the requested names
        name_to_idx = {name: i for i, name in enumerate(raw_sensor_cols)}
        indices: list[int] = []
        for name in window_sensors:
            if name not in name_to_idx:
                raise ValueError(
                    f"--window-sensors: {name!r} not found in raw SECOM columns"
                )
            indices.append(name_to_idx[name])
        x_window = x_test_raw[:, indices]
        return x_window, window_sensors

    if window_size is not None:
        n_raw = len(raw_sensor_cols)
        if window_size < 1:
            raise ValueError(
                f"--window-size must be a positive integer, got {window_size}"
            )
        if window_size > n_raw:
            raise ValueError(
                f"--window-size {window_size} exceeds raw sensor count {n_raw}"
            )
        return x_test_raw[:, :window_size], raw_sensor_cols[:window_size]

    # Default: full row using the preprocessor's fitted raw schema
    return x_test_raw, list(checkpoint.preprocessor.raw_sensor_cols)


def _metrics_to_classification(seq: SequenceMetrics) -> ClassificationMetrics:
    """Convert ``SequenceMetrics`` to ``ClassificationMetrics`` for reporting.

    Args:
        seq: Sequence model metrics.

    Returns:
        A ``ClassificationMetrics`` with the overlapping fields populated.
    """
    return ClassificationMetrics(
        roc_auc=seq.roc_auc if not math.isnan(seq.roc_auc) else 0.0,
        pr_auc=seq.pr_auc if not math.isnan(seq.pr_auc) else 0.0,
        precision=seq.precision,
        recall=seq.recall,
        f1=seq.f1,
        confusion_matrix=seq.confusion_matrix,
    )


def main() -> None:
    """Entry point: evaluate the GRU and write metrics, figures, and comparison.

    Raises:
        SystemExit: With code 1 on ``FileNotFoundError``, ``ValueError``, or
            ``KeyError``; exits 0 on success (including when baseline is absent).
    """
    parser = _build_parser()
    args = parser.parse_args()

    try:
        cfg = load_config()

        checkpoint_path: Path = (
            args.checkpoint
            if args.checkpoint is not None
            else cfg.paths.models_dir / "sequence_gru.pt"
        )
        metrics_out: Path = (
            args.metrics_out
            if args.metrics_out is not None
            else cfg.paths.reports_dir / "sequence_model_metrics.json"
        )
        comparison_out: Path = (
            args.comparison_out
            if args.comparison_out is not None
            else cfg.paths.reports_dir / "sequence_model_comparison.json"
        )
        figures_dir: Path = (
            args.figures_dir
            if args.figures_dir is not None
            else cfg.paths.figures_dir
        )

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device=args.device)

        print("Reproducing held-out test split...")
        x_test_raw, y_test, raw_sensor_cols = _reproduce_test_split(
            checkpoint,
            cfg.paths.raw_dir,
            cfg.run.test_size,
        )

        x_window_raw, window_sensor_cols = _choose_window(
            checkpoint,
            raw_sensor_cols,
            x_test_raw,
            args.window_size,
            args.window_sensors,
        )

        print(
            f"Scoring {x_window_raw.shape[0]} test wafers "
            f"with {len(window_sensor_cols)} raw sensor columns..."
        )
        y_prob, metrics = evaluate(
            checkpoint=checkpoint,
            x_window_raw=x_window_raw,
            y_true=y_test,
            window_sensor_cols=window_sensor_cols,
            threshold=args.threshold,
            device=args.device,
        )

        cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        save_sequence_metrics(metrics, metrics_out)
        print(f"Metrics written to: {metrics_out}")

        # Figures
        if not args.no_figures:
            figures_dir.mkdir(parents=True, exist_ok=True)
            cm_path = figures_dir / "sequence_confusion_matrix.png"
            plot_confusion_matrix(metrics.confusion_matrix, cm_path)
            print(f"Confusion matrix saved: {cm_path}")

            if not math.isnan(metrics.roc_auc):
                roc_path = figures_dir / "sequence_roc_curve.png"
                plot_roc_curve(
                    y_test.astype(np.int64),
                    y_prob,
                    metrics.roc_auc,
                    roc_path,
                )
                print(f"ROC curve saved: {roc_path}")
            else:
                print("ROC curve skipped (AUC undefined for single-class y_true).")

            if not math.isnan(metrics.pr_auc):
                pr_path = figures_dir / "sequence_precision_recall_curve.png"
                plot_precision_recall_curve(
                    y_test.astype(np.int64),
                    y_prob,
                    metrics.pr_auc,
                    pr_path,
                )
                print(f"PR curve saved: {pr_path}")
            else:
                print("PR curve skipped (AUC undefined for single-class y_true).")

        # Baseline comparison
        baseline = load_tabular_baseline(cfg.paths.reports_dir, cfg.paths.models_dir)
        if baseline is None:
            print(
                "warning: no tabular baseline found; "
                "comparison written with verdict='no_baseline'",
                file=sys.stderr,
            )

        # Derive comparison_dir from the comparison_out path
        comparison_dir = comparison_out.parent
        comparison_dir.mkdir(parents=True, exist_ok=True)
        write_baseline_comparison(metrics, baseline, comparison_dir)
        print(f"Baseline comparison written to: {comparison_dir}")

        # Print summary
        cls_metrics = _metrics_to_classification(metrics)
        print()
        print(format_report(cls_metrics, model_name="GRU Sequence Model"))
        print(f"Balanced Acc: {metrics.balanced_accuracy:.3f}")
        print(f"Verdict: {compute_verdict(metrics, baseline)}")

    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
