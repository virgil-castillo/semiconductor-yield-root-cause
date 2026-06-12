"""Evaluate top GRU checkpoints across a grid of scoring windows.

The main hyperparameter sweep (``run_sequence_sweep.py``) holds ``window_sizes``
fixed and only ever scores the full retained row. This follow-up takes a handful
of representative trial checkpoints — chosen as distinct *hyperparameter
families* — and re-evaluates each one across a grid of ``--window-size`` values
using the existing ``evaluate_sequence_model.py`` flow. It answers the open
question from ``docs/gru_sweep_results.md``: how gracefully do the top models
degrade when scored on only the first ``k`` raw sensors (current-window
scoring)?

This is an *eval-time* sweep: no model is retrained. Per-run metric files are
written to a throwaway temp directory; the only persisted artifact is the ranked
summary ``reports/sequence_window_sweep_results.csv``. The production
``reports/sequence_model_metrics.json`` is never touched.

Default families (override with ``--trials``):

* ``trial_025`` — shallow + fast (1 layer, lr 1e-3); the sweep winner.
* ``trial_028`` — deep + fast (2 layers, lr 1e-3); best validation ROC-AUC.
* ``trial_031`` — deep + slow (2 layers, lr 3e-4); least overfit (low val loss).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TRIALS = ("trial_025", "trial_028", "trial_031")
DEFAULT_WINDOWS = (64, 128, 256, 384, 512)
FULL_ROW_LABEL = "full"

# CSV column order for the ranked summary.
_SUMMARY_FIELDS = (
    "trial_id",
    "family",
    "window",
    "emb_dim",
    "hidden_size",
    "num_layers",
    "dropout",
    "lr",
    "val_pr_auc",
    "test_pr_auc",
    "test_roc_auc",
    "test_recall",
    "test_precision",
    "n_pos",
    "n_neg",
    "status",
    "error",
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the window sweep script.

    Returns:
        A configured ``ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate top GRU checkpoints across a grid of windows.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("reports/sequence_sweep_results.csv"),
        help="Ranked sweep summary used to resolve trial configs and paths.",
    )
    parser.add_argument(
        "--trials",
        type=str,
        default=",".join(DEFAULT_TRIALS),
        help="Comma-separated trial ids to evaluate (default: top 3 families).",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default=",".join(str(w) for w in DEFAULT_WINDOWS),
        help="Comma-separated raw-column window sizes (full row always added).",
    )
    parser.add_argument(
        "--no-full-row",
        action="store_true",
        default=False,
        help="Skip the full-row (no --window-size) evaluation.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Compute device passed through to the evaluation script.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for thresholded metrics.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/sequence_window_sweep_results.csv"),
        help="Path for the ranked window-sweep summary CSV.",
    )
    return parser


def _load_trial_rows(
    results_csv: Path, trial_ids: list[str]
) -> dict[str, dict[str, str]]:
    """Load selected trial rows from the ranked sweep summary CSV.

    Args:
        results_csv: Path to ``sequence_sweep_results.csv``.
        trial_ids: Trial ids to extract, in the requested order.

    Returns:
        Mapping of trial id to its CSV row (column name -> string value).

    Raises:
        FileNotFoundError: If ``results_csv`` does not exist.
        KeyError: If any requested trial id is absent from the summary.
    """
    if not results_csv.is_file():
        raise FileNotFoundError(f"Sweep results not found: {results_csv}")

    with results_csv.open(newline="") as handle:
        by_id = {row["trial_id"]: row for row in csv.DictReader(handle)}

    missing = [tid for tid in trial_ids if tid not in by_id]
    if missing:
        raise KeyError(f"Trial ids not in {results_csv}: {', '.join(missing)}")

    return {tid: by_id[tid] for tid in trial_ids}


def _family_label(row: dict[str, str]) -> str:
    """Derive a short human-readable family label from a trial's config.

    Args:
        row: A trial row from the sweep summary CSV.

    Returns:
        A label such as ``"shallow+fast"`` describing depth and learning rate.
    """
    depth = "shallow" if int(row["num_layers"]) == 1 else "deep"
    speed = "fast" if float(row["lr"]) >= 1e-3 else "slow"
    return f"{depth}+{speed}"


def _evaluate_window(
    checkpoint: Path,
    window: int | None,
    device: str,
    threshold: float,
) -> dict[str, object]:
    """Run one evaluation at a single window size and parse its metrics.

    Invokes ``scripts/evaluate_sequence_model.py`` as a subprocess, writing its
    metrics and comparison JSON to a throwaway temp directory so production
    report artifacts are never overwritten.

    Args:
        checkpoint: Path to the trial checkpoint to score.
        window: Number of leading raw sensor columns, or ``None`` for full row.
        device: Compute device to pass through.
        threshold: Decision threshold for thresholded metrics.

    Returns:
        A dict with ``status`` plus parsed metric fields. On failure,
        ``status`` is ``"failed"`` and ``error`` carries a short message.
    """
    with tempfile.TemporaryDirectory() as tmp:
        metrics_out = Path(tmp) / "metrics.json"
        cmd = [
            sys.executable,
            "scripts/evaluate_sequence_model.py",
            "--checkpoint",
            str(checkpoint),
            "--device",
            device,
            "--threshold",
            str(threshold),
            "--metrics-out",
            str(metrics_out),
            "--comparison-out",
            str(Path(tmp) / "comparison.json"),
            "--no-figures",
        ]
        if window is not None:
            cmd += ["--window-size", str(window)]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0 or not metrics_out.is_file():
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            msg = tail[-1] if tail else f"exit code {result.returncode}"
            return {"status": "failed", "error": msg}

        metrics = json.loads(metrics_out.read_text())

    return {
        "status": "ok",
        "error": "",
        "test_pr_auc": metrics["pr_auc"],
        "test_roc_auc": metrics["roc_auc"],
        "test_recall": metrics["recall"],
        "test_precision": metrics["precision"],
        "n_pos": metrics["n_pos"],
        "n_neg": metrics["n_neg"],
    }


def _write_summary(out: Path, rows: list[dict[str, object]]) -> None:
    """Write the ranked window-sweep summary CSV.

    Args:
        out: Destination CSV path.
        rows: Result rows, each keyed by ``_SUMMARY_FIELDS``.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_SUMMARY_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in _SUMMARY_FIELDS})


def _print_pivot(rows: list[dict[str, object]]) -> None:
    """Print a trial x window grid of test PR-AUC to stdout.

    Args:
        rows: Result rows produced by the sweep.
    """
    windows = sorted(
        {str(r["window"]) for r in rows},
        key=lambda w: (w == FULL_ROW_LABEL, int(w) if w.isdigit() else 0),
    )
    trials = sorted({str(r["trial_id"]) for r in rows})
    lookup = {(str(r["trial_id"]), str(r["window"])): r for r in rows}

    header = f"{'trial':>10} " + " ".join(f"{w:>7}" for w in windows)
    print("\nTest PR-AUC by window:")
    print(header)
    for tid in trials:
        cells = []
        for w in windows:
            cell = lookup.get((tid, w))
            if cell and cell["status"] == "ok":
                cells.append(f"{float(str(cell['test_pr_auc'])):>7.4f}")
            else:
                cells.append(f"{'--':>7}")
        print(f"{tid:>10} " + " ".join(cells))


def main() -> None:
    """Entry point: run the window sweep and write the ranked summary.

    Raises:
        SystemExit: With code 1 on ``FileNotFoundError``, ``ValueError``, or
            ``KeyError``; exits 0 on success.
    """
    parser = _build_parser()
    args = parser.parse_args()

    try:
        trial_ids = [t.strip() for t in args.trials.split(",") if t.strip()]
        windows: list[int | None] = [
            int(w.strip()) for w in args.windows.split(",") if w.strip()
        ]
        if not args.no_full_row:
            windows.append(None)

        trial_rows = _load_trial_rows(args.results_csv, trial_ids)

        summary: list[dict[str, object]] = []
        for tid in trial_ids:
            row = trial_rows[tid]
            checkpoint = Path(row["checkpoint_path"])
            family = _family_label(row)
            if not checkpoint.is_file():
                print(f"warning: missing checkpoint for {tid}: {checkpoint}")
            for window in windows:
                label = FULL_ROW_LABEL if window is None else str(window)
                print(f"Evaluating {tid} ({family}) @ window={label} ...")
                metrics = _evaluate_window(
                    checkpoint, window, args.device, args.threshold
                )
                summary.append(
                    {
                        "trial_id": tid,
                        "family": family,
                        "window": label,
                        "emb_dim": row["emb_dim"],
                        "hidden_size": row["hidden_size"],
                        "num_layers": row["num_layers"],
                        "dropout": row["dropout"],
                        "lr": row["lr"],
                        "val_pr_auc": row["val_pr_auc"],
                        **metrics,
                    }
                )

        _write_summary(args.out, summary)
        n_ok = sum(1 for r in summary if r["status"] == "ok")
        print(f"\nWindow sweep complete: {n_ok}/{len(summary)} evaluations ok.")
        print(f"Summary written to: {args.out}")
        _print_pivot(summary)

    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
