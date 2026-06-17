"""Run early-detection study/evaluation and write artifacts.

Entry point for the Spec B / Spec C CLI. Loads the SECOM dataset, optionally
runs the Optuna study on the training portion, and always evaluates the best
early-detection record on the held-out split.

Study mode writes:

* ``models/early_detection_study.pkl`` - serialised ``optuna.Study``
* ``models/early_detection_best.json`` - best-trial summary record
* ``reports/early_detection_trials.csv`` - full trials dataframe

Both modes write:

* ``reports/early_detection_metrics.json``
* ``reports/early_detection_comparison.json``
* ``reports/early_detection_comparison.csv``
* ``reports/early_detection_curve.csv``
* ``reports/figures/early_detection_curve.png``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from yield_risk.config import load_config, load_cost_config
from yield_risk.data import load_secom
from yield_risk.early_detection import (
    build_best_record,
    evaluate_best_on_holdout,
    load_early_detection_config,
    run_study,
)
from yield_risk.preprocess import split_stratified
from yield_risk.validation import validate_secom


def _build_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Build the overrides mapping from parsed CLI arguments.

    Only flags the user actually supplied (non-``None``) are included.
    ``--seed`` maps to the ``sampler_seed`` config key. ``--config`` is the
    path argument and is never included here.

    Args:
        args: Parsed argument namespace. Unsupplied flags have value ``None``.

    Returns:
        A ``dict[str, object]`` whose ``None`` values are kept (the
        ``load_early_detection_config`` loader silently ignores them), matching
        exactly the four overrideable config keys.
    """
    return {
        "n_trials": args.n_trials,
        "alpha": args.alpha,
        "detection_metric": args.detection_metric,
        "sampler_seed": args.seed,
    }


def _load_best_record(path: Path) -> dict[str, object]:
    """Load an existing early-detection best-record JSON artifact.

    Args:
        path: Path to ``models/early_detection_best.json``.

    Returns:
        Parsed JSON mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the artifact is not a JSON object.
    """
    if not path.exists():
        raise FileNotFoundError(f"Early-detection best artifact not found: {path}")

    parsed: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Early-detection best artifact must be a JSON object: {path}")
    return cast("dict[str, object]", parsed)


def _rows_from_result(
    result: Mapping[str, object],
    key: str,
) -> list[Mapping[str, object]]:
    """Return a list of row mappings from a holdout result.

    Args:
        result: Holdout-evaluation result from ``evaluate_best_on_holdout``.
        key: Row-list key to extract.

    Returns:
        Row mappings under ``key``.

    Raises:
        ValueError: If ``key`` is missing or does not contain row mappings.
    """
    rows_obj = result.get(key)
    if not isinstance(rows_obj, list):
        raise ValueError(f"{key} must be a list")

    rows: list[Mapping[str, object]] = []
    for row in rows_obj:
        if not isinstance(row, Mapping):
            raise ValueError(f"{key} entries must be objects")
        rows.append(cast("Mapping[str, object]", row))
    return rows


def _finite_float(value: object) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when unavailable.

    Args:
        value: Candidate numeric value.

    Returns:
        Finite float, or ``None``.
    """
    if value is None:
        return None
    try:
        result = float(cast("Any", value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _plot_early_detection_curve(
    curve_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    """Plot held-out PR-AUC by prefix length using the Agg backend.

    Args:
        curve_rows: Diagnostic prefix-sweep metric rows.
        comparison_rows: Held-out comparison rows.
        output_path: Destination PNG path.

    Raises:
        ValueError: If no curve rows have numeric ``latest_index`` and
            ``test_pr_auc`` values.
    """
    points: list[tuple[float, float]] = []
    for row in curve_rows:
        x_value = _finite_float(row.get("latest_index"))
        y_value = _finite_float(row.get("test_pr_auc"))
        if x_value is not None and y_value is not None:
            points.append((x_value, y_value))

    if not points:
        raise ValueError("early_detection_curve.csv requires at least one point")

    points.sort(key=lambda item: item[0])

    figure = Figure(figsize=(7.0, 4.5), tight_layout=True)
    FigureCanvas(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        marker="o",
        label="Prefix sweep",
    )

    for row in comparison_rows:
        role = str(row.get("role"))
        pr_auc = _finite_float(row.get("test_pr_auc"))
        latest = _finite_float(row.get("latest_index"))
        if role == "early_optuna_best" and pr_auc is not None and latest is not None:
            axis.scatter(
                [latest],
                [pr_auc],
                marker="*",
                s=140,
                label="Selected early model",
                zorder=3,
            )
        elif role in {"full_prefix_same_config", "tabular_selected_model"}:
            if pr_auc is not None:
                axis.axhline(
                    pr_auc,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"{role} PR-AUC",
                )

    axis.set_xlabel("latest_index")
    axis.set_ylabel("test_pr_auc")
    axis.set_title("Held-out early-detection curve")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="png", dpi=150)


def _write_spec_c_artifacts(
    holdout_result: Mapping[str, object],
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    """Write Spec C JSON, CSV, and curve figure artifacts.

    Args:
        holdout_result: JSON-safe result from ``evaluate_best_on_holdout``.
        reports_dir: Configured reports directory.
        figures_dir: Configured figures directory.

    Raises:
        ValueError: If comparison or curve rows are malformed.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = _rows_from_result(holdout_result, "comparison_rows")
    curve_rows = _rows_from_result(holdout_result, "curve_rows")

    comparison_path = reports_dir / "early_detection_comparison.json"
    comparison_path.write_text(
        json.dumps(dict(holdout_result), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    metrics_path = reports_dir / "early_detection_metrics.json"
    metrics_payload = {
        "primary_metric": holdout_result.get("primary_metric"),
        "verdict": holdout_result.get("verdict"),
        "comparison_rows": comparison_rows,
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    pd.DataFrame(comparison_rows).to_csv(
        reports_dir / "early_detection_comparison.csv",
        index=False,
    )
    pd.DataFrame(curve_rows).to_csv(
        reports_dir / "early_detection_curve.csv",
        index=False,
    )
    _plot_early_detection_curve(
        curve_rows,
        comparison_rows,
        figures_dir / "early_detection_curve.png",
    )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, evaluate early detection, and write artifacts.

    Args:
        argv: Optional argument list for testing (defaults to ``sys.argv[1:]``
            when ``None``).

    Raises:
        FileNotFoundError: If ``--evaluate-existing`` is used and the best JSON
            artifact does not exist.
        SystemExit: With code 1 if no ``sensor_`` columns are found in the
            dataset.
    """
    parser = argparse.ArgumentParser(
        description="Run early-detection Optuna study on SECOM data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/early_detection_config.yaml",
        help="Path to early_detection_config.yaml.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Earliness penalty weight.",
    )
    parser.add_argument(
        "--detection-metric",
        type=str,
        default=None,
        help="Detection metric name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optuna sampler seed (overrides sampler_seed in config).",
    )
    parser.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="Skip Optuna and evaluate models/early_detection_best.json.",
    )

    args: argparse.Namespace = parser.parse_args(argv)
    overrides = _build_overrides(args)

    cfg = load_config()
    ed_cfg = load_early_detection_config(args.config, overrides=overrides)

    study_path = cfg.paths.models_dir / "early_detection_study.pkl"
    best_json_path = cfg.paths.models_dir / "early_detection_best.json"
    trials_csv_path = cfg.paths.reports_dir / "early_detection_trials.csv"

    best_record: dict[str, object]
    if args.evaluate_existing:
        best_record = _load_best_record(best_json_path)

    df = load_secom(cfg.paths.raw_dir)
    validate_secom(df)

    raw_sensor_cols: list[str] = [c for c in df.columns if c.startswith("sensor_")]
    if len(raw_sensor_cols) == 0:
        print("No raw sensor_ columns found", file=sys.stderr)
        sys.exit(1)

    cost_cfg = load_cost_config()
    study_cost_matrix = None
    if ed_cfg.detection_metric == "neg_expected_cost":
        study_cost_matrix = cost_cfg.cost_matrix
    holdout_cost_matrix = cost_cfg.cost_matrix

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate_existing:
        print(f"Evaluating existing best artifact: {best_json_path}")
    else:
        cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
        train_df, _test_df = split_stratified(
            df, cfg.run.test_size, cfg.run.random_seed
        )
        x_train = train_df[raw_sensor_cols]
        y_train: Any = train_df["label"].to_numpy()

        study = run_study(
            x_train,
            y_train,
            raw_sensor_cols,
            ed_cfg,
            random_seed=cfg.run.random_seed,
            cost_matrix=study_cost_matrix,
        )
        joblib.dump(study, study_path)

        best_record = build_best_record(
            study,
            ed_cfg,
            n_sensors=len(raw_sensor_cols),
            test_size=cfg.run.test_size,
            random_seed=cfg.run.random_seed,
        )
        best_json_path.write_text(
            json.dumps(best_record, indent=2),
            encoding="utf-8",
        )
        study.trials_dataframe().to_csv(trials_csv_path, index=False)

        bt = study.best_trial
        print(
            f"Best trial : #{bt.number}\n"
            f"  penalized_score    : {best_record['penalized_score']}\n"
            f"  detection_metric   : {best_record['detection_metric']}\n"
            f"  observation_frac   : {best_record['observation_fraction']}\n"
            f"  latest_sensor_idx  : {best_record['latest_index']}\n"
            f"  model_family       : {bt.user_attrs.get('model_family', 'n/a')}\n"
            f"Spec B artifacts written to:\n"
            f"  {study_path}\n"
            f"  {best_json_path}\n"
            f"  {trials_csv_path}"
        )

    holdout_result = evaluate_best_on_holdout(
        df,
        best_record,
        ed_cfg,
        holdout_cost_matrix,
        model_comparison_path=cfg.paths.reports_dir / "model_comparison.json",
    )
    _write_spec_c_artifacts(
        holdout_result,
        cfg.paths.reports_dir,
        cfg.paths.figures_dir,
    )
    print(
        "Spec C artifacts written to:\n"
        f"  {cfg.paths.reports_dir / 'early_detection_metrics.json'}\n"
        f"  {cfg.paths.reports_dir / 'early_detection_comparison.json'}\n"
        f"  {cfg.paths.reports_dir / 'early_detection_comparison.csv'}\n"
        f"  {cfg.paths.reports_dir / 'early_detection_curve.csv'}\n"
        f"  {cfg.paths.figures_dir / 'early_detection_curve.png'}"
    )


if __name__ == "__main__":
    main()
