"""Run an Optuna early-detection hyperparameter search and write artifacts.

Entry point for the Spec B / §5 CLI.  Loads the SECOM dataset, splits off a
held-out test set (never touched beyond this split), runs the Optuna study on
the training portion, and writes three artifacts:

* ``models/early_detection_study.pkl``  — serialised ``optuna.Study``
* ``models/early_detection_best.json``  — best-trial summary record
* ``reports/early_detection_trials.csv`` — full trials dataframe
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import joblib

from yield_risk.config import load_config, load_cost_config
from yield_risk.data import load_secom
from yield_risk.early_detection import (
    build_best_record,
    load_early_detection_config,
    run_study,
)
from yield_risk.preprocess import split_stratified
from yield_risk.validation import validate_secom


def _build_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Build the overrides mapping from parsed CLI arguments.

    Only flags the user actually supplied (non-``None``) are included.
    ``--seed`` maps to the ``sampler_seed`` config key.  ``--config`` is the
    path argument and is never included here.

    Args:
        args: Parsed argument namespace.  Unsupplied flags have value ``None``.

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


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the Optuna study, and write the three artifacts.

    Args:
        argv: Optional argument list for testing (defaults to ``sys.argv[1:]``
            when ``None``).

    Raises:
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

    args: argparse.Namespace = parser.parse_args(argv)
    overrides = _build_overrides(args)

    # Load configs
    cfg = load_config()
    ed_cfg = load_early_detection_config(args.config, overrides=overrides)

    # Load and validate data
    df = load_secom(cfg.paths.raw_dir)
    validate_secom(df)

    # Identify sensor columns — column order is the progress axis
    raw_sensor_cols: list[str] = [c for c in df.columns if c.startswith("sensor_")]

    if len(raw_sensor_cols) == 0:
        print("No raw sensor_ columns found", file=sys.stderr)
        sys.exit(1)

    # Stratified split — test split is discarded here, untouched
    train_df, _test_df = split_stratified(
        df, cfg.run.test_size, cfg.run.random_seed
    )

    x_train = train_df[raw_sensor_cols]
    y_train: Any = train_df["label"].to_numpy()

    # Load cost matrix only when needed
    cost_matrix = None
    if ed_cfg.detection_metric == "neg_expected_cost":
        cost_matrix = load_cost_config().cost_matrix

    # Run Optuna study
    study = run_study(
        x_train,
        y_train,
        raw_sensor_cols,
        ed_cfg,
        random_seed=cfg.run.random_seed,
        cost_matrix=cost_matrix,
    )

    # Create output directories
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)

    # --- Artifact 1: serialised study ----------------------------------------
    study_path = cfg.paths.models_dir / "early_detection_study.pkl"
    joblib.dump(study, study_path)

    # --- Artifact 2: best-trial JSON record ----------------------------------
    n_sensors = len(raw_sensor_cols)
    best_record = build_best_record(
        study,
        ed_cfg,
        n_sensors=n_sensors,
        test_size=cfg.run.test_size,
        random_seed=cfg.run.random_seed,
    )
    best_json_path = cfg.paths.models_dir / "early_detection_best.json"
    best_json_path.write_text(json.dumps(best_record, indent=2), encoding="utf-8")

    # --- Artifact 3: trials CSV ----------------------------------------------
    trials_csv_path = cfg.paths.reports_dir / "early_detection_trials.csv"
    study.trials_dataframe().to_csv(trials_csv_path, index=False)

    # --- Summary print -------------------------------------------------------
    bt = study.best_trial
    print(
        f"Best trial : #{bt.number}\n"
        f"  penalized_score    : {best_record['penalized_score']}\n"
        f"  detection_metric   : {best_record['detection_metric']}\n"
        f"  observation_frac   : {best_record['observation_fraction']}\n"
        f"  latest_sensor_idx  : {best_record['latest_index']}\n"
        f"  model_family       : {bt.user_attrs.get('model_family', 'n/a')}\n"
        f"Artifacts written to:\n"
        f"  {study_path}\n"
        f"  {best_json_path}\n"
        f"  {trials_csv_path}"
    )


if __name__ == "__main__":
    main()
