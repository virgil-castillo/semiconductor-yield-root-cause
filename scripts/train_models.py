"""Train all model families, select the winner, and save artifacts.

Reads the processed training split only. Writes per-family fitted pipelines,
a stable selected_model.joblib for the winner, reports/cv_results.json
(cross-validation metrics + per-family frozen thresholds derived from OOF
train predictions), and reports/threshold.json (selected model's threshold).
The test set is never loaded here.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from yield_risk.config import load_config, load_cost_config, load_model_config
from yield_risk.model import (
    MODEL_REGISTRY,
    SearchResult,
    compute_fold_diagnostics,
    frozen_operating_threshold,
    run_search,
    select_best,
)
from yield_risk.monitoring import build_reference_profile, save_reference_profile


def main() -> None:
    """Train and tune every family, then persist artifacts and CV results."""
    cfg = load_config()
    model_cfg = load_model_config()
    cost_cfg = load_cost_config()

    train = pd.read_csv(cfg.paths.splits_dir / "train.csv")
    sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
    X_train = train[sensor_cols]
    y_train = train["label"]

    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)

    search_results: list[SearchResult] = []
    fold_diagnostics_map: dict[str, list[dict[str, float | int]]] = {}
    threshold_map: dict[str, float] = {}

    for name in MODEL_REGISTRY:
        print(f"=== {name} ===")
        result = run_search(
            name,
            X_train,
            y_train,
            model_cfg,
            cfg.run.cv_folds,
            cfg.run.random_seed,
            cfg.run.missing_threshold,
            cfg.run.cv_threshold,
            cfg.run.correlation_threshold,
        )
        print(
            f"  CV PR-AUC: {result.cv_pr_auc_mean:.3f} "
            f"+/- {result.cv_pr_auc_std:.3f}"
        )
        folds = compute_fold_diagnostics(
            result.estimator, X_train, y_train, cfg.run.cv_folds, cfg.run.random_seed
        )
        for fd in folds:
            print(
                f"  fold {fd['fold']}: prevalence={fd['val_prevalence']:.3f}"
                f"  roc_auc={fd['roc_auc']:.3f}"
            )
        fold_diagnostics_map[name] = folds
        # Derive frozen threshold from OOF train predictions (never from test)
        threshold = frozen_operating_threshold(
            result.estimator,
            X_train,
            y_train,
            cfg.run.cv_folds,
            cfg.run.random_seed,
            cost_cfg.cost_matrix,
            cost_cfg.threshold_search,
        )
        threshold_map[name] = threshold
        print(f"  frozen threshold: {threshold:.3f}")
        joblib.dump(result.estimator, cfg.paths.models_dir / f"{name}.joblib")
        search_results.append(result)

    winner = select_best(search_results)
    print(f"Winner: {winner}")
    winning = next(r for r in search_results if r.name == winner)
    joblib.dump(winning.estimator, cfg.paths.models_dir / "selected_model.joblib")

    cv_results = [
        {
            "model": r.name,
            "cv_pr_auc_mean": r.cv_pr_auc_mean,
            "cv_pr_auc_std": r.cv_pr_auc_std,
            "best_params": r.best_params,
            "selected": r.name == winner,
            "folds": fold_diagnostics_map[r.name],
            "threshold": threshold_map[r.name],
        }
        for r in search_results
    ]
    cv_path = cfg.paths.reports_dir / "cv_results.json"
    cv_path.write_text(json.dumps(cv_results, indent=2))
    print(f"Saved CV results to {cv_path}")

    # Persist the winner's threshold as a standalone artifact for convenience
    threshold_path = cfg.paths.reports_dir / "threshold.json"
    threshold_path.write_text(
        json.dumps(
            {"model": winner, "threshold": threshold_map[winner]}, indent=2
        )
    )
    print(f"Saved threshold to {threshold_path}")
    print(f"Saved selected model to {cfg.paths.models_dir / 'selected_model.joblib'}")

    # Snapshot the reference distribution for the monitoring notebook so it never
    # reloads raw training data at runtime.
    reference_profile = build_reference_profile(
        train, sensor_cols, winning.estimator, threshold_map[winner]
    )
    reference_path = cfg.paths.models_dir / "reference_profile.joblib"
    save_reference_profile(reference_profile, reference_path)
    print(f"Saved reference profile to {reference_path}")


if __name__ == "__main__":
    main()
