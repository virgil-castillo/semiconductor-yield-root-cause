"""Evaluate the selected model on the test set and build the comparison table.

Loads the held-out test split exactly once. Reports the selected (winning)
model with its cost-sensitive operating point, regenerates its diagnostic
figures, and writes reports/model_comparison.{csv,json} combining the
cross-validation metrics (from train_models.py) with test metrics for every
family.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from yield_risk.config import load_config, load_cost_config
from yield_risk.evaluate import (
    compute_metrics,
    format_report,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_metrics,
)
from yield_risk.thresholding import find_optimal_threshold


def main() -> None:
    """Evaluate the winner on test and assemble the model comparison table."""
    cfg = load_config()
    cost_cfg = load_cost_config()

    test = pd.read_csv(cfg.paths.processed_dir / "test.csv")
    sensor_cols = [c for c in test.columns if c.startswith("sensor_")]
    X_test = test[sensor_cols]
    y_test = test["label"].to_numpy()

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)

    cv_results = json.loads((cfg.paths.reports_dir / "cv_results.json").read_text())
    selected = next(r["model"] for r in cv_results if r["selected"])

    # --- Selected model: report + figures at the cost-optimal threshold ---
    pipeline = joblib.load(cfg.paths.models_dir / "selected_model.joblib")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    opt = find_optimal_threshold(
        y_test, y_prob, cost_cfg.cost_matrix, cost_cfg.threshold_search
    )
    metrics = compute_metrics(y_test, y_prob, threshold=opt.threshold)
    save_metrics(metrics, cfg.paths.reports_dir / "selected_model_metrics.json")
    print(format_report(metrics, selected))
    print(
        f"Optimal threshold: {opt.threshold:.3f}  "
        f"expected cost: {opt.expected_cost:.1f}"
    )
    plot_confusion_matrix(
        metrics.confusion_matrix, cfg.paths.figures_dir / "confusion_matrix.png"
    )
    plot_roc_curve(
        y_test, y_prob, metrics.roc_auc, cfg.paths.figures_dir / "roc_curve.png"
    )
    plot_precision_recall_curve(
        y_test,
        y_prob,
        metrics.pr_auc,
        cfg.paths.figures_dir / "precision_recall_curve.png",
    )

    # --- Comparison table: CV (already computed) + test metrics per family ---
    rows = []
    for r in cv_results:
        fam_pipe = joblib.load(cfg.paths.models_dir / f"{r['model']}.joblib")
        fam_prob = fam_pipe.predict_proba(X_test)[:, 1]
        fam_opt = find_optimal_threshold(
            y_test, fam_prob, cost_cfg.cost_matrix, cost_cfg.threshold_search
        )
        fam_metrics = compute_metrics(y_test, fam_prob, threshold=fam_opt.threshold)
        rows.append(
            {
                "model": r["model"],
                "cv_pr_auc_mean": r["cv_pr_auc_mean"],
                "test_pr_auc": fam_metrics.pr_auc,
                "test_roc_auc": fam_metrics.roc_auc,
                "test_recall": fam_metrics.recall,
                "test_precision": fam_metrics.precision,
                "opt_threshold": fam_opt.threshold,
                "expected_cost": fam_opt.expected_cost,
                "selected": r["selected"],
            }
        )

    pd.DataFrame(rows).to_csv(
        cfg.paths.reports_dir / "model_comparison.csv", index=False
    )
    (cfg.paths.reports_dir / "model_comparison.json").write_text(
        json.dumps(rows, indent=2)
    )
    print(f"Saved comparison to {cfg.paths.reports_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
