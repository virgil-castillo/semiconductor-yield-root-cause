"""Evaluate the selected model on the test set and build the comparison table.

Loads the held-out test split exactly once. Reports the selected (winning)
model at its frozen operating threshold (derived from OOF train predictions by
train_models.py and persisted in cv_results.json), regenerates its diagnostic
figures, and writes reports/model_comparison.{csv,json} combining the
cross-validation metrics (from train_models.py) with test metrics for every
family.

The threshold is NEVER computed from test labels here.  Each family's frozen
threshold is read from the ``"threshold"`` field of its cv_results.json entry.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from yield_risk.config import load_config, load_cost_config
from yield_risk.evaluate import (
    evaluate_at_threshold,
    format_report,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_metrics,
)


def main() -> None:
    """Evaluate the winner on test and assemble the model comparison table."""
    cfg = load_config()
    cost_cfg = load_cost_config()

    test = pd.read_csv(cfg.paths.splits_dir / "test.csv")
    sensor_cols = [c for c in test.columns if c.startswith("sensor_")]
    X_test = test[sensor_cols]
    y_test = test["label"].to_numpy()

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)

    cv_results = json.loads((cfg.paths.reports_dir / "cv_results.json").read_text())
    selected = next(r["model"] for r in cv_results if r["selected"])
    # Build a map of family → frozen threshold (from OOF train predictions)
    frozen_thresholds = {r["model"]: float(r["threshold"]) for r in cv_results}

    # --- Selected model: report + figures at the frozen threshold ---
    pipeline = joblib.load(cfg.paths.models_dir / "selected_model.joblib")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    frozen = frozen_thresholds[selected]
    metrics, expected_cost = evaluate_at_threshold(
        y_test, y_prob, frozen, cost_cfg.cost_matrix
    )
    save_metrics(metrics, cfg.paths.reports_dir / "selected_model_metrics.json")
    print(format_report(metrics, selected))
    print(
        f"Frozen threshold: {frozen:.3f}  "
        f"expected cost: {expected_cost:.1f}"
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
        fam_frozen = frozen_thresholds[r["model"]]
        fam_metrics, fam_cost = evaluate_at_threshold(
            y_test, fam_prob, fam_frozen, cost_cfg.cost_matrix
        )
        rows.append(
            {
                "model": r["model"],
                "cv_pr_auc_mean": r["cv_pr_auc_mean"],
                "test_pr_auc": fam_metrics.pr_auc,
                "test_roc_auc": fam_metrics.roc_auc,
                "test_recall": fam_metrics.recall,
                "test_precision": fam_metrics.precision,
                "opt_threshold": fam_frozen,
                "expected_cost": fam_cost,
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
