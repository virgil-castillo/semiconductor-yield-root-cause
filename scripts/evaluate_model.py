"""Evaluate the baseline model on the test set and save results."""
from __future__ import annotations

import joblib
import pandas as pd

from yield_risk.config import load_config
from yield_risk.evaluate import (
    compute_metrics,
    format_report,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_metrics,
)


def main() -> None:
    """Evaluate the baseline model on the test set and save results."""
    cfg = load_config()
    test = pd.read_csv(cfg.paths.processed_dir / "test.csv")
    pipeline = joblib.load(cfg.paths.models_dir / "baseline_lr.joblib")
    sensor_cols = [c for c in test.columns if c.startswith("sensor_")]
    X_test = test[sensor_cols]
    y_test = test["label"].to_numpy()
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.paths.reports_dir / "baseline_metrics.json"
    save_metrics(metrics, metrics_path)
    print(format_report(metrics))
    plot_confusion_matrix(
        metrics.confusion_matrix,
        cfg.paths.figures_dir / "confusion_matrix.png",
    )
    plot_roc_curve(
        y_test,
        y_prob,
        metrics.roc_auc,
        cfg.paths.figures_dir / "roc_curve.png",
    )
    plot_precision_recall_curve(
        y_test,
        y_prob,
        metrics.pr_auc,
        cfg.paths.figures_dir / "precision_recall_curve.png",
    )
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved figures to {cfg.paths.figures_dir}")


if __name__ == "__main__":
    main()
