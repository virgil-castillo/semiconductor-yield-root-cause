"""Train baseline logistic regression and save the fitted pipeline."""
from __future__ import annotations

import joblib
import pandas as pd

from yield_risk.config import load_config
from yield_risk.model import build_baseline_pipeline, cross_validate_model, train_model


def main() -> None:
    """Train baseline logistic regression and save the fitted pipeline."""
    cfg = load_config()
    train = pd.read_csv(cfg.paths.processed_dir / "train.csv")
    sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
    X_train = train[sensor_cols]
    y_train = train["label"]
    pipeline = build_baseline_pipeline(cfg.run.random_seed)
    cv_results = cross_validate_model(
        pipeline, X_train, y_train, cfg.run.cv_folds, cfg.run.random_seed
    )
    roc_mean = float(cv_results["test_roc_auc"].mean())
    roc_std = float(cv_results["test_roc_auc"].std())
    f1_mean = float(cv_results["test_f1"].mean())
    f1_std = float(cv_results["test_f1"].std())
    print(f"CV ROC-AUC: {roc_mean:.3f} +/- {roc_std:.3f}")
    print(f"CV F1:      {f1_mean:.3f} +/- {f1_std:.3f}")
    pipeline = train_model(pipeline, X_train, y_train)
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.paths.models_dir / "baseline_lr.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
