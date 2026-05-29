"""Extract and visualize feature importance from the baseline model."""
from __future__ import annotations

import joblib
import pandas as pd

from yield_risk.config import load_config
from yield_risk.importance import extract_lr_coefficients, plot_top_features


def main() -> None:
    """Extract and visualize feature importance from the baseline model."""
    cfg = load_config()
    test = pd.read_csv(cfg.paths.processed_dir / "test.csv")
    pipeline = joblib.load(cfg.paths.models_dir / "baseline_lr.joblib")
    sensor_cols = [c for c in test.columns if c.startswith("sensor_")]
    importance_df = extract_lr_coefficients(pipeline, sensor_cols)
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.paths.reports_dir / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = cfg.paths.figures_dir / "feature_importance.png"
    plot_top_features(importance_df, top_n=20, output_path=png_path)
    print("Top 10 features:")
    print(importance_df.head(10).to_string(index=False))
    print(f"Saved importance to {csv_path}")
    print(f"Saved figure to {png_path}")


if __name__ == "__main__":
    main()
