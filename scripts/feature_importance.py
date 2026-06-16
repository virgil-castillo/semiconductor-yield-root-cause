"""Extract and visualize feature importance from the selected model."""
from __future__ import annotations

import joblib

from yield_risk.config import load_config
from yield_risk.importance import extract_feature_importance, plot_top_features
from yield_risk.model import model_feature_names


def main() -> None:
    """Extract and visualize feature importance from the selected model."""
    cfg = load_config()
    pipeline = joblib.load(cfg.paths.models_dir / "selected_model.joblib")
    feature_names = model_feature_names(pipeline)
    importance_df = extract_feature_importance(pipeline, feature_names)
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.paths.reports_dir / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = cfg.paths.figures_dir / "feature_importance.png"
    plot_top_features(
        importance_df, top_n=20, output_path=png_path, value_column="importance"
    )
    print("Top 10 features:")
    print(importance_df.head(10).to_string(index=False))
    print(f"Saved importance to {csv_path}")
    print(f"Saved figure to {png_path}")


if __name__ == "__main__":
    main()
