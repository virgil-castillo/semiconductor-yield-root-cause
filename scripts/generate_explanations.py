"""Generate SHAP explanations and root-cause candidates from a fitted model."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from yield_risk.config import load_config
from yield_risk.explainability import (
    compute_shap_values,
    global_feature_importance,
    save_shap_values,
)
from yield_risk.root_cause import (
    fail_shap_lift,
    rank_root_cause_candidates,
    spc_flag_rate,
)


def main() -> None:
    """Generate SHAP explanations and root-cause report.

    Loads the selected model and processed train/test splits from
    paths defined in ``configs/config.yaml``.  Saves three artefacts to
    ``reports/``:

    - ``shap_values.npz``: raw SHAP values for the test set
    - ``shap_global_importance.csv``: mean |SHAP| per feature, sorted
    - ``root_cause_candidates.csv``: sensors ranked by composite score

    Raises:
        ValueError: If the processed test CSV contains no sensor_ columns.
    """
    cfg = load_config()

    model_path: Path = cfg.paths.models_dir / "selected_model.joblib"
    pipeline = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    train = pd.read_csv(cfg.paths.splits_dir / "train.csv")
    test = pd.read_csv(cfg.paths.splits_dir / "test.csv")
    sensor_cols = [c for c in test.columns if c.startswith("sensor_")]
    if not sensor_cols:
        raise ValueError(
            "No columns starting with 'sensor_' found in test.csv. "
            "Verify the processed data has the expected schema."
        )

    X_train = train[sensor_cols]
    X_test = test[sensor_cols]
    y_test = test["label"].to_numpy()

    print(f"Computing SHAP values for {len(X_test)} test wafers …")
    explanations = compute_shap_values(pipeline, X_train, X_test)

    global_imp = global_feature_importance(explanations)
    flag_rates = spc_flag_rate(test, sensor_cols)
    lift_df = fail_shap_lift(explanations, y_test)
    root_cause_df = rank_root_cause_candidates(global_imp, lift_df, flag_rates)

    reports_dir: Path = cfg.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    shap_path = reports_dir / "shap_values.npz"
    save_shap_values(explanations, shap_path)
    print(f"Saved SHAP values to {shap_path}")

    imp_path = reports_dir / "shap_global_importance.csv"
    global_imp.to_csv(imp_path, index=False)
    print(f"Saved global importance to {imp_path}")

    rc_path = reports_dir / "root_cause_candidates.csv"
    root_cause_df.to_csv(rc_path, index=False)
    print(f"Saved root-cause candidates to {rc_path}")

    top5 = root_cause_df.head(5)
    print("\n=== Top 5 Root-Cause Candidates ===")
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
