"""Generate models/model_metadata.json for the yield-risk prediction API.

Loads the selected pipeline and test predictions, computes the cost-optimal
threshold from configs/cost_config.yaml, and writes a JSON artifact that
``yield_risk.scoring.load_model_bundle`` reads at serving time.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from yield_risk.config import load_config, load_cost_config
from yield_risk.thresholding import find_optimal_threshold


def export_model_metadata(
    model_path: Path,
    test_path: Path,
    cv_results_path: Path,
    metrics_path: Path,
    cost_config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Export model metadata to a JSON file.

    Loads the selected pipeline and held-out test set, computes the
    cost-optimal threshold, and writes a JSON artifact that
    ``yield_risk.scoring.load_model_bundle`` reads at serving time.

    Args:
        model_path: Path to the joblib-serialised selected pipeline.
        test_path: Path to the held-out test CSV (sensor cols + label).
        cv_results_path: Path to cv_results.json produced by train_models.py.
        metrics_path: Path to selected_model_metrics.json.
        cost_config_path: Path to cost_config.yaml.
        output_path: Destination path for model_metadata.json.

    Returns:
        The metadata dict that was written to output_path.

    Raises:
        FileNotFoundError: If any required input file does not exist.
        ValueError: If no entry in cv_results_path has ``selected: true``.
    """
    # Load pipeline
    pipeline = joblib.load(model_path)

    # Load test set and derive sensor columns
    test_df = pd.read_csv(test_path)
    sensor_cols = [c for c in test_df.columns if c.startswith("sensor_")]
    X_test = test_df[sensor_cols]
    y_test = test_df["label"].to_numpy()
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Compute cost-optimal threshold
    cost_cfg = load_cost_config(cost_config_path)
    result = find_optimal_threshold(
        y_test, y_prob, cost_cfg.cost_matrix, cost_cfg.threshold_search
    )
    optimal_threshold = float(result.threshold)

    # Model family from cv_results
    cv_results = json.loads(cv_results_path.read_text())
    family: str | None = next(
        (r["model"] for r in cv_results if r["selected"]), None
    )
    if family is None:
        raise ValueError(f"No selected model found in {cv_results_path}")

    # Short hash: first 7 chars of sha256 of model file bytes
    short_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:7]
    model_version = f"{family}-{short_hash}"

    # Metrics snapshot
    metrics: dict[str, Any] = json.loads(metrics_path.read_text())

    # Assemble metadata dict
    metadata: dict[str, Any] = {
        "model_version": model_version,
        "optimal_threshold": optimal_threshold,
        "cost_matrix": dataclasses.asdict(cost_cfg.cost_matrix),
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "expected_sensors": sensor_cols,
    }

    # Write JSON, creating parent dirs if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2))

    return metadata


def main() -> None:
    """Build paths from load_config() defaults and call export_model_metadata.

    Reads ``configs/config.yaml`` for model/report/data paths, calls
    ``export_model_metadata`` with the standard artifact locations, and
    prints the model version and output path on success.
    """
    cfg = load_config()
    model_path = cfg.paths.models_dir / "selected_model.joblib"
    test_path = cfg.paths.processed_dir / "test.csv"
    cv_results_path = cfg.paths.reports_dir / "cv_results.json"
    metrics_path = cfg.paths.reports_dir / "selected_model_metrics.json"
    cost_config_path = Path("configs/cost_config.yaml")
    output_path = cfg.paths.models_dir / "model_metadata.json"

    result = export_model_metadata(
        model_path=model_path,
        test_path=test_path,
        cv_results_path=cv_results_path,
        metrics_path=metrics_path,
        cost_config_path=cost_config_path,
        output_path=output_path,
    )
    print(f"Wrote metadata for {result['model_version']} to {output_path}")


if __name__ == "__main__":
    main()
