"""Generate markdown reports from processed data and model artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from yield_risk.config import load_config
from yield_risk.monitoring import build_monitoring_summary
from yield_risk.reporting import (
    load_report_inputs,
    render_report_bundle,
    write_report_bundle,
)


class _ProbabilityEstimator(Protocol):
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for feature rows.

        Args:
            features: Feature matrix to score.

        Returns:
            Probability array with one column per class.
        """


def _split_reference_current(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into deterministic reference and current batches.

    Args:
        df: Input rows ordered as they should be split.

    Returns:
        A tuple containing the first-half reference batch and second-half
        current batch.

    Raises:
        ValueError: If the frame has fewer than 2 rows.
    """
    if len(df) < 2:
        raise ValueError("DataFrame must contain at least 2 rows to split batches.")
    midpoint = len(df) // 2
    return df.iloc[:midpoint].copy(), df.iloc[midpoint:].copy()


def generate_reports(config_path: Path = Path("configs/config.yaml")) -> None:
    """Generate the markdown report bundle from existing pipeline artifacts.

    Args:
        config_path: Path to the run configuration YAML file.

    Returns:
        None.

    Raises:
        FileNotFoundError: If required configured artifacts are absent.
        ValueError: If split data or comparison artifacts are invalid.
    """
    cfg = load_config(config_path)
    train = pd.read_csv(cfg.paths.splits_dir / "train.csv")
    test = pd.read_csv(cfg.paths.splits_dir / "test.csv")
    sensor_cols = _sensor_columns(test)
    if not sensor_cols:
        raise ValueError(
            "No columns starting with 'sensor_' found in test.csv. "
            "Verify the processed data has the expected schema."
        )

    reference, current = _split_reference_current(test)
    pipeline: _ProbabilityEstimator = joblib.load(
        cfg.paths.models_dir / "selected_model.joblib"
    )
    reference_scores = _score_batch(pipeline, reference[sensor_cols])
    current_scores = _score_batch(pipeline, current[sensor_cols])

    model_comparison = pd.read_csv(cfg.paths.reports_dir / "model_comparison.csv")
    threshold = _selected_threshold(model_comparison)
    monitoring_summary = build_monitoring_summary(
        reference=reference[sensor_cols],
        current=current[sensor_cols],
        feature_cols=sensor_cols,
        reference_scores=reference_scores,
        current_scores=current_scores,
        high_risk_threshold=threshold,
    )
    report_inputs = load_report_inputs(
        cfg.paths.reports_dir,
        train,
        test,
        monitoring_summary,
    )
    bundle = render_report_bundle(report_inputs)
    write_report_bundle(bundle, cfg.paths.reports_dir)


def main() -> None:
    """Parse CLI arguments and generate markdown reports.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Generate markdown reports from model and monitoring artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to the run configuration YAML file.",
    )
    args = parser.parse_args()
    generate_reports(args.config)


def _sensor_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("sensor_")]


def _score_batch(pipeline: _ProbabilityEstimator, features: pd.DataFrame) -> np.ndarray:
    scores = pipeline.predict_proba(features)[:, 1]
    return np.asarray(scores, dtype=float)


def _selected_threshold(model_comparison: pd.DataFrame) -> float:
    if "selected" not in model_comparison.columns:
        raise ValueError("model_comparison.csv missing required column: selected")
    if "opt_threshold" not in model_comparison.columns:
        raise ValueError("model_comparison.csv missing required column: opt_threshold")
    selected_rows = model_comparison[model_comparison["selected"].astype(bool)]
    if len(selected_rows) != 1:
        raise ValueError("model_comparison.csv must contain exactly one selected row.")
    return float(selected_rows.iloc[0]["opt_threshold"])


if __name__ == "__main__":
    main()
