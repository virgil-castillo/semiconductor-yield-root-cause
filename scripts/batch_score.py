"""Batch scoring: load a trained model and score an input CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from yield_risk.scoring import load_model_bundle, score_frame


def score_batch(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Load a fitted pipeline and score all rows in the input CSV.

    Appends two columns to the input data:

    - ``score``: predicted probability for the positive class (fail)
    - ``predicted_label``: 1 if ``score >= threshold``, else 0

    Args:
        model_path: Path to a joblib-serialised sklearn Pipeline.
        input_path: Path to a CSV file with sensor columns (and optionally
            ``label`` and ``timestamp``).
        output_path: Destination CSV path; parent directories are created
            if they do not exist.
        threshold: Decision threshold for converting scores to labels.

    Returns:
        DataFrame written to *output_path*, including all original columns
        plus ``score`` and ``predicted_label``.

    Raises:
        ValueError: If the input CSV contains no sensor_ columns.
    """
    bundle = load_model_bundle(model_path)
    df = pd.read_csv(input_path)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    if not sensor_cols:
        raise ValueError(
            f"No columns starting with 'sensor_' found in {input_path}. "
            "Verify the input CSV has the expected schema."
        )
    result = score_frame(bundle, df, threshold=threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    """CLI entry point for batch scoring."""
    parser = argparse.ArgumentParser(description="Score wafers with a fitted model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/selected_model.joblib"),
        help="Path to joblib model file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/splits/test.csv"),
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/batch_scores.csv"),
        help="Path for output CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold (default 0.5).",
    )
    args = parser.parse_args()
    result = score_batch(args.model, args.input, args.output, args.threshold)
    n_fail = int((result["predicted_label"] == 1).sum())
    print(f"Scored {len(result)} wafers. Predicted failures: {n_fail}")
    print(f"Saved scores to {args.output}")


if __name__ == "__main__":
    main()
