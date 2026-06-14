"""Framework-agnostic scoring service for the yield-risk prediction API."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Loaded model artefact with associated metadata.

    Attributes:
        pipeline: Fitted sklearn Pipeline used for inference.
        model_version: Version string for the loaded model.
        threshold: Decision threshold for converting scores to labels.
        expected_sensors: Ordered list of sensor column names the model expects.
    """

    pipeline: Pipeline
    model_version: str
    threshold: float
    expected_sensors: list[str]


def _short_hash(path: Path) -> str:
    """Return the first 7 hex characters of the sha256 hash of *path*'s bytes.

    Args:
        path: Path to the file to hash.

    Returns:
        Seven-character lowercase hex string.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:7]


def _sensor_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Extract sensor column names from a fitted pipeline.

    Uses ``pipeline.feature_names_in_`` (set by sklearn when the pipeline is
    fitted on a DataFrame), filtered to names starting with ``sensor_``.

    Args:
        pipeline: A fitted sklearn Pipeline.

    Returns:
        List of sensor column names in the pipeline's expected input order.
    """
    feature_names: list[str] = list(
        getattr(pipeline, "feature_names_in_", np.array([], dtype=object))
    )
    return [name for name in feature_names if name.startswith("sensor_")]


def load_model_bundle(
    model_path: Path,
    metadata_path: Path | None = None,
) -> ModelBundle:
    """Load a joblib pipeline and its associated metadata.

    When *metadata_path* is ``None``, defaults to
    ``model_path.parent / "model_metadata.json"``.  If that file does not
    exist, falls back to ``threshold = 0.5``, a ``model_version`` derived from
    the file's sha256 hash, and ``expected_sensors`` from the pipeline's
    ``feature_names_in_`` attribute.  A warning is logged on fallback.

    When the file exists, ``expected_sensors`` is taken from it if recorded,
    otherwise derived from the pipeline's ``feature_names_in_`` attribute;
    ``frozen_threshold`` and ``model_version`` are required.

    Args:
        model_path: Path to the joblib-serialised sklearn Pipeline.
        metadata_path: Path to the metadata JSON. Defaults to
            ``model_path.parent / "model_metadata.json"``.

    Returns:
        ModelBundle populated from the metadata file when present, or from
        pipeline introspection with fallback defaults.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
        ValueError: If *metadata_path* exists but is missing a required key
            (``frozen_threshold`` or ``model_version``).
    """
    pipeline: Pipeline = joblib.load(model_path)

    if metadata_path is None:
        metadata_path = model_path.parent / "model_metadata.json"

    if metadata_path.exists():
        with metadata_path.open() as fh:
            meta: dict[str, Any] = json.load(fh)
        try:
            threshold = float(meta["frozen_threshold"])
            model_version = str(meta["model_version"])
        except KeyError as exc:
            raise ValueError(
                f"Metadata {metadata_path} is missing required key: {exc}"
            ) from exc
        # expected_sensors is optional in the metadata: use it if recorded
        # (an explicit empty list is respected), otherwise fall back to the
        # fitted pipeline's input feature names.
        recorded_sensors = meta.get("expected_sensors")
        expected_sensors: list[str] = (
            list(recorded_sensors)
            if recorded_sensors is not None
            else _sensor_names_from_pipeline(pipeline)
        )
    else:
        logger.warning(
            "Metadata file not found at %s; using defaults (threshold=0.5, "
            "model_version from hash, sensors from pipeline).",
            metadata_path,
        )
        threshold = 0.5
        model_version = f"model-{_short_hash(model_path)}"
        expected_sensors = _sensor_names_from_pipeline(pipeline)

    return ModelBundle(
        pipeline=pipeline,
        model_version=model_version,
        threshold=threshold,
        expected_sensors=expected_sensors,
    )


def score_frame(
    bundle: ModelBundle,
    df: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Score a DataFrame using a loaded ModelBundle.

    Selects and aligns the model-input matrix to ``bundle.expected_sensors``
    in that exact order, filling any missing columns with ``NaN``.  Extra
    columns in *df* that are not in ``expected_sensors`` are ignored for
    scoring but preserved in the output.  The caller's DataFrame is never
    mutated.

    Args:
        bundle: Loaded ModelBundle containing the pipeline and metadata.
        df: Input DataFrame; may contain any columns, including non-sensor ones.
        threshold: Decision threshold. Uses ``bundle.threshold`` when ``None``.

    Returns:
        Copy of *df* with two additional columns appended:
        ``score`` (positive-class probability, float) and
        ``predicted_label`` (int 0/1, where 1 means ``score >= threshold``).
    """
    effective_threshold = bundle.threshold if threshold is None else threshold

    # Build model-input matrix aligned to expected_sensors (fills missing with NaN)
    aligned = {
        col: df[col] if col in df.columns else np.nan
        for col in bundle.expected_sensors
    }
    X = pd.DataFrame(aligned, index=df.index)

    scores: np.ndarray = bundle.pipeline.predict_proba(X)[:, 1]

    result = df.copy()
    result["score"] = scores
    result["predicted_label"] = (scores >= effective_threshold).astype(int)
    return result
