"""Configuration loading for the yield-risk pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    """Filesystem paths used throughout the pipeline.

    Attributes:
        raw_dir: Directory containing raw SECOM data files.
        interim_dir: Directory for intermediate processed artifacts.
        processed_dir: Directory for final train/test splits.
        models_dir: Directory for serialized model artifacts.
        reports_dir: Directory for generated reports.
        figures_dir: Directory for generated figures.
    """

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path


@dataclass
class RunConfig:
    """Run-level parameters controlling data processing and cross-validation.

    Attributes:
        random_seed: Global RNG seed for reproducibility.
        test_size: Fraction of data held out for final evaluation.
        val_size: Fraction of training data used for validation.
        cv_folds: Number of stratified CV folds.
        missing_threshold: Drop features with missing rate above this value.
        variance_threshold: Drop features with variance below this value.
        correlation_threshold: Drop one of each pair with |r| above this value.
    """

    random_seed: int
    test_size: float
    val_size: float
    cv_folds: int
    missing_threshold: float
    variance_threshold: float
    correlation_threshold: float


@dataclass
class Config:
    """Top-level pipeline configuration.

    Attributes:
        paths: Filesystem path configuration.
        run: Run-level parameter configuration.
    """

    paths: PathsConfig
    run: RunConfig


@dataclass
class CostMatrix:
    """Per-outcome costs used for threshold optimization.

    Attributes:
        true_pass: Cost of correctly releasing a passing wafer.
        true_fail: Cost of correctly flagging a failing wafer.
        false_fail: Cost of unnecessarily holding a passing wafer.
        false_pass: Cost of releasing a failing wafer (quality escape).
    """

    true_pass: float
    true_fail: float
    false_fail: float
    false_pass: float


@dataclass
class ThresholdSearchConfig:
    """Search grid for cost-sensitive threshold optimization.

    Attributes:
        low: Lower bound of threshold search range.
        high: Upper bound of threshold search range.
        steps: Number of candidate thresholds to evaluate.
    """

    low: float
    high: float
    steps: int


@dataclass
class CostConfig:
    """Cost configuration for threshold optimization.

    Attributes:
        cost_matrix: Per-outcome cost assignments.
        threshold_search: Threshold search grid parameters.
    """

    cost_matrix: CostMatrix
    threshold_search: ThresholdSearchConfig


def load_config(path: Path | str = "configs/config.yaml") -> Config:
    """Load run configuration from a YAML file.

    Args:
        path: Path to config.yaml.

    Returns:
        Parsed Config dataclass.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    paths = PathsConfig(
        raw_dir=Path(raw["paths"]["raw_dir"]),
        interim_dir=Path(raw["paths"]["interim_dir"]),
        processed_dir=Path(raw["paths"]["processed_dir"]),
        models_dir=Path(raw["paths"]["models_dir"]),
        reports_dir=Path(raw["paths"]["reports_dir"]),
        figures_dir=Path(raw["paths"]["figures_dir"]),
    )
    run = RunConfig(**raw["run"])
    return Config(paths=paths, run=run)


def load_model_config(
    path: Path | str = "configs/model_config.yaml",
) -> dict[str, Any]:
    """Load model hyperparameter search configuration from a YAML file.

    Args:
        path: Path to model_config.yaml.

    Returns:
        Dict with keys ``models`` (per-model search grids) and ``search``
        (RandomizedSearchCV settings).

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")
    with cfg_path.open() as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def load_cost_config(path: Path | str = "configs/cost_config.yaml") -> CostConfig:
    """Load cost matrix configuration from a YAML file.

    Args:
        path: Path to cost_config.yaml.

    Returns:
        Parsed CostConfig dataclass.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cost config not found: {cfg_path}")
    with cfg_path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return CostConfig(
        cost_matrix=CostMatrix(**raw["cost_matrix"]),
        threshold_search=ThresholdSearchConfig(**raw["threshold_search"]),
    )
