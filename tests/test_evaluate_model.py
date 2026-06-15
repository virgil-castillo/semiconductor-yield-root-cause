"""Integration tests for scripts/evaluate_model.py.

Drives ``evaluate_model.main()`` end-to-end with hermetic, tmp_path-scoped
artifacts and asserts that ``find_optimal_threshold`` is never called — the
frozen threshold from ``cv_results.json`` must be used instead.

The matplotlib backend is forced to ``Agg`` here (before any import of the
script, which in turn imports plotting functions) to prevent GUI-related
errors in headless CI environments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import evaluate_model  # noqa: E402  (scripts/ on pythonpath via pyproject)

from yield_risk.config import (  # noqa: E402
    Config,
    CostConfig,
    CostMatrix,
    PathsConfig,
    RunConfig,
    ThresholdSearchConfig,
)
from yield_risk.model import build_pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSOR_COLS = ["sensor_001", "sensor_002", "sensor_003"]
FAMILIES = ["logistic_regression", "random_forest"]
# Frozen thresholds written into cv_results.json — must appear verbatim in
# model_comparison.json rather than being recomputed.
FROZEN_THRESHOLDS: dict[str, float] = {
    "logistic_regression": 0.4,
    "random_forest": 0.3,
}
SELECTED_FAMILY = "random_forest"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_train_frame(rng: np.random.Generator) -> pd.DataFrame:
    """Return a small training frame with SENSOR_COLS and a 0/1 label.

    Args:
        rng: NumPy random generator for reproducible data.

    Returns:
        DataFrame with columns ``sensor_001``, ``sensor_002``, ``sensor_003``,
        and ``label``, containing both positive and negative examples.
    """
    n = 40
    df = pd.DataFrame(rng.random((n, len(SENSOR_COLS))), columns=SENSOR_COLS)
    # Ensure both classes are present.
    labels = np.zeros(n, dtype=int)
    labels[: n // 4] = 1
    rng.shuffle(labels)
    df["label"] = labels
    return df


def _make_test_frame(rng: np.random.Generator) -> pd.DataFrame:
    """Return a small test frame with SENSOR_COLS and a 0/1 label.

    Args:
        rng: NumPy random generator for reproducible data.

    Returns:
        DataFrame with columns ``sensor_001``, ``sensor_002``, ``sensor_003``,
        and ``label``, containing both positive and negative examples.
    """
    n = 20
    df = pd.DataFrame(rng.random((n, len(SENSOR_COLS))), columns=SENSOR_COLS)
    labels = np.zeros(n, dtype=int)
    labels[: n // 4] = 1
    rng.shuffle(labels)
    df["label"] = labels
    return df


def _fit_and_dump_pipeline(
    name: str,
    train: pd.DataFrame,
    models_dir: Path,
) -> Path:
    """Fit a pipeline for ``name`` on ``train`` and dump to ``models_dir``.

    Args:
        name: Registered model family identifier.
        train: DataFrame with SENSOR_COLS + label.
        models_dir: Directory to write the serialized pipeline.

    Returns:
        Path of the written ``.joblib`` file.
    """
    pipe = build_pipeline(
        name,
        random_seed=0,
        missing_threshold=0.9,
        cv_threshold=0.0,
        correlation_threshold=0.999,
    )
    X = train[SENSOR_COLS]
    y = train["label"]
    pipe.fit(X, y)
    out = models_dir / f"{name}.joblib"
    joblib.dump(pipe, out)
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def artifact_root(tmp_path: Path) -> Path:
    """Create the directory tree expected by evaluate_model.main().

    Returns:
        The ``tmp_path`` root; subdirectories are created inside.
    """
    for subdir in ("splits", "models", "reports", "figures", "raw", "interim"):
        (tmp_path / subdir).mkdir()
    return tmp_path


@pytest.fixture()
def populated_artifacts(artifact_root: Path) -> dict[str, Path]:
    """Populate all artifact files required by evaluate_model.main().

    Writes:
    - ``splits/test.csv`` — small test frame.
    - ``models/<family>.joblib`` — fitted pipeline per family.
    - ``models/selected_model.joblib`` — copy of the selected family's pipeline.
    - ``reports/cv_results.json`` — one entry per family with frozen threshold.

    Returns:
        Mapping of artifact names to their paths, plus the root dirs.
    """
    rng = np.random.default_rng(42)
    root = artifact_root

    # Test CSV
    test_df = _make_test_frame(rng)
    test_csv = root / "splits" / "test.csv"
    test_df.to_csv(test_csv, index=False)

    # Train data for fitting pipelines
    train_df = _make_train_frame(rng)

    models_dir = root / "models"
    for name in FAMILIES:
        _fit_and_dump_pipeline(name, train_df, models_dir)

    # selected_model.joblib is a copy of the SELECTED_FAMILY's pipeline
    selected_src = models_dir / f"{SELECTED_FAMILY}.joblib"
    joblib.dump(joblib.load(selected_src), models_dir / "selected_model.joblib")

    # cv_results.json
    cv_records: list[dict[str, Any]] = [
        {
            "model": name,
            "cv_pr_auc_mean": 0.85 if name == SELECTED_FAMILY else 0.72,
            "selected": name == SELECTED_FAMILY,
            "threshold": FROZEN_THRESHOLDS[name],
        }
        for name in FAMILIES
    ]
    cv_json = root / "reports" / "cv_results.json"
    cv_json.write_text(json.dumps(cv_records))

    return {
        "root": root,
        "test_csv": test_csv,
        "cv_json": cv_json,
        "models_dir": models_dir,
        "reports_dir": root / "reports",
        "figures_dir": root / "figures",
    }


@pytest.fixture()
def fake_config(artifact_root: Path) -> Config:
    """Return a Config whose paths all point into artifact_root.

    Args:
        artifact_root: Temporary directory tree created by the fixture.

    Returns:
        Config instance ready to be monkeypatched onto evaluate_model.load_config.
    """
    root = artifact_root
    return Config(
        paths=PathsConfig(
            raw_dir=root / "raw",
            interim_dir=root / "interim",
            splits_dir=root / "splits",
            models_dir=root / "models",
            reports_dir=root / "reports",
            figures_dir=root / "figures",
        ),
        run=RunConfig(
            random_seed=0,
            test_size=0.2,
            cv_folds=3,
            missing_threshold=0.9,
            cv_threshold=0.0,
            correlation_threshold=0.999,
        ),
    )


@pytest.fixture()
def fake_cost_config() -> CostConfig:
    """Return a minimal CostConfig for evaluate_model.main().

    Returns:
        CostConfig with a simple cost matrix and threshold search grid.
    """
    return CostConfig(
        cost_matrix=CostMatrix(
            true_pass=0.0,
            true_fail=0.0,
            false_fail=1.0,
            false_pass=10.0,
        ),
        threshold_search=ThresholdSearchConfig(low=0.1, high=0.9, steps=9),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_main_never_calls_find_optimal_threshold(
    populated_artifacts: dict[str, Path],
    fake_config: Config,
    fake_cost_config: CostConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate_model.main() never calls find_optimal_threshold.

    Patches find_optimal_threshold at its definition site with a spy whose
    side_effect raises AssertionError so any accidental call fails loudly.
    Asserts call_count is zero after main() completes successfully.
    """
    spy = Mock(
        side_effect=AssertionError(
            "find_optimal_threshold must not be called on the evaluate path"
        )
    )
    monkeypatch.setattr("yield_risk.thresholding.find_optimal_threshold", spy)
    monkeypatch.setattr(
        evaluate_model, "load_config", lambda: fake_config
    )
    monkeypatch.setattr(
        evaluate_model, "load_cost_config", lambda: fake_cost_config
    )

    evaluate_model.main()

    assert spy.call_count == 0


def test_main_writes_model_comparison_csv(
    populated_artifacts: dict[str, Path],
    fake_config: Config,
    fake_cost_config: CostConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate_model.main() writes reports/model_comparison.csv.

    Args are populated by their respective fixtures.
    """
    monkeypatch.setattr(evaluate_model, "load_config", lambda: fake_config)
    monkeypatch.setattr(evaluate_model, "load_cost_config", lambda: fake_cost_config)

    evaluate_model.main()

    assert (fake_config.paths.reports_dir / "model_comparison.csv").exists()


def test_main_writes_model_comparison_json(
    populated_artifacts: dict[str, Path],
    fake_config: Config,
    fake_cost_config: CostConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate_model.main() writes reports/model_comparison.json.

    Args are populated by their respective fixtures.
    """
    monkeypatch.setattr(evaluate_model, "load_config", lambda: fake_config)
    monkeypatch.setattr(evaluate_model, "load_cost_config", lambda: fake_cost_config)

    evaluate_model.main()

    assert (fake_config.paths.reports_dir / "model_comparison.json").exists()


def test_main_writes_selected_model_metrics_json(
    populated_artifacts: dict[str, Path],
    fake_config: Config,
    fake_cost_config: CostConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate_model.main() writes reports/selected_model_metrics.json.

    Args are populated by their respective fixtures.
    """
    monkeypatch.setattr(evaluate_model, "load_config", lambda: fake_config)
    monkeypatch.setattr(evaluate_model, "load_cost_config", lambda: fake_cost_config)

    evaluate_model.main()

    assert (fake_config.paths.reports_dir / "selected_model_metrics.json").exists()


def test_model_comparison_frozen_threshold_equals_frozen_threshold(
    populated_artifacts: dict[str, Path],
    fake_config: Config,
    fake_cost_config: CostConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each family's frozen_threshold in model_comparison.json equals the frozen value.

    Proves the threshold written to the comparison report is the frozen value
    read from cv_results.json, not a recomputed one derived from test labels.
    """
    monkeypatch.setattr(evaluate_model, "load_config", lambda: fake_config)
    monkeypatch.setattr(evaluate_model, "load_cost_config", lambda: fake_cost_config)

    evaluate_model.main()

    comparison_path = fake_config.paths.reports_dir / "model_comparison.json"
    rows: list[dict[str, Any]] = json.loads(comparison_path.read_text())
    for row in rows:
        family = row["model"]
        expected_threshold = FROZEN_THRESHOLDS[family]
        assert row["frozen_threshold"] == pytest.approx(expected_threshold), (
            f"Family '{family}': expected frozen threshold {expected_threshold}, "
            f"got {row['frozen_threshold']}"
        )
