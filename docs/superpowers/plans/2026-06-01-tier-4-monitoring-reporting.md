# Tier 4 Monitoring and Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic monitoring utilities, reproducible markdown report generation, and an empirical monitoring notebook for Tier 4.

**Architecture:** Monitoring logic lives in pure library functions that accept pandas/numpy inputs and return typed dataclasses. Reporting code loads existing artifacts, validates required columns, renders markdown, and writes reports through a single orchestration path. The notebook calls the monitoring library on processed data and selected-model scores so empirical evidence uses the same code path as generated reports.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, scikit-learn/joblib, pytest, ruff, mypy, nbformat JSON.

---

## File Structure

- Create `src/yield_risk/monitoring.py`: dataclasses and pure drift-comparison functions.
- Create `tests/test_monitoring.py`: unit tests for monitoring calculations and validation errors.
- Create `src/yield_risk/reporting.py`: artifact loaders, markdown renderers, and report writer.
- Create `tests/test_reporting.py`: unit tests for report rendering, artifact validation, and write behavior.
- Create `scripts/generate_reports.py`: CLI entry point that loads config/artifacts, scores deterministic batches, builds monitoring evidence, and writes markdown reports.
- Create `notebooks/07_model_monitoring_drift_checks.ipynb`: report-style empirical confirmation notebook using `yield_risk.monitoring`.
- Generate `reports/executive_summary.md`, `reports/model_card.md`, `reports/data_card.md`, and `reports/root_cause_report.md`.
- Create `tmp/tier4-pr-report.md`: PR-description style summary of changes, verification, and limitations.

## Shared Commands

Run Python commands through the repository conda environment:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
```

---

### Task 1: Monitoring Utilities

**Files:**
- Create: `tests/test_monitoring.py`
- Create: `src/yield_risk/monitoring.py`

- [ ] **Step 1: Write failing monitoring tests**

Create `tests/test_monitoring.py` with tests covering missingness drift, feature distribution drift, prediction drift, high-risk-rate drift, summary assembly, and validation errors.

```python
"""Tests for lightweight monitoring utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yield_risk.monitoring import (
    build_monitoring_summary,
    compare_feature_distributions,
    compare_high_risk_rate,
    compare_missingness,
    compare_prediction_distributions,
)


def _reference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor_001": [1.0, 2.0, 3.0, 4.0],
            "sensor_002": [10.0, 10.0, 11.0, 11.0],
            "label": [0, 0, 1, 0],
        }
    )


def _current_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor_001": [2.0, 4.0, 6.0, 8.0],
            "sensor_002": [10.0, np.nan, np.nan, 14.0],
            "label": [0, 1, 1, 0],
        }
    )


class TestCompareMissingness:
    def test_flags_features_above_missingness_delta_threshold(self) -> None:
        result = compare_missingness(
            _reference_frame(),
            _current_frame(),
            ["sensor_001", "sensor_002"],
            threshold=0.25,
        )

        rows = result.to_frame().set_index("feature")
        assert rows.loc["sensor_002", "missing_rate_delta"] == pytest.approx(0.5)
        assert bool(rows.loc["sensor_002", "alert"]) is True
        assert bool(rows.loc["sensor_001", "alert"]) is False


class TestCompareFeatureDistributions:
    def test_computes_numeric_drift_statistics(self) -> None:
        result = compare_feature_distributions(
            _reference_frame(),
            _current_frame(),
            ["sensor_001"],
            threshold=1.0,
        )

        row = result.to_frame().iloc[0]
        assert row["feature"] == "sensor_001"
        assert row["reference_mean"] == pytest.approx(2.5)
        assert row["current_mean"] == pytest.approx(5.0)
        assert row["mean_delta"] == pytest.approx(2.5)
        assert row["ks_statistic"] >= 0.0
        assert bool(row["alert"]) is True


class TestComparePredictionDistributions:
    def test_flags_changed_score_distribution(self) -> None:
        result = compare_prediction_distributions(
            np.array([0.05, 0.10, 0.15, 0.20]),
            np.array([0.40, 0.50, 0.60, 0.70]),
            threshold=0.20,
        )

        assert result.reference_mean == pytest.approx(0.125)
        assert result.current_mean == pytest.approx(0.55)
        assert result.mean_delta == pytest.approx(0.425)
        assert result.alert is True


class TestCompareHighRiskRate:
    def test_uses_supplied_operating_threshold(self) -> None:
        result = compare_high_risk_rate(
            np.array([0.05, 0.10, 0.80, 0.90]),
            np.array([0.70, 0.75, 0.80, 0.90]),
            threshold=0.50,
            rate_delta_threshold=0.25,
        )

        assert result.reference_high_risk_rate == pytest.approx(0.5)
        assert result.current_high_risk_rate == pytest.approx(1.0)
        assert result.rate_delta == pytest.approx(0.5)
        assert result.alert is True


class TestBuildMonitoringSummary:
    def test_combines_all_monitoring_sections(self) -> None:
        summary = build_monitoring_summary(
            _reference_frame(),
            _current_frame(),
            ["sensor_001", "sensor_002"],
            np.array([0.05, 0.10, 0.80, 0.90]),
            np.array([0.70, 0.75, 0.80, 0.90]),
            high_risk_threshold=0.50,
        )

        assert len(summary.missingness.results) == 2
        assert len(summary.feature_drift.results) == 2
        assert summary.prediction_drift.alert is True
        assert summary.high_risk_rate.alert is True


class TestValidation:
    def test_missing_feature_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="missing required feature columns"):
            compare_missingness(
                _reference_frame(),
                _current_frame(),
                ["sensor_missing"],
            )

    def test_invalid_threshold_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="threshold must be"):
            compare_prediction_distributions(
                np.array([0.1, 0.2]),
                np.array([0.2, 0.3]),
                threshold=-0.1,
            )

    def test_empty_scores_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="must contain at least one score"):
            compare_prediction_distributions(np.array([]), np.array([0.1]))
```

- [ ] **Step 2: Run monitoring tests to verify they fail**

Run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
pytest tests/test_monitoring.py -v
```

Expected: collection or import failure because `yield_risk.monitoring` does not exist.

- [ ] **Step 3: Implement monitoring utilities**

Create `src/yield_risk/monitoring.py` with:

```python
"""Lightweight batch monitoring utilities for yield-risk scoring."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class MissingnessDriftResult:
    """Missingness drift result for one feature.

    Attributes:
        feature: Feature name.
        reference_missing_rate: Null fraction in the reference batch.
        current_missing_rate: Null fraction in the current batch.
        missing_rate_delta: Absolute difference between current and reference rates.
        alert: Whether the delta exceeds the configured threshold.
    """

    feature: str
    reference_missing_rate: float
    current_missing_rate: float
    missing_rate_delta: float
    alert: bool


@dataclass
class MissingnessDriftSummary:
    """Missingness drift results for multiple features.

    Attributes:
        threshold: Alert threshold used for missing-rate deltas.
        results: Per-feature missingness drift results.
    """

    threshold: float
    results: list[MissingnessDriftResult]

    def to_frame(self) -> pd.DataFrame:
        """Convert results to a DataFrame.

        Returns:
            DataFrame with one row per feature.
        """


@dataclass
class FeatureDriftResult:
    """Numeric distribution drift result for one feature.

    Attributes:
        feature: Feature name.
        reference_mean: Reference batch mean.
        current_mean: Current batch mean.
        mean_delta: Absolute mean difference.
        reference_std: Reference batch standard deviation.
        current_std: Current batch standard deviation.
        std_delta: Absolute standard-deviation difference.
        reference_median: Reference batch median.
        current_median: Current batch median.
        median_delta: Absolute median difference.
        ks_statistic: Two-sample Kolmogorov-Smirnov statistic.
        ks_p_value: Two-sample Kolmogorov-Smirnov p-value.
        alert: Whether drift exceeds the configured threshold.
    """

    feature: str
    reference_mean: float
    current_mean: float
    mean_delta: float
    reference_std: float
    current_std: float
    std_delta: float
    reference_median: float
    current_median: float
    median_delta: float
    ks_statistic: float
    ks_p_value: float
    alert: bool


@dataclass
class FeatureDriftSummary:
    """Numeric distribution drift results for multiple features.

    Attributes:
        threshold: Alert threshold used for drift statistics.
        results: Per-feature distribution drift results.
    """

    threshold: float
    results: list[FeatureDriftResult]

    def to_frame(self) -> pd.DataFrame:
        """Convert results to a DataFrame.

        Returns:
            DataFrame with one row per feature sorted by alert and mean delta.
        """


@dataclass
class PredictionDriftResult:
    """Prediction score distribution drift summary.

    Attributes:
        reference_mean: Reference score mean.
        current_mean: Current score mean.
        mean_delta: Absolute score mean difference.
        reference_median: Reference score median.
        current_median: Current score median.
        median_delta: Absolute score median difference.
        reference_std: Reference score standard deviation.
        current_std: Current score standard deviation.
        std_delta: Absolute score standard-deviation difference.
        reference_p90: Reference 90th percentile.
        current_p90: Current 90th percentile.
        p90_delta: Absolute 90th-percentile difference.
        ks_statistic: Two-sample Kolmogorov-Smirnov statistic.
        ks_p_value: Two-sample Kolmogorov-Smirnov p-value.
        threshold: Alert threshold.
        alert: Whether any monitored score statistic exceeds the threshold.
    """

    reference_mean: float
    current_mean: float
    mean_delta: float
    reference_median: float
    current_median: float
    median_delta: float
    reference_std: float
    current_std: float
    std_delta: float
    reference_p90: float
    current_p90: float
    p90_delta: float
    ks_statistic: float
    ks_p_value: float
    threshold: float
    alert: bool


@dataclass
class HighRiskRateDriftResult:
    """High-risk-rate drift summary.

    Attributes:
        threshold: Score threshold used to flag high-risk wafers.
        reference_high_risk_rate: Reference high-risk fraction.
        current_high_risk_rate: Current high-risk fraction.
        rate_delta: Absolute high-risk-rate difference.
        rate_delta_threshold: Alert threshold for the rate delta.
        alert: Whether the rate delta exceeds the alert threshold.
    """

    threshold: float
    reference_high_risk_rate: float
    current_high_risk_rate: float
    rate_delta: float
    rate_delta_threshold: float
    alert: bool


@dataclass
class MonitoringSummary:
    """Combined monitoring summary for feature and score drift.

    Attributes:
        missingness: Missingness drift summary.
        feature_drift: Numeric feature drift summary.
        prediction_drift: Prediction distribution drift summary.
        high_risk_rate: High-risk-rate drift summary.
    """

    missingness: MissingnessDriftSummary
    feature_drift: FeatureDriftSummary
    prediction_drift: PredictionDriftResult
    high_risk_rate: HighRiskRateDriftResult
```

Implement helper validation functions:

```python
def _validate_threshold(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} threshold must be non-negative.")


def _validate_feature_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one feature.")
    missing = sorted(
        (set(feature_cols) - set(reference.columns))
        | (set(feature_cols) - set(current.columns))
    )
    if missing:
        raise ValueError(
            "missing required feature columns: " + ", ".join(missing)
        )
    if reference.empty or current.empty:
        raise ValueError("reference and current batches must not be empty.")
```

Implement the public functions named in the spec. Use `dataclasses.asdict` in
`to_frame()` methods. For KS statistics, drop null values first; if either side
is empty after dropping nulls, set `ks_statistic=0.0` and `ks_p_value=1.0`.

- [ ] **Step 4: Run monitoring tests to verify they pass**

Run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
pytest tests/test_monitoring.py -v
```

Expected: all tests in `tests/test_monitoring.py` pass.

- [ ] **Step 5: Commit monitoring utilities**

Run:

```powershell
git add src/yield_risk/monitoring.py tests/test_monitoring.py
git commit -m "feat: add monitoring drift utilities"
```

---

### Task 2: Reporting Utilities

**Files:**
- Create: `tests/test_reporting.py`
- Create: `src/yield_risk/reporting.py`

- [ ] **Step 1: Write failing reporting tests**

Create `tests/test_reporting.py` with tests that exercise markdown rendering
without relying on the repository's real artifacts.

```python
"""Tests for markdown report generation utilities."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from yield_risk.monitoring import (
    HighRiskRateDriftResult,
    MissingnessDriftResult,
    MissingnessDriftSummary,
    MonitoringSummary,
    PredictionDriftResult,
    FeatureDriftResult,
    FeatureDriftSummary,
)
from yield_risk.reporting import (
    DataSummary,
    ReportBundle,
    ReportInputs,
    render_data_card,
    render_executive_summary,
    render_model_card,
    render_root_cause_report,
    write_report_bundle,
)


def _model_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["random_forest", "xgboost"],
            "cv_pr_auc_mean": [0.217, 0.208],
            "test_pr_auc": [0.193, 0.261],
            "test_roc_auc": [0.758, 0.802],
            "test_recall": [0.571, 0.667],
            "test_precision": [0.171, 0.222],
            "opt_threshold": [0.08, 0.03],
            "expected_cost": [148.0, 119.0],
            "selected": [True, False],
        }
    )


def _root_cause() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor": ["sensor_059", "sensor_033"],
            "mean_abs_shap": [0.0057, 0.0042],
            "shap_lift": [0.0106, 0.0018],
            "spc_flag_rate": [0.019, 0.016],
            "composite_score": [0.871, 0.479],
        }
    )


def _summary() -> MonitoringSummary:
    return MonitoringSummary(
        missingness=MissingnessDriftSummary(
            threshold=0.05,
            results=[
                MissingnessDriftResult(
                    feature="sensor_059",
                    reference_missing_rate=0.0,
                    current_missing_rate=0.1,
                    missing_rate_delta=0.1,
                    alert=True,
                )
            ],
        ),
        feature_drift=FeatureDriftSummary(
            threshold=0.1,
            results=[
                FeatureDriftResult(
                    feature="sensor_059",
                    reference_mean=1.0,
                    current_mean=1.5,
                    mean_delta=0.5,
                    reference_std=0.1,
                    current_std=0.2,
                    std_delta=0.1,
                    reference_median=1.0,
                    current_median=1.4,
                    median_delta=0.4,
                    ks_statistic=0.5,
                    ks_p_value=0.2,
                    alert=True,
                )
            ],
        ),
        prediction_drift=PredictionDriftResult(
            reference_mean=0.10,
            current_mean=0.20,
            mean_delta=0.10,
            reference_median=0.08,
            current_median=0.18,
            median_delta=0.10,
            reference_std=0.02,
            current_std=0.03,
            std_delta=0.01,
            reference_p90=0.15,
            current_p90=0.25,
            p90_delta=0.10,
            ks_statistic=0.4,
            ks_p_value=0.3,
            threshold=0.1,
            alert=True,
        ),
        high_risk_rate=HighRiskRateDriftResult(
            threshold=0.08,
            reference_high_risk_rate=0.10,
            current_high_risk_rate=0.25,
            rate_delta=0.15,
            rate_delta_threshold=0.05,
            alert=True,
        ),
    )


def _inputs() -> ReportInputs:
    return ReportInputs(
        model_comparison=_model_comparison(),
        selected_metrics={
            "roc_auc": 0.758,
            "pr_auc": 0.193,
            "precision": 0.171,
            "recall": 0.571,
            "f1": 0.264,
            "confusion_matrix": [[235, 58], [9, 12]],
        },
        root_cause_candidates=_root_cause(),
        sensitivity_summary=pd.DataFrame(
            {"metric": ["top_5_overlap"], "value": [0.8]}
        ),
        data_summary=DataSummary(
            train_rows=1253,
            test_rows=314,
            feature_count=590,
            train_fail_rate=0.066,
            test_fail_rate=0.067,
            missing_value_rate=0.04,
        ),
        monitoring_summary=_summary(),
    )


class TestRenderReports:
    def test_executive_summary_states_selection_protocol(self) -> None:
        report = render_executive_summary(_inputs())
        assert "random forest" in report.lower()
        assert "training-only 5-fold cross-validation PR-AUC" in report
        assert "not used to reopen model selection" in report

    def test_model_card_contains_threshold_and_confusion_matrix(self) -> None:
        report = render_model_card(_inputs())
        assert "0.080" in report
        assert "235" in report
        assert "58" in report

    def test_data_card_contains_static_dataset_caveat(self) -> None:
        report = render_data_card(_inputs())
        assert "static historical SECOM" in report
        assert "anonymous" in report.lower()

    def test_root_cause_report_keeps_triage_framing(self) -> None:
        report = render_root_cause_report(_inputs())
        assert "candidate triage" in report
        assert "does not prove physical causality" in report
        assert "sensor_059" in report


class TestWriteReportBundle:
    def test_writes_all_markdown_files(self, tmp_path: Path) -> None:
        bundle = ReportBundle(
            executive_summary="executive",
            model_card="model",
            data_card="data",
            root_cause_report="root",
        )

        write_report_bundle(bundle, tmp_path)

        assert (tmp_path / "executive_summary.md").read_text() == "executive"
        assert (tmp_path / "model_card.md").read_text() == "model"
        assert (tmp_path / "data_card.md").read_text() == "data"
        assert (tmp_path / "root_cause_report.md").read_text() == "root"


class TestValidation:
    def test_missing_model_comparison_column_raises_value_error(self) -> None:
        inputs = _inputs()
        inputs.model_comparison = inputs.model_comparison.drop(columns=["selected"])

        with pytest.raises(ValueError, match="model_comparison"):
            render_executive_summary(inputs)
```

- [ ] **Step 2: Run reporting tests to verify they fail**

Run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
pytest tests/test_reporting.py -v
```

Expected: import failure because `yield_risk.reporting` does not exist.

- [ ] **Step 3: Implement reporting utilities**

Create `src/yield_risk/reporting.py` with:

```python
"""Markdown report generation utilities for yield-risk artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from yield_risk.monitoring import MonitoringSummary


@dataclass
class DataSummary:
    """Summary of processed SECOM data used in reports.

    Attributes:
        train_rows: Number of rows in the processed training split.
        test_rows: Number of rows in the processed test split.
        feature_count: Number of sensor features.
        train_fail_rate: Fraction of failing wafers in the training split.
        test_fail_rate: Fraction of failing wafers in the test split.
        missing_value_rate: Overall missing-value rate across sensor columns.
    """

    train_rows: int
    test_rows: int
    feature_count: int
    train_fail_rate: float
    test_fail_rate: float
    missing_value_rate: float


@dataclass
class ReportInputs:
    """All structured inputs required to render reports.

    Attributes:
        model_comparison: Model comparison metrics.
        selected_metrics: Selected-model metric dictionary.
        root_cause_candidates: Ranked root-cause candidate table.
        sensitivity_summary: Root-cause model sensitivity summary table.
        data_summary: Processed data summary.
        monitoring_summary: Drift monitoring summary.
    """

    model_comparison: pd.DataFrame
    selected_metrics: dict[str, Any]
    root_cause_candidates: pd.DataFrame
    sensitivity_summary: pd.DataFrame
    data_summary: DataSummary
    monitoring_summary: MonitoringSummary


@dataclass
class ReportBundle:
    """Rendered markdown reports.

    Attributes:
        executive_summary: Executive summary markdown.
        model_card: Model card markdown.
        data_card: Data card markdown.
        root_cause_report: Root-cause report markdown.
    """

    executive_summary: str
    model_card: str
    data_card: str
    root_cause_report: str
```

Implement:

- `summarize_processed_data(train: pd.DataFrame, test: pd.DataFrame) -> DataSummary`
- `load_report_inputs(...paths..., monitoring_summary: MonitoringSummary) -> ReportInputs`
- `render_executive_summary(inputs: ReportInputs) -> str`
- `render_model_card(inputs: ReportInputs) -> str`
- `render_data_card(inputs: ReportInputs) -> str`
- `render_root_cause_report(inputs: ReportInputs) -> str`
- `render_report_bundle(inputs: ReportInputs) -> ReportBundle`
- `write_report_bundle(bundle: ReportBundle, output_dir: Path) -> None`

Validation details:

```python
_MODEL_COMPARISON_COLUMNS = {
    "model",
    "cv_pr_auc_mean",
    "test_pr_auc",
    "test_roc_auc",
    "test_recall",
    "test_precision",
    "opt_threshold",
    "expected_cost",
    "selected",
}
_ROOT_CAUSE_COLUMNS = {
    "sensor",
    "mean_abs_shap",
    "shap_lift",
    "spc_flag_rate",
    "composite_score",
}
```

Use a helper:

```python
def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
```

Markdown requirements:

- Executive summary includes model selection protocol, selected-model metrics,
  XGBoost comparison wording, monitoring signal overview, and limitations.
- Model card includes intended use, model family, selection protocol, metrics,
  threshold, confusion matrix, and monitoring hooks.
- Data card includes source, split sizes, fail rates, missingness summary, static
  historical data caveat, and anonymous sensor caveat.
- Root-cause report includes top candidates, signal definitions, XGBoost
  sensitivity evidence when available, and candidate-triage framing.

- [ ] **Step 4: Run reporting tests to verify they pass**

Run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
pytest tests/test_reporting.py -v
```

Expected: all tests in `tests/test_reporting.py` pass.

- [ ] **Step 5: Commit reporting utilities**

Run:

```powershell
git add src/yield_risk/reporting.py tests/test_reporting.py
git commit -m "feat: add markdown reporting utilities"
```

---

### Task 3: Report Generation Script

**Files:**
- Create: `scripts/generate_reports.py`
- Modify if needed: `tests/test_reporting.py`

- [ ] **Step 1: Add script-oriented test coverage**

Extend `tests/test_reporting.py` with a test for `render_report_bundle`:

```python
from yield_risk.reporting import render_report_bundle


class TestRenderReportBundle:
    def test_renders_all_report_sections(self) -> None:
        bundle = render_report_bundle(_inputs())

        assert "Executive Summary" in bundle.executive_summary
        assert "Model Card" in bundle.model_card
        assert "Data Card" in bundle.data_card
        assert "Root-Cause Candidate Report" in bundle.root_cause_report
```

- [ ] **Step 2: Run targeted reporting tests to verify the new test fails**

Run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
pytest tests/test_reporting.py::TestRenderReportBundle -v
```

Expected: failure until `render_report_bundle` exists or returns the expected sections.

- [ ] **Step 3: Implement `scripts/generate_reports.py`**

Create `scripts/generate_reports.py`:

```python
"""Generate markdown reports from current yield-risk artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from yield_risk.config import load_config
from yield_risk.monitoring import build_monitoring_summary
from yield_risk.reporting import (
    load_report_inputs,
    render_report_bundle,
    write_report_bundle,
)


def _split_reference_current(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a batch into deterministic reference/current halves.

    Args:
        df: Input batch.

    Returns:
        Tuple of reference and current DataFrames.

    Raises:
        ValueError: If the input has fewer than two rows.
    """
    if len(df) < 2:
        raise ValueError("At least two rows are required for monitoring reports.")
    midpoint = len(df) // 2
    return df.iloc[:midpoint].copy(), df.iloc[midpoint:].copy()


def generate_reports(config_path: Path = Path("configs/config.yaml")) -> None:
    """Generate markdown reports from configured artifacts.

    Args:
        config_path: Path to the main YAML config.
    """
    cfg = load_config(config_path)
    train = pd.read_csv(cfg.paths.processed_dir / "train.csv")
    test = pd.read_csv(cfg.paths.processed_dir / "test.csv")
    sensor_cols = [c for c in test.columns if c.startswith("sensor_")]
    if not sensor_cols:
        raise ValueError("No sensor_ columns found in processed test data.")

    reference, current = _split_reference_current(test)
    model = joblib.load(cfg.paths.models_dir / "selected_model.joblib")
    reference_scores = model.predict_proba(reference[sensor_cols])[:, 1]
    current_scores = model.predict_proba(current[sensor_cols])[:, 1]

    model_comparison = pd.read_csv(cfg.paths.reports_dir / "model_comparison.csv")
    selected = model_comparison.loc[model_comparison["selected"]].iloc[0]
    threshold = float(selected["opt_threshold"])

    monitoring_summary = build_monitoring_summary(
        reference,
        current,
        sensor_cols,
        reference_scores,
        current_scores,
        high_risk_threshold=threshold,
    )

    inputs = load_report_inputs(
        reports_dir=cfg.paths.reports_dir,
        train=train,
        test=test,
        monitoring_summary=monitoring_summary,
    )
    bundle = render_report_bundle(inputs)
    write_report_bundle(bundle, cfg.paths.reports_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate markdown reports.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to config.yaml.",
    )
    args = parser.parse_args()
    generate_reports(args.config)


if __name__ == "__main__":
    main()
```

Ensure `src/yield_risk/reporting.py` exports `load_report_inputs`,
`render_report_bundle`, and `write_report_bundle` with the signatures used by
the script above.

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
pytest tests/test_reporting.py -v
```

Expected: all reporting tests pass.

- [ ] **Step 5: Commit report generation script**

Run:

```powershell
git add scripts/generate_reports.py src/yield_risk/reporting.py tests/test_reporting.py
git commit -m "feat: add report generation script"
```

---

### Task 4: Empirical Monitoring Notebook

**Files:**
- Create: `notebooks/07_model_monitoring_drift_checks.ipynb`

- [ ] **Step 1: Create notebook JSON**

Create `notebooks/07_model_monitoring_drift_checks.ipynb` with markdown and code cells:

1. Title and objective.
2. Imports.
3. Load config, processed test data, and selected model.
4. Deterministic reference/current split.
5. Score both batches.
6. Build monitoring summary.
7. Display missingness drift alerts.
8. Display top feature drift rows.
9. Display prediction drift.
10. Display high-risk-rate drift.
11. Interpretation and limitations.

Use these code cells:

```python
from pathlib import Path

import joblib
import pandas as pd

from yield_risk.config import load_config
from yield_risk.monitoring import build_monitoring_summary
```

```python
cfg = load_config(Path("../configs/config.yaml"))
test = pd.read_csv(Path("../data/processed/test.csv"))
sensor_cols = [c for c in test.columns if c.startswith("sensor_")]
model = joblib.load(Path("../models/selected_model.joblib"))
len(test), len(sensor_cols)
```

```python
midpoint = len(test) // 2
reference = test.iloc[:midpoint].copy()
current = test.iloc[midpoint:].copy()

reference_scores = model.predict_proba(reference[sensor_cols])[:, 1]
current_scores = model.predict_proba(current[sensor_cols])[:, 1]

reference.shape, current.shape
```

```python
model_comparison = pd.read_csv(Path("../reports/model_comparison.csv"))
selected = model_comparison.loc[model_comparison["selected"]].iloc[0]
threshold = float(selected["opt_threshold"])
threshold
```

```python
summary = build_monitoring_summary(
    reference,
    current,
    sensor_cols,
    reference_scores,
    current_scores,
    high_risk_threshold=threshold,
)
```

```python
summary.missingness.to_frame().query("alert").head(10)
```

```python
summary.feature_drift.to_frame().head(10)
```

```python
pd.DataFrame([summary.prediction_drift.__dict__])
```

```python
pd.DataFrame([summary.high_risk_rate.__dict__])
```

- [ ] **Step 2: Validate notebook JSON parses**

Run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
python -m json.tool notebooks/07_model_monitoring_drift_checks.ipynb > $null
```

Expected: exit code 0.

- [ ] **Step 3: Commit notebook**

Run:

```powershell
git add notebooks/07_model_monitoring_drift_checks.ipynb
git commit -m "docs: add monitoring drift evidence notebook"
```

---

### Task 5: Generate Reports and Final Verification

**Files:**
- Generate: `reports/executive_summary.md`
- Generate: `reports/model_card.md`
- Generate: `reports/data_card.md`
- Generate: `reports/root_cause_report.md`
- Create: `tmp/tier4-pr-report.md`

- [ ] **Step 1: Run the report generation script**

Run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
python scripts/generate_reports.py
```

Expected: four markdown reports are written under `reports/`.

- [ ] **Step 2: Inspect generated reports for required framing**

Run:

```powershell
rg -n "training-only|not used to reopen|candidate triage|does not prove physical causality|static historical SECOM|anonymous" reports/*.md
```

Expected: matches across the generated reports.

- [ ] **Step 3: Run full verification**

Run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
ruff check src/ tests/
mypy src/yield_risk
pytest
```

Expected: all commands exit 0. If any command fails, fix the issue and rerun the
same command until it exits 0.

- [ ] **Step 4: Write PR-description style report**

Create `tmp/tier4-pr-report.md` with:

```markdown
# Tier 4 Monitoring and Reporting

## Summary

- Added deterministic monitoring utilities for missingness, feature distribution, prediction distribution, and high-risk-rate drift.
- Added reproducible markdown report generation from current model, data, root-cause, and monitoring artifacts.
- Added a report-style monitoring notebook that demonstrates the same library utilities on processed SECOM batches.

## Generated Reports

- `reports/executive_summary.md`
- `reports/model_card.md`
- `reports/data_card.md`
- `reports/root_cause_report.md`

## Verification

- `ruff check src/ tests/`
- `mypy src/yield_risk`
- `pytest`

## Notes

- The selected model remains random forest by training-only cross-validation PR-AUC.
- XGBoost held-out performance is reported as comparison evidence, not model reselection.
- Root-cause outputs are candidate triage signals and do not prove physical causality.
- Monitoring is a static-batch demonstration using available SECOM artifacts, not live telemetry.
```

Update the Verification section with the actual command results.

- [ ] **Step 5: Commit reports and final report**

Run:

```powershell
git add reports/executive_summary.md reports/model_card.md reports/data_card.md reports/root_cause_report.md tmp/tier4-pr-report.md
git commit -m "docs: add tier 4 generated reports"
```

---

## Final Review Checklist

- [ ] `src/yield_risk/monitoring.py` has public Google-style docstrings and fully annotated signatures.
- [ ] `src/yield_risk/reporting.py` has public Google-style docstrings and fully annotated signatures.
- [ ] Report text does not describe repository audience or positioning.
- [ ] Report text preserves the model-selection protocol.
- [ ] Notebook calls `yield_risk.monitoring` and does not duplicate drift calculations.
- [ ] `ruff check src/ tests/` passes.
- [ ] `mypy src/yield_risk` passes.
- [ ] `pytest` passes.
- [ ] The branch remains checked out for user review.
