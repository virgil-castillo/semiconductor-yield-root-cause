"""Tests for markdown reporting utilities."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_reports import _split_reference_current
from yield_risk.monitoring import (
    FeatureDriftResult,
    FeatureDriftSummary,
    HighRiskRateDriftResult,
    MissingnessDriftResult,
    MissingnessDriftSummary,
    MonitoringSummary,
    PredictionDriftResult,
)
from yield_risk.reporting import (
    DataSummary,
    ReportInputs,
    load_report_inputs,
    render_data_card,
    render_executive_summary,
    render_model_card,
    render_report_bundle,
    render_root_cause_report,
    summarize_processed_data,
    write_report_bundle,
)


@pytest.fixture
def model_comparison() -> pd.DataFrame:
    """Return model comparison rows with random forest selected."""
    return pd.DataFrame(
        [
            {
                "model": "random_forest",
                "cv_pr_auc_mean": 0.42,
                "test_pr_auc": 0.38,
                "test_roc_auc": 0.81,
                "test_recall": 0.72,
                "test_precision": 0.34,
                "frozen_threshold": 0.27,
                "expected_cost": 11.5,
                "selected": True,
            },
            {
                "model": "xgboost",
                "cv_pr_auc_mean": 0.39,
                "test_pr_auc": 0.41,
                "test_roc_auc": 0.84,
                "test_recall": 0.77,
                "test_precision": 0.31,
                "frozen_threshold": 0.33,
                "expected_cost": 12.8,
                "selected": False,
            },
        ]
    )


@pytest.fixture
def selected_metrics() -> dict[str, float | int | str]:
    """Return selected model metrics including confusion matrix values."""
    return {
        "model": "random_forest",
        "threshold": 0.27,
        "true_positive": 18,
        "false_positive": 7,
        "true_negative": 92,
        "false_negative": 5,
    }


@pytest.fixture
def root_cause_candidates() -> pd.DataFrame:
    """Return root-cause candidate rows."""
    return pd.DataFrame(
        [
            {
                "sensor": "sensor_011",
                "mean_abs_shap": 0.61,
                "shap_lift": 2.4,
                "spc_flag_rate": 0.18,
                "composite_score": 0.89,
            },
            {
                "sensor": "sensor_002",
                "mean_abs_shap": 0.38,
                "shap_lift": 1.6,
                "spc_flag_rate": 0.12,
                "composite_score": 0.54,
            },
        ]
    )


@pytest.fixture
def sensitivity_summary() -> pd.DataFrame:
    """Return model sensitivity summary rows."""
    return pd.DataFrame(
        [
            {"sensor": "sensor_011", "random_forest_rank": 1, "xgboost_rank": 2},
            {"sensor": "sensor_002", "random_forest_rank": 2, "xgboost_rank": 1},
        ]
    )


@pytest.fixture
def data_summary() -> DataSummary:
    """Return processed data summary."""
    return DataSummary(
        train_rows=100,
        test_rows=50,
        sensor_count=3,
        train_fail_rate=0.08,
        test_fail_rate=0.1,
        sensor_missing_rate=0.12,
    )


@pytest.fixture
def monitoring_summary() -> MonitoringSummary:
    """Return a batch monitoring summary."""
    return MonitoringSummary(
        missingness=MissingnessDriftSummary(
            results=[
                MissingnessDriftResult(
                    feature="sensor_011",
                    reference_missing_rate=0.1,
                    current_missing_rate=0.2,
                    missing_rate_delta=0.1,
                    alert=True,
                )
            ],
            threshold=0.05,
        ),
        features=FeatureDriftSummary(
            results=[
                FeatureDriftResult(
                    feature="sensor_011",
                    reference_mean=0.0,
                    current_mean=0.4,
                    mean_delta=0.4,
                    reference_std=1.0,
                    current_std=1.1,
                    std_delta=0.1,
                    reference_median=0.0,
                    current_median=0.3,
                    median_delta=0.3,
                    ks_statistic=0.2,
                    ks_p_value=0.04,
                    alert=True,
                )
            ],
            threshold=0.1,
        ),
        predictions=PredictionDriftResult(
            reference_mean=0.2,
            current_mean=0.3,
            mean_delta=0.1,
            reference_median=0.15,
            current_median=0.24,
            median_delta=0.09,
            reference_std=0.05,
            current_std=0.08,
            std_delta=0.03,
            reference_p90=0.31,
            current_p90=0.45,
            p90_delta=0.14,
            ks_statistic=0.22,
            ks_p_value=0.03,
            alert=True,
        ),
        high_risk_rate=HighRiskRateDriftResult(
            threshold=0.27,
            reference_rate=0.12,
            current_rate=0.2,
            rate_delta=0.08,
            rate_delta_threshold=0.05,
            alert=True,
        ),
    )


@pytest.fixture
def report_inputs(
    model_comparison: pd.DataFrame,
    selected_metrics: dict[str, float | int | str],
    root_cause_candidates: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    data_summary: DataSummary,
    monitoring_summary: MonitoringSummary,
) -> ReportInputs:
    """Return complete report inputs."""
    return ReportInputs(
        model_comparison=model_comparison,
        selected_model_metrics=selected_metrics,
        root_cause_candidates=root_cause_candidates,
        sensitivity_summary=sensitivity_summary,
        data_summary=data_summary,
        monitoring_summary=monitoring_summary,
    )


def test_render_executive_summary_preserves_selection_protocol(
    report_inputs: ReportInputs,
) -> None:
    """Executive summary states model selection and the XGBoost comparison."""
    summary = render_executive_summary(report_inputs)

    assert "training-only 5-fold cross-validation PR-AUC" in summary
    assert "random forest" in summary.lower()
    assert "XGBoost" in summary
    assert "not used to reopen model selection" not in summary
    # Bug B: the XGBoost line must not claim a sensitivity-comparator role,
    # which the root-cause report explicitly states is not produced.
    assert "sensitivity comparator" not in summary
    assert "no cross-model sensitivity overlap is produced" in summary


def test_render_model_card_includes_threshold_and_confusion_matrix(
    report_inputs: ReportInputs,
) -> None:
    """Model card includes selected operating threshold and confusion matrix."""
    model_card = render_model_card(report_inputs)

    assert "## Intended Use" in model_card
    assert "## Selection Protocol" in model_card
    assert "training-only 5-fold cross-validation PR-AUC" in model_card
    assert "highest mean" in model_card
    assert "cv_pr_auc_mean" in model_card
    assert "Selected threshold: 0.270" in model_card
    assert "True positives: 18" in model_card
    assert "False positives: 7" in model_card
    assert "True negatives: 92" in model_card
    assert "False negatives: 5" in model_card
    assert "## Monitoring Hooks" in model_card
    assert "static-batch demonstration" not in model_card
    assert "not a substitute" not in model_card


def test_render_model_card_selection_protocol_uses_selected_model_name(
    report_inputs: ReportInputs,
) -> None:
    """Selection Protocol must name the selected model, not a hardcoded family.

    When selection picks logistic_regression (the leak-free pipeline's choice),
    the Selection Protocol prose must say "logistic regression" and must not
    hardcode "random forest".
    """
    comparison = report_inputs.model_comparison
    comparison.loc[comparison["model"] == "random_forest", "selected"] = False
    comparison.loc[len(comparison)] = {
        "model": "logistic_regression",
        "cv_pr_auc_mean": 0.08,
        "test_pr_auc": 0.15,
        "test_roc_auc": 0.72,
        "test_recall": 0.35,
        "test_precision": 0.21,
        "frozen_threshold": 0.47,
        "expected_cost": 132.0,
        "selected": True,
    }

    model_card = render_model_card(report_inputs)

    protocol = model_card.split("## Selection Protocol", 1)[1].split("##", 1)[0]
    assert "logistic regression" in protocol
    assert "random forest" not in protocol


def test_render_data_card_states_dataset_facts(
    report_inputs: ReportInputs,
) -> None:
    """Data card states SECOM source and anonymous sensors as dataset facts."""
    data_card = render_data_card(report_inputs)

    assert "## Dataset Facts" in data_card
    assert "SECOM" in data_card
    assert "anonymous sensor" in data_card
    assert "not live telemetry" not in data_card
    assert "caveat" not in data_card.lower()


def test_render_data_card_describes_raw_split_not_preprocessed(
    report_inputs: ReportInputs,
) -> None:
    """Data card must state the split is raw and preprocessing is per-fold.

    Under the leak-free pipeline, train.csv/test.csv hold unprocessed sensor
    readings; preprocessing happens per CV fold inside the model. The old
    "after preprocessing" phrasing was misleading and must be gone.
    """
    data_card = render_data_card(report_inputs)

    assert "raw sensor columns" in data_card
    assert "per cross-validation fold" in data_card
    assert "after preprocessing" not in data_card


def test_render_root_cause_report_leads_with_top_sensor_and_next_action(
    report_inputs: ReportInputs,
) -> None:
    """Root-cause report leads with the top sensor and a concrete next action."""
    root_cause_report = render_root_cause_report(report_inputs)

    assert "top root-cause candidate" in root_cause_report
    assert "sensor_011" in root_cause_report
    assert "Next engineering action" in root_cause_report
    assert "candidate triage" not in root_cause_report
    assert "does not prove physical causality" not in root_cause_report


def test_write_report_bundle_writes_all_expected_markdown_files(
    report_inputs: ReportInputs,
    tmp_path: Path,
) -> None:
    """Report bundle writer creates the expected markdown files."""
    bundle = render_report_bundle(report_inputs)

    write_report_bundle(bundle, tmp_path)

    assert (tmp_path / "executive_summary.md").read_text(encoding="utf-8")
    assert (tmp_path / "model_card.md").read_text(encoding="utf-8")
    assert (tmp_path / "data_card.md").read_text(encoding="utf-8")
    assert (tmp_path / "root_cause_report.md").read_text(encoding="utf-8")


def test_render_report_bundle_returns_all_four_sections(
    report_inputs: ReportInputs,
) -> None:
    """Report bundle renderer returns every markdown section."""
    bundle = render_report_bundle(report_inputs)

    assert bundle.executive_summary.startswith("# Executive Summary")
    assert bundle.model_card.startswith("# Model Card")
    assert bundle.data_card.startswith("# Data Card")
    assert bundle.root_cause_report.startswith("# Root-Cause Candidate Report")


def test_split_reference_current_uses_deterministic_halves() -> None:
    """Report batches split into first-half reference and second-half current."""
    data = pd.DataFrame({"sensor_001": [10, 20, 30, 40, 50], "label": [0, 1, 0, 1, 0]})

    reference, current = _split_reference_current(data)

    assert reference["sensor_001"].tolist() == [10, 20]
    assert current["sensor_001"].tolist() == [30, 40, 50]


def test_split_reference_current_rejects_too_few_rows() -> None:
    """Report batches need at least one row per side."""
    data = pd.DataFrame({"sensor_001": [10], "label": [0]})

    with pytest.raises(ValueError, match="at least 2 rows"):
        _split_reference_current(data)


def test_missing_model_comparison_columns_raise_value_error(
    report_inputs: ReportInputs,
) -> None:
    """Model comparison validation reports the missing artifact name."""
    report_inputs.model_comparison.drop(columns=["selected"], inplace=True)

    with pytest.raises(ValueError, match="model_comparison"):
        render_executive_summary(report_inputs)


def test_load_report_inputs_rejects_non_mapping_selected_metrics(
    tmp_path: Path,
    model_comparison: pd.DataFrame,
    root_cause_candidates: pd.DataFrame,
    monitoring_summary: MonitoringSummary,
) -> None:
    """Selected metrics JSON must decode to a mapping."""
    model_comparison.to_csv(tmp_path / "model_comparison.csv", index=False)
    root_cause_candidates.to_csv(tmp_path / "root_cause_candidates.csv", index=False)
    (tmp_path / "selected_model_metrics.json").write_text(
        json.dumps(["not", "a", "mapping"]),
        encoding="utf-8",
    )
    train = pd.DataFrame({"sensor_001": [1.0, 2.0], "label": [0, 1]})
    test = pd.DataFrame({"sensor_001": [3.0], "label": [0]})

    with pytest.raises(ValueError, match="selected_model_metrics"):
        load_report_inputs(tmp_path, train, test, monitoring_summary)


def test_render_model_card_rejects_missing_confusion_counts(
    report_inputs: ReportInputs,
) -> None:
    """Selected metrics must contain explicit counts or a 2x2 matrix."""
    report_inputs.selected_model_metrics = {"model": "random_forest", "threshold": 0.27}

    with pytest.raises(ValueError, match="selected_model_metrics"):
        render_model_card(report_inputs)


def test_render_model_card_rejects_null_explicit_confusion_count(
    report_inputs: ReportInputs,
) -> None:
    """Explicit selected metrics counts must not be null."""
    report_inputs.selected_model_metrics = {
        "model": "random_forest",
        "threshold": 0.27,
        "true_positive": None,
        "false_positive": 7,
        "true_negative": 92,
        "false_negative": 5,
    }

    with pytest.raises(ValueError, match="selected_model_metrics"):
        render_model_card(report_inputs)


def test_multiple_selected_model_rows_raise_value_error(
    report_inputs: ReportInputs,
) -> None:
    """Model comparison must contain exactly one selected row."""
    report_inputs.model_comparison.loc[:, "selected"] = True

    with pytest.raises(ValueError, match="model_comparison"):
        render_executive_summary(report_inputs)


def test_load_report_inputs_ignores_stale_sensitivity_file(
    tmp_path: Path,
    model_comparison: pd.DataFrame,
    root_cause_candidates: pd.DataFrame,
    selected_metrics: dict[str, float | int | str],
    monitoring_summary: MonitoringSummary,
) -> None:
    """A stale on-disk sensitivity CSV must NOT be loaded into report inputs.

    The current pipeline does not regenerate challenger sensitivity data, so any
    such file is a pre-rewrite artifact over an old feature space. Loading it
    would mix feature spaces in the report.
    """
    model_comparison.to_csv(tmp_path / "model_comparison.csv", index=False)
    root_cause_candidates.to_csv(tmp_path / "root_cause_candidates.csv", index=False)
    (tmp_path / "selected_model_metrics.json").write_text(
        json.dumps(selected_metrics), encoding="utf-8"
    )
    # Plant a stale sensitivity artifact on disk.
    pd.DataFrame({"top_n": [5], "overlap_count": [4]}).to_csv(
        tmp_path / "root_cause_model_sensitivity_summary.csv", index=False
    )
    train = pd.DataFrame({"sensor_001": [1.0, 2.0], "label": [0, 1]})
    test = pd.DataFrame({"sensor_001": [3.0, 4.0], "label": [0, 1]})

    inputs = load_report_inputs(tmp_path, train, test, monitoring_summary)

    assert inputs.sensitivity_summary.empty


def test_root_cause_report_marks_sensitivity_unavailable_when_absent(
    report_inputs: ReportInputs,
) -> None:
    """With no sensitivity data, the report states the section is not generated."""
    report_inputs.sensitivity_summary = pd.DataFrame()

    root_cause_report = render_root_cause_report(report_inputs)

    assert "## Sensitivity" in root_cause_report
    assert "No challenger sensitivity comparison was generated" in root_cause_report


def test_summarize_processed_data_rejects_test_only_sensor_columns() -> None:
    """Processed data validation rejects sensor schema skew in either direction."""
    train = pd.DataFrame({"sensor_001": [1.0, 2.0], "label": [0, 1]})
    test = pd.DataFrame(
        {"sensor_001": [3.0], "sensor_999": [4.0], "label": [0]}
    )

    with pytest.raises(ValueError, match="sensor columns"):
        summarize_processed_data(train, test)
