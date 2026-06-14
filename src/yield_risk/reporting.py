"""Markdown reporting utilities for model, data, and root-cause artifacts."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import cast

import pandas as pd

from yield_risk.monitoring import MonitoringSummary

MODEL_COMPARISON_COLUMNS = {
    "model",
    "cv_pr_auc_mean",
    "test_pr_auc",
    "test_roc_auc",
    "test_recall",
    "test_precision",
    "frozen_threshold",
    "expected_cost",
    "selected",
}
ROOT_CAUSE_COLUMNS = [
    "sensor",
    "mean_abs_shap",
    "shap_lift",
    "spc_flag_rate",
    "composite_score",
]
REPORT_FILENAMES = {
    "executive_summary": "executive_summary.md",
    "model_card": "model_card.md",
    "data_card": "data_card.md",
    "root_cause_report": "root_cause_report.md",
}
CONFUSION_COUNT_KEYS = {
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
}


@dataclass
class DataSummary:
    """Summary statistics for processed train and test data.

    Attributes:
        train_rows: Number of rows in the training split.
        test_rows: Number of rows in the test split.
        sensor_count: Number of columns with the ``sensor_`` prefix.
        train_fail_rate: Mean failure-label rate in the training split.
        test_fail_rate: Mean failure-label rate in the test split.
        sensor_missing_rate: Overall missing-value rate across sensor columns.
    """

    train_rows: int
    test_rows: int
    sensor_count: int
    train_fail_rate: float
    test_fail_rate: float
    sensor_missing_rate: float


@dataclass
class ReportInputs:
    """Structured inputs used to render markdown reports.

    Attributes:
        model_comparison: Model comparison table from modeling artifacts.
        selected_model_metrics: Metrics for the selected operating point.
        root_cause_candidates: Ranked root-cause candidate table.
        sensitivity_summary: Optional root-cause model sensitivity table.
        data_summary: Processed data split and sensor summary.
        monitoring_summary: Batch monitoring summary.
    """

    model_comparison: pd.DataFrame
    selected_model_metrics: Mapping[str, object]
    root_cause_candidates: pd.DataFrame
    sensitivity_summary: pd.DataFrame
    data_summary: DataSummary
    monitoring_summary: MonitoringSummary


@dataclass
class ReportBundle:
    """Rendered markdown report bundle.

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


def summarize_processed_data(train: pd.DataFrame, test: pd.DataFrame) -> DataSummary:
    """Summarize processed train and test data for reports.

    Args:
        train: Training split containing a ``label`` column and sensor columns.
        test: Test split containing a ``label`` column and sensor columns.

    Returns:
        DataSummary with split sizes, fail rates, sensor count, and missingness.

    Raises:
        ValueError: If either split lacks ``label`` or shared ``sensor_`` columns.
    """
    _validate_processed_data(train, test)
    sensor_cols = _sensor_columns(train)
    combined_sensors = pd.concat([train[sensor_cols], test[sensor_cols]], axis=0)
    return DataSummary(
        train_rows=len(train),
        test_rows=len(test),
        sensor_count=len(sensor_cols),
        train_fail_rate=float(train["label"].mean()),
        test_fail_rate=float(test["label"].mean()),
        sensor_missing_rate=float(combined_sensors.isna().to_numpy().mean()),
    )


def load_report_inputs(
    reports_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    monitoring_summary: MonitoringSummary,
) -> ReportInputs:
    """Load report artifacts and combine them with runtime summaries.

    Args:
        reports_dir: Directory containing reporting artifact files.
        train: Training split used to compute the data summary.
        test: Test split used to compute the data summary.
        monitoring_summary: Batch monitoring summary to include.

    Returns:
        ReportInputs populated from artifact files and processed data.

    Raises:
        ValueError: If loaded artifacts, selected metrics, or data splits are
            invalid.
        FileNotFoundError: If a required artifact file is absent.
        json.JSONDecodeError: If selected metrics JSON is invalid.
    """
    model_comparison = pd.read_csv(reports_dir / "model_comparison.csv")
    selected_metrics_json = json.loads(
        (reports_dir / "selected_model_metrics.json").read_text(encoding="utf-8")
    )
    selected_metrics = _validate_selected_model_metrics(selected_metrics_json)
    root_cause_candidates = pd.read_csv(reports_dir / "root_cause_candidates.csv")
    # The current leak-free pipeline does not regenerate a challenger
    # sensitivity comparison. Any *_sensitivity_*.csv on disk is a stale
    # pre-rewrite artifact computed over a different (198-feature) feature
    # space, so we deliberately do NOT load it: mixing it with the
    # selected-model root-cause candidates (new feature space) would produce a
    # silently inconsistent report. The Sensitivity section degrades to an
    # explicit "not generated" note via ``_sensitivity_text``.
    sensitivity_summary = pd.DataFrame()
    inputs = ReportInputs(
        model_comparison=model_comparison,
        selected_model_metrics=selected_metrics,
        root_cause_candidates=root_cause_candidates,
        sensitivity_summary=sensitivity_summary,
        data_summary=summarize_processed_data(train, test),
        monitoring_summary=monitoring_summary,
    )
    _validate_report_inputs(inputs)
    return inputs


def render_executive_summary(inputs: ReportInputs) -> str:
    """Render the executive summary markdown.

    Args:
        inputs: Structured report inputs.

    Returns:
        Executive summary markdown.

    Raises:
        ValueError: If required model comparison columns are absent.
    """
    selected = _selected_model_row(inputs)
    xgboost = _comparison_row(inputs.model_comparison, "xgboost")
    selected_name = _display_model_name(str(selected["model"]))
    threshold = _selected_threshold(inputs, selected)
    top = _top_root_cause_candidate(inputs)
    missing_alerts = _missingness_alert_count(inputs.monitoring_summary)
    feature_alerts = _feature_alert_count(inputs.monitoring_summary)
    top_text = (
        "No root-cause candidate available yet."
        if top is None
        else (
            f"`{top['sensor']}` "
            f"(composite score {_format_float(top['composite_score'])})"
        )
    )
    xgboost_text = ""
    if xgboost is not None:
        xgboost_text = (
            "\n"
            f"- XGBoost held-out challenger PR-AUC: "
            f"{_format_float(xgboost['test_pr_auc'])} (reference comparison "
            "only; no cross-model sensitivity overlap is produced for this run)."
        )

    return (
        "# Executive Summary\n\n"
        f"- Selected model: {selected_name}, chosen by training-only 5-fold "
        f"cross-validation PR-AUC ({_format_float(selected['cv_pr_auc_mean'])}).\n"
        f"- Held-out PR-AUC: {_format_float(selected['test_pr_auc'])}.\n"
        f"- Operating threshold: {threshold:.3f}; expected cost: "
        f"{_format_float(selected['expected_cost'])}.\n"
        f"- Top sensor to inspect first: {top_text}."
        f"{xgboost_text}\n"
        f"- Monitoring flags: {missing_alerts} missingness alerts, "
        f"{feature_alerts} feature drift alerts, prediction drift "
        f"{_format_alert(inputs.monitoring_summary.predictions.alert)}, "
        f"high-risk-rate drift "
        f"{_format_alert(inputs.monitoring_summary.high_risk_rate.alert)}."
    )


def render_model_card(inputs: ReportInputs) -> str:
    """Render the model card markdown.

    Args:
        inputs: Structured report inputs.

    Returns:
        Model card markdown.

    Raises:
        ValueError: If required model comparison columns are absent.
    """
    selected = _selected_model_row(inputs)
    threshold = _selected_threshold(inputs, selected)
    true_positive = _metric_int(inputs.selected_model_metrics, "true_positive")
    false_positive = _metric_int(inputs.selected_model_metrics, "false_positive")
    true_negative = _metric_int(inputs.selected_model_metrics, "true_negative")
    false_negative = _metric_int(inputs.selected_model_metrics, "false_negative")
    model_name = _display_model_name(str(selected["model"]))

    return (
        "# Model Card\n\n"
        "## Intended Use\n\n"
        "This model scores wafer-level process records for early yield-risk "
        "review and threshold-based hold/release simulation.\n\n"
        "Designed output: risk score, threshold flag, and sensor ranking for "
        "engineering review.\n\n"
        "## Model Summary\n\n"
        f"- Model family: {model_name}\n"
        f"- Selected threshold: {threshold:.3f}\n"
        f"- Expected cost at selected threshold: "
        f"{_format_float(selected['expected_cost'])}\n"
        f"- Test PR-AUC: {_format_float(selected['test_pr_auc'])}\n"
        f"- Test ROC-AUC: {_format_float(selected['test_roc_auc'])}\n"
        f"- Test recall: {_format_float(selected['test_recall'])}\n"
        f"- Test precision: {_format_float(selected['test_precision'])}\n\n"
        "## Selection Protocol\n\n"
        f"Selection is fixed before held-out evaluation: {model_name} by the "
        "highest mean training-only 5-fold cross-validation PR-AUC "
        "(`cv_pr_auc_mean`). Held-out test metrics measure generalization and "
        "threshold performance.\n\n"
        "## Confusion Matrix\n\n"
        f"- True positives: {true_positive}\n"
        f"- False positives: {false_positive}\n"
        f"- True negatives: {true_negative}\n"
        f"- False negatives: {false_negative}\n\n"
        "## Monitoring Hooks\n\n"
        "- Available checks: missingness drift, feature distribution drift, "
        "prediction distribution drift, and high-risk-rate drift.\n"
        "- The checks compare reference and current batches from the processed "
        "dataset and surface alert counts and drift flags for engineering "
        "review.\n"
    )


def render_data_card(inputs: ReportInputs) -> str:
    """Render the data card markdown.

    Args:
        inputs: Structured report inputs.

    Returns:
        Data card markdown.
    """
    data = inputs.data_summary
    missing_alerts = sum(
        result.alert for result in inputs.monitoring_summary.missingness.results
    )
    feature_alerts = sum(
        result.alert for result in inputs.monitoring_summary.features.results
    )
    return (
        "# Data Card\n\n"
        "## Dataset Facts\n\n"
        "- Source: public historical SECOM benchmark with static wafer records "
        "rather than a complete fab execution trace.\n"
        "- Label: binary pass/fail; it does not identify the failure mode.\n"
        "- Sensors: anonymous sensor identifiers; process step, tool, chamber, "
        "recipe, lot, and maintenance metadata are not included.\n"
        f"- Split: {data.train_rows} training rows "
        f"(fail rate {data.train_fail_rate:.3f}) and {data.test_rows} test rows "
        f"(fail rate {data.test_fail_rate:.3f}).\n"
        f"- Sensor matrix: {data.sensor_count} raw sensor columns with overall "
        f"missing-value rate {data.sensor_missing_rate:.3f}. The train/test CSVs "
        "hold the unprocessed sensor readings (all columns, missing values "
        "intact, no feature selection). Preprocessing — missing/CV/"
        "correlation filtering plus median imputation — is fit per "
        "cross-validation fold inside the model pipeline to avoid leakage, not "
        "applied before the split.\n\n"
        "## Batch Monitoring Checks\n\n"
        f"- Missingness alerts: {missing_alerts}\n"
        f"- Feature drift alerts: {feature_alerts}\n"
        f"- Prediction drift alert: {inputs.monitoring_summary.predictions.alert}\n"
        f"- High-risk-rate alert: {inputs.monitoring_summary.high_risk_rate.alert}\n"
    )


def render_root_cause_report(inputs: ReportInputs) -> str:
    """Render the root-cause candidate report markdown.

    Args:
        inputs: Structured report inputs.

    Returns:
        Root-cause candidate report markdown.

    Raises:
        ValueError: If required root-cause candidate columns are absent.
    """
    _validate_columns(
        inputs.root_cause_candidates,
        set(ROOT_CAUSE_COLUMNS),
        "root_cause_candidates",
    )
    ranked = inputs.root_cause_candidates.sort_values(
        "composite_score",
        ascending=False,
    )
    top = ranked.iloc[0] if not ranked.empty else None
    lead_text = (
        "No root-cause candidates were available."
        if top is None
        else (
            f"`{top['sensor']}` is the top root-cause candidate "
            f"(composite score {_format_float(top['composite_score'])})."
        )
    )
    evidence_text = (
        ""
        if top is None
        else (
            "Why it leads: "
            f"mean absolute SHAP {_format_float(top['mean_abs_shap'])}, "
            f"fail/pass lift {_format_float(top['shap_lift'])}, "
            f"SPC flag rate {_format_float(top['spc_flag_rate'])}, and "
            f"composite score {_format_float(top['composite_score'])}."
        )
    )
    top_sensor = None if top is None else str(top["sensor"])
    sensitivity_text = _sensitivity_text(inputs.sensitivity_summary, top_sensor)

    return (
        "# Root-Cause Candidate Report\n\n"
        f"{lead_text}\n\n"
        f"{evidence_text}\n\n"
        "Next engineering action: map the top sensor IDs to process step, tool, "
        "chamber, recipe, lot, and maintenance context before changing process "
        "settings. The SECOM sensor IDs are anonymized, so this metadata join "
        "connects the ranking to fab action.\n\n"
        "## Top Candidates\n\n"
        f"{_markdown_table(ranked.head(10), ROOT_CAUSE_COLUMNS)}\n\n"
        f"{sensitivity_text}"
    )


def render_report_bundle(inputs: ReportInputs) -> ReportBundle:
    """Render all markdown reports.

    Args:
        inputs: Structured report inputs.

    Returns:
        ReportBundle containing all markdown report strings.

    Raises:
        ValueError: If report inputs fail required validation.
    """
    _validate_report_inputs(inputs)
    return ReportBundle(
        executive_summary=render_executive_summary(inputs),
        model_card=render_model_card(inputs),
        data_card=render_data_card(inputs),
        root_cause_report=render_root_cause_report(inputs),
    )


def write_report_bundle(bundle: ReportBundle, output_dir: Path) -> None:
    """Write a markdown report bundle to disk.

    Args:
        bundle: Rendered report bundle.
        output_dir: Directory where markdown files will be written.

    Returns:
        None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for field_name, filename in REPORT_FILENAMES.items():
        text = getattr(bundle, field_name)
        (output_dir / filename).write_text(text, encoding="utf-8")


def _validate_report_inputs(inputs: ReportInputs) -> None:
    _validate_columns(
        inputs.model_comparison,
        MODEL_COMPARISON_COLUMNS,
        "model_comparison",
    )
    _validate_columns(
        inputs.root_cause_candidates,
        set(ROOT_CAUSE_COLUMNS),
        "root_cause_candidates",
    )
    _validate_selected_model_metrics(inputs.selected_model_metrics)


def _validate_selected_model_metrics(metrics: object) -> Mapping[str, object]:
    if not isinstance(metrics, Mapping):
        raise ValueError("selected_model_metrics must be a mapping.")
    has_explicit_counts = CONFUSION_COUNT_KEYS.issubset(metrics.keys())
    confusion_matrix = metrics.get("confusion_matrix")
    if has_explicit_counts:
        for key in CONFUSION_COUNT_KEYS:
            _coerce_selected_metric_count(metrics[key])
        return metrics
    if _is_confusion_matrix(confusion_matrix):
        return metrics
    raise ValueError(
        "selected_model_metrics must include explicit confusion counts or a "
        "valid 2x2 confusion_matrix."
    )


def _validate_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    artifact_name: str,
) -> None:
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{artifact_name} missing required columns: {missing}")


def _validate_processed_data(train: pd.DataFrame, test: pd.DataFrame) -> None:
    missing_label = [
        name
        for name, frame in (("train", train), ("test", test))
        if "label" not in frame.columns
    ]
    if missing_label:
        raise ValueError(f"processed data missing label column in: {missing_label}")
    train_sensors = _sensor_columns(train)
    test_sensors = _sensor_columns(test)
    if not train_sensors:
        raise ValueError("processed data must include sensor_ columns.")
    missing_test = sorted(set(train_sensors).difference(test_sensors))
    missing_train = sorted(set(test_sensors).difference(train_sensors))
    if missing_test or missing_train:
        raise ValueError(
            "processed data sensor columns must match across splits: "
            f"missing in test={missing_test}, missing in train={missing_train}"
        )


def _sensor_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("sensor_")]


def _selected_model_row(inputs: ReportInputs) -> pd.Series:
    _validate_columns(
        inputs.model_comparison,
        MODEL_COMPARISON_COLUMNS,
        "model_comparison",
    )
    selected_rows = inputs.model_comparison[
        inputs.model_comparison["selected"].astype(bool)
    ]
    if len(selected_rows) != 1:
        raise ValueError("model_comparison must contain exactly one selected row.")
    return selected_rows.iloc[0]


def _comparison_row(frame: pd.DataFrame, model: str) -> pd.Series | None:
    matches = frame[frame["model"].astype(str).str.lower() == model.lower()]
    if matches.empty:
        return None
    return matches.iloc[0]


def _top_root_cause_candidate(inputs: ReportInputs) -> pd.Series | None:
    _validate_columns(
        inputs.root_cause_candidates,
        set(ROOT_CAUSE_COLUMNS),
        "root_cause_candidates",
    )
    if inputs.root_cause_candidates.empty:
        return None
    ranked = inputs.root_cause_candidates.sort_values(
        "composite_score",
        ascending=False,
    )
    return ranked.iloc[0]


def _missingness_alert_count(summary: MonitoringSummary) -> int:
    return sum(result.alert for result in summary.missingness.results)


def _feature_alert_count(summary: MonitoringSummary) -> int:
    return sum(result.alert for result in summary.features.results)


def _selected_threshold(inputs: ReportInputs, selected: pd.Series) -> float:
    metric_threshold = inputs.selected_model_metrics.get("threshold")
    if metric_threshold is not None:
        return _coerce_float(metric_threshold)
    return float(selected["frozen_threshold"])


def _metric_int(metrics: Mapping[str, object], key: str) -> int:
    _validate_selected_model_metrics(metrics)
    value = metrics.get(key)
    if value is not None:
        return _coerce_int(value)
    confusion_matrix = metrics.get("confusion_matrix")
    if _is_confusion_matrix(confusion_matrix):
        return _metric_from_confusion_matrix(
            cast("list[list[object]]", confusion_matrix),
            key,
        )
    return 0


def _metric_from_confusion_matrix(
    confusion_matrix: list[list[object]],
    key: str,
) -> int:
    value_by_key = {
        "true_negative": confusion_matrix[0][0],
        "false_positive": confusion_matrix[0][1],
        "false_negative": confusion_matrix[1][0],
        "true_positive": confusion_matrix[1][1],
    }
    return _coerce_int(value_by_key[key])


def _format_float(value: object) -> str:
    return f"{_coerce_float(value):.3f}"


def _format_alert(value: object) -> str:
    return "triggered" if bool(value) else "clear"


def _coerce_float(value: object) -> float:
    if isinstance(value, Real):
        return float(value)
    return float(str(value))


def _coerce_int(value: object) -> int:
    if isinstance(value, Real):
        return int(float(value))
    return int(str(value))


def _coerce_selected_metric_count(value: object) -> int:
    try:
        return _coerce_int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "selected_model_metrics confusion counts must be coercible to ints."
        ) from exc


def _is_confusion_matrix(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in value)
    )


def _display_model_name(model: str) -> str:
    if model.lower() == "random_forest":
        return "random forest"
    if model.lower() == "xgboost":
        return "XGBoost"
    return model.replace("_", " ")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    existing_columns = [column for column in columns if column in frame.columns]
    header = "| " + " | ".join(existing_columns) + " |"
    separator = "| " + " | ".join("---" for _ in existing_columns) + " |"
    rows = [
        "| "
        + " | ".join(_format_table_value(row[column]) for column in existing_columns)
        + " |"
        for _, row in frame[existing_columns].iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _format_table_value(value: object) -> str:
    if isinstance(value, Real):
        return f"{value:.3f}"
    return str(value)


def _sensitivity_text(
    sensitivity_summary: pd.DataFrame,
    top_sensor: str | None = None,
) -> str:
    if sensitivity_summary.empty:
        return (
            "## Sensitivity\n\n"
            "No challenger sensitivity comparison was generated for this run. "
            "The current pipeline ranks root-cause candidates for the selected "
            "model only; a cross-model (selected vs. challenger) sensitivity "
            "overlap is not produced, so this section is intentionally omitted "
            "rather than populated from a prior run's artifacts."
        )
    if top_sensor is not None and "overlap_sensors" in sensitivity_summary.columns:
        overlap_values = sensitivity_summary["overlap_sensors"].astype(str)
        if overlap_values.str.contains(top_sensor, regex=False).any():
            intro = (
                f"XGBoost sensitivity overlap keeps `{top_sensor}` in the "
                "top-k overlap sets shown below."
            )
        else:
            intro = (
                "XGBoost sensitivity overlap compares selected-model and "
                "challenger sensor rankings."
            )
    else:
        intro = (
            "XGBoost sensitivity overlap compares selected-model and "
            "challenger sensor rankings."
        )
    return (
        "## Sensitivity\n\n"
        f"{intro}\n\n"
        f"{_markdown_table(sensitivity_summary, list(sensitivity_summary.columns))}"
    )
