# Spec 01 — Preprocessing Script, Baseline Model, Evaluation, Feature Importance

## 1. Overview

Build the end-to-end baseline pipeline: a preprocessing script that persists train/test splits, a logistic regression baseline trained with class-weight balancing, an evaluation module that computes metrics and saves diagnostic plots, and a feature importance module that ranks sensors by logistic regression coefficient magnitude. All four stages run as sequential CLI scripts with no human gates.

## 2. File Manifest

### Modify

| File | Change |
|---|---|
| `src/yield_risk/preprocess.py` | Add `run_preprocessing(df, run_cfg) -> (train, test)` orchestrator |
| `tests/test_preprocess.py` | Add `TestRunPreprocessing` class (4 tests) |

### Create

| File | Purpose |
|---|---|
| `scripts/preprocess_data.py` | CLI: load raw data, validate, preprocess, save train/test CSVs |
| `src/yield_risk/model.py` | `build_baseline_pipeline`, `train_model`, `cross_validate_model` |
| `scripts/train_baseline.py` | CLI: train baseline LR, print CV scores, save model |
| `src/yield_risk/evaluate.py` | `ClassificationMetrics`, `compute_metrics`, `format_report`, plot functions |
| `scripts/evaluate_model.py` | CLI: load model + test set, compute metrics, save JSON + plots |
| `src/yield_risk/importance.py` | `extract_lr_coefficients`, `plot_top_features` |
| `scripts/feature_importance.py` | CLI: load model + test set, extract coefficients, save CSV + plot |
| `tests/test_model.py` | 8 tests for model.py |
| `tests/test_evaluate.py` | 10 tests for evaluate.py |
| `tests/test_importance.py` | 4 tests for importance.py |
| `notebooks/03_baseline_results.ipynb` | Read-only notebook displaying saved artifacts |

---

## 3. Stage A — Preprocessing Script

### 3A.1 Library addition: `src/yield_risk/preprocess.py`

Add one function to the existing module. Do not modify existing functions.

```python
from yield_risk.config import RunConfig

def run_preprocessing(
    df: pd.DataFrame,
    run_cfg: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply feature selection, imputation, and stratified train/test split.

    Pipeline order:
    1. drop_high_missing(df, run_cfg.missing_threshold)
    2. impute_median(result)
    3. drop_low_variance(result, run_cfg.variance_threshold)
    4. drop_high_correlation(result, run_cfg.correlation_threshold)
    5. split_stratified(result, run_cfg.test_size, run_cfg.random_seed)

    Args:
        df: Raw SECOM DataFrame (output of load_secom, already validated).
        run_cfg: RunConfig with thresholds and split parameters.

    Returns:
        Tuple of (train_df, test_df). Both contain sensor columns, label,
        and timestamp. No NaN values in sensor columns.
    """
```

Import `RunConfig` from `yield_risk.config`. Add it to the existing imports at the top of the file.

### 3A.2 CLI script: `scripts/preprocess_data.py`

```python
def main() -> None:
    """Load raw SECOM data, preprocess, and save train/test splits."""
```

Flow:
1. `cfg = load_config()`
2. `raw = load_secom(cfg.paths.raw_dir)`
3. `validate_secom(raw)`
4. `train, test = run_preprocessing(raw, cfg.run)`
5. `cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)`
6. `train.to_csv(cfg.paths.processed_dir / "train.csv", index=False)`
7. `test.to_csv(cfg.paths.processed_dir / "test.csv", index=False)`
8. Print to stdout: number of features retained, train row count, test row count, class distribution in each split.

Include `if __name__ == "__main__": main()` block.

### 3A.3 Acceptance criteria

- `python scripts/preprocess_data.py` exits 0.
- `data/processed/train.csv` and `data/processed/test.csv` exist.
- Both CSVs have identical sensor column sets.
- Sensor count is less than 590 (features were dropped by selection steps).
- No NaN values in any sensor column in either file.
- Both files have `label` (values 0 and 1 only) and `timestamp` columns.
- Train has ~80% of 1567 rows, test has ~20% (within rounding).
- Fail rate (label=1) is approximately 6.6% in both splits.

### 3A.4 Tests

**File: `tests/test_preprocess.py`** — add the following class.

Use a synthetic fixture: 40 rows, 5 sensor columns (mix of constant, NaN-heavy, correlated, and normal), labels `[0]*37 + [1]*3` to approximate class imbalance, timestamps `"2024-01-01"`. Construct a `RunConfig` with the production thresholds from config.yaml.

```
class TestRunPreprocessing:

    test_returns_two_dataframes
        Assert return type is a tuple of two DataFrames.

    test_output_has_label_column
        Assert both DataFrames have a "label" column with values in {0, 1}.

    test_no_nans_in_sensor_columns
        Assert no NaN in any sensor_* column of either DataFrame.

    test_train_test_disjoint
        Assert the index sets of train and test are disjoint.
```

---

## 4. Stage B — Baseline Model (Logistic Regression)

### 4B.1 Module: `src/yield_risk/model.py`

```python
"""Baseline model construction, training, and cross-validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline_pipeline(random_seed: int) -> Pipeline:
    """Construct a StandardScaler -> LogisticRegression pipeline.

    The classifier uses class_weight="balanced" to handle the ~6.6% fail-rate
    imbalance. Solver is "lbfgs" with max_iter=1000.

    Args:
        random_seed: Random state for the LogisticRegression.

    Returns:
        Unfitted sklearn Pipeline with steps "scaler" and "classifier".
    """


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit a sklearn Pipeline on training features and labels.

    Args:
        pipeline: Unfitted sklearn Pipeline.
        X_train: Training feature matrix (sensor columns only).
        y_train: Training labels (0/1).

    Returns:
        The same Pipeline object, now fitted.
    """


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Run stratified k-fold cross-validation and return per-fold scores.

    Uses sklearn.model_selection.cross_validate with
    scoring=["roc_auc", "f1"]. The pipeline is cloned internally per fold.

    Args:
        pipeline: sklearn Pipeline (fitted or unfitted; cloned internally).
        X: Feature matrix.
        y: Labels.
        cv_folds: Number of stratified folds.
        random_seed: Random state for StratifiedKFold.

    Returns:
        Dict with keys "test_roc_auc" and "test_f1", each a numpy array
        of length cv_folds.
    """
```

Implementation notes for `cross_validate_model`:
- Create `StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)`.
- Pass it as the `cv` parameter to `sklearn.model_selection.cross_validate`.
- Pass `scoring=["roc_auc", "f1"]`.
- Return the result dict directly (it contains the keys we need plus extras like `fit_time`).

### 4B.2 CLI script: `scripts/train_baseline.py`

```python
def main() -> None:
    """Train baseline logistic regression and save the fitted pipeline."""
```

Flow:
1. `cfg = load_config()`
2. `train = pd.read_csv(cfg.paths.processed_dir / "train.csv")`
3. Extract features: `sensor_cols = [c for c in train.columns if c.startswith("sensor_")]`
4. `X_train = train[sensor_cols]`
5. `y_train = train["label"]`
6. `pipeline = build_baseline_pipeline(cfg.run.random_seed)`
7. `cv_results = cross_validate_model(pipeline, X_train, y_train, cfg.run.cv_folds, cfg.run.random_seed)`
8. Print CV results to stdout:
   ```
   CV ROC-AUC: {mean:.3f} +/- {std:.3f}
   CV F1:      {mean:.3f} +/- {std:.3f}
   ```
9. `pipeline = train_model(pipeline, X_train, y_train)`
10. `cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)`
11. `joblib.dump(pipeline, cfg.paths.models_dir / "baseline_lr.joblib")`
12. Print: `Saved model to {path}`

Include `if __name__ == "__main__": main()` block.

### 4B.3 Acceptance criteria

- `python scripts/train_baseline.py` exits 0.
- `models/baseline_lr.joblib` exists and is loadable via `joblib.load`.
- Loaded object is a fitted `Pipeline` with steps named `"scaler"` and `"classifier"`.
- `classifier` is a `LogisticRegression` with `class_weight="balanced"`.
- CV ROC-AUC mean printed to stdout is > 0.5 (better than random).

### 4B.4 Tests

**File: `tests/test_model.py`**

Use a synthetic fixture: 100 rows, 5 features (random normal), binary labels (85 zeros, 15 ones). Seed all randomness.

```
class TestBuildBaselinePipeline:

    test_has_scaler_step
        Assert pipeline.named_steps["scaler"] is a StandardScaler.

    test_has_classifier_step
        Assert pipeline.named_steps["classifier"] is a LogisticRegression.

    test_classifier_uses_balanced_class_weight
        Assert classifier.class_weight == "balanced".

    test_classifier_uses_given_seed
        Assert classifier.random_state equals the seed passed to build_baseline_pipeline.


class TestTrainModel:

    test_pipeline_is_fitted_after_training
        Build pipeline, call train_model, assert hasattr(classifier, "coef_").

    test_predict_proba_returns_valid_probabilities
        Train model, call predict_proba, assert all values in [0, 1].


class TestCrossValidateModel:

    test_returns_expected_keys
        Assert "test_roc_auc" and "test_f1" are in the returned dict.

    test_per_fold_scores_length
        Assert each score array has length == cv_folds.
```

Total: 8 tests.

---

## 5. Stage C — Evaluation Module

### 5C.1 Module: `src/yield_risk/evaluate.py`

```python
"""Classification metrics, formatted reports, and diagnostic plots."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    """Container for binary classification evaluation metrics.

    Attributes:
        roc_auc: Area under the ROC curve.
        pr_auc: Area under the precision-recall curve (average precision).
        precision: Precision at the classification threshold.
        recall: Recall at the classification threshold.
        f1: F1 score at the classification threshold.
        confusion_matrix: 2x2 matrix as [[TN, FP], [FN, TP]].
    """

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute classification metrics from true labels and predicted probabilities.

    Args:
        y_true: Ground truth binary labels (0 or 1), shape (n_samples,).
        y_prob: Predicted probabilities for the positive class, shape (n_samples,).
        threshold: Decision threshold for converting probabilities to binary.

    Returns:
        Populated ClassificationMetrics.
    """
```

Implementation notes for `compute_metrics`:
- `y_pred = (y_prob >= threshold).astype(int)`
- `roc_auc = roc_auc_score(y_true, y_prob)` (uses probabilities, not binary predictions)
- `pr_auc = average_precision_score(y_true, y_prob)` (uses probabilities)
- `precision = precision_score(y_true, y_pred, zero_division=0)`
- `recall = recall_score(y_true, y_pred, zero_division=0)`
- `f1 = f1_score(y_true, y_pred, zero_division=0)`
- `cm = confusion_matrix(y_true, y_pred).tolist()`

```python
def format_report(metrics: ClassificationMetrics) -> str:
    """Format metrics as a human-readable multi-line text report.

    Args:
        metrics: Computed classification metrics.

    Returns:
        Formatted string with all metric values and confusion matrix.
    """
```

Expected output format:
```
=== Baseline Model Evaluation ===
ROC AUC:    0.XXX
PR AUC:     0.XXX
Precision:  0.XXX
Recall:     0.XXX
F1 Score:   0.XXX

Confusion Matrix:
              Pred Pass  Pred Fail
Actual Pass       XXX        XXX
Actual Fail       XXX        XXX
```

```python
def save_metrics(metrics: ClassificationMetrics, output_path: Path) -> None:
    """Serialize metrics to a JSON file.

    Uses dataclasses.asdict for conversion. confusion_matrix is stored
    as a nested list of ints.

    Args:
        metrics: Computed classification metrics.
        output_path: Path to write the JSON file.
    """


def plot_confusion_matrix(
    cm: list[list[int]],
    output_path: Path,
) -> None:
    """Save confusion matrix heatmap as PNG.

    Labels: x-axis "Predicted" (Pass/Fail), y-axis "Actual" (Pass/Fail).
    Annotate each cell with the count value.

    Args:
        cm: 2x2 confusion matrix [[TN, FP], [FN, TP]].
        output_path: File path for the saved PNG.
    """


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    output_path: Path,
) -> None:
    """Save ROC curve plot as PNG.

    Includes diagonal reference line. Legend shows AUC value.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities for positive class.
        roc_auc: Pre-computed AUC (displayed in legend).
        output_path: File path for the saved PNG.
    """


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    output_path: Path,
) -> None:
    """Save precision-recall curve plot as PNG.

    Legend shows AP value.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities for positive class.
        pr_auc: Pre-computed average precision (displayed in legend).
        output_path: File path for the saved PNG.
    """
```

All plot functions must call `plt.close()` after `plt.savefig()` to avoid memory leaks and display-backend issues.

### 5C.2 CLI script: `scripts/evaluate_model.py`

```python
def main() -> None:
    """Evaluate the baseline model on the test set and save results."""
```

Flow:
1. `cfg = load_config()`
2. `test = pd.read_csv(cfg.paths.processed_dir / "test.csv")`
3. `pipeline = joblib.load(cfg.paths.models_dir / "baseline_lr.joblib")`
4. `sensor_cols = [c for c in test.columns if c.startswith("sensor_")]`
5. `X_test, y_test = test[sensor_cols], test["label"].values`
6. `y_prob = pipeline.predict_proba(X_test)[:, 1]`
7. `metrics = compute_metrics(y_test, y_prob)`
8. Create output dirs: `cfg.paths.reports_dir` and `cfg.paths.figures_dir` (`mkdir(parents=True, exist_ok=True)`)
9. `save_metrics(metrics, cfg.paths.reports_dir / "baseline_metrics.json")`
10. `print(format_report(metrics))`
11. `plot_confusion_matrix(metrics.confusion_matrix, cfg.paths.figures_dir / "confusion_matrix.png")`
12. `plot_roc_curve(y_test, y_prob, metrics.roc_auc, cfg.paths.figures_dir / "roc_curve.png")`
13. `plot_precision_recall_curve(y_test, y_prob, metrics.pr_auc, cfg.paths.figures_dir / "precision_recall_curve.png")`
14. Print: `Saved metrics to {path}` and `Saved figures to {dir}`

Include `if __name__ == "__main__": main()` block.

### 5C.3 Acceptance criteria

- `python scripts/evaluate_model.py` exits 0.
- `reports/baseline_metrics.json` exists and is valid JSON.
- JSON keys: `roc_auc`, `pr_auc`, `precision`, `recall`, `f1`, `confusion_matrix`.
- All scalar metrics are floats in [0, 1].
- `confusion_matrix` is a 2x2 list of non-negative integers.
- `reports/figures/confusion_matrix.png` exists and is a valid PNG (file size > 0).
- `reports/figures/roc_curve.png` exists and is a valid PNG.
- `reports/figures/precision_recall_curve.png` exists and is a valid PNG.

### 5C.4 Tests

**File: `tests/test_evaluate.py`**

Use synthetic data for metrics tests. For perfect-prediction tests: `y_true = [0,0,0,1,1]`, `y_prob = [0.1, 0.2, 0.1, 0.9, 0.8]`. For near-random tests: `y_prob = [0.5]*5`.

```
class TestComputeMetrics:

    test_perfect_predictions_roc_auc_is_one
        y_prob perfectly separates classes. Assert roc_auc == 1.0.

    test_perfect_predictions_f1_is_one
        Same input. Assert f1 == 1.0.

    test_confusion_matrix_is_2x2
        Assert len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2.

    test_all_fields_populated
        Assert every field of ClassificationMetrics is not None.

    test_threshold_changes_predictions
        With y_prob = [0.4, 0.6], threshold=0.5 gives [0,1].
        With threshold=0.7, gives [0,0]. Assert different confusion matrices.


class TestFormatReport:

    test_contains_all_metric_names
        Assert "ROC AUC", "PR AUC", "Precision", "Recall", "F1" all
        appear in the returned string.

    test_returns_string
        Assert isinstance(result, str).


class TestPlotFunctions:

    test_plot_confusion_matrix_creates_file(tmp_path)
        Call plot_confusion_matrix with a synthetic 2x2 cm.
        Assert output file exists and has size > 0.

    test_plot_roc_curve_creates_file(tmp_path)
        Call with synthetic y_true, y_prob.
        Assert output file exists and has size > 0.

    test_plot_precision_recall_curve_creates_file(tmp_path)
        Call with synthetic y_true, y_prob.
        Assert output file exists and has size > 0.
```

Total: 10 tests.

---

## 6. Stage D — Feature Importance

### 6D.1 Module: `src/yield_risk/importance.py`

```python
"""Feature importance extraction and visualization for linear models."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def extract_lr_coefficients(
    pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract logistic regression coefficients ranked by absolute value.

    Accesses pipeline.named_steps["classifier"].coef_[0] to get the
    coefficient vector. Pairs each coefficient with its feature name,
    computes the absolute value, and sorts descending.

    Args:
        pipeline: Fitted Pipeline whose "classifier" step is a
            LogisticRegression with a coef_ attribute.
        feature_names: Feature names matching the order of coef_ values.

    Returns:
        DataFrame with columns "feature" (str), "coefficient" (float),
        "abs_coefficient" (float), sorted by abs_coefficient descending,
        index reset to 0..N-1.
    """


def plot_top_features(
    importance_df: pd.DataFrame,
    top_n: int,
    output_path: Path,
) -> None:
    """Save horizontal bar chart of top-N features by absolute coefficient.

    Bars colored by coefficient sign (positive = one color, negative = another).
    X-axis: coefficient value. Y-axis: feature name. Title includes top_n.

    Args:
        importance_df: DataFrame from extract_lr_coefficients (pre-sorted).
        top_n: Number of features to include in the chart.
        output_path: File path for the saved PNG.
    """
```

`plot_top_features` must call `plt.tight_layout()` before saving and `plt.close()` after saving.

### 6D.2 CLI script: `scripts/feature_importance.py`

```python
def main() -> None:
    """Extract and visualize feature importance from the baseline model."""
```

Flow:
1. `cfg = load_config()`
2. `test = pd.read_csv(cfg.paths.processed_dir / "test.csv")`
3. `pipeline = joblib.load(cfg.paths.models_dir / "baseline_lr.joblib")`
4. `sensor_cols = [c for c in test.columns if c.startswith("sensor_")]`
5. `importance_df = extract_lr_coefficients(pipeline, sensor_cols)`
6. `cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)`
7. `importance_df.to_csv(cfg.paths.reports_dir / "feature_importance.csv", index=False)`
8. `cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)`
9. `plot_top_features(importance_df, top_n=20, output_path=cfg.paths.figures_dir / "feature_importance.png")`
10. Print: top 10 features with their coefficients to stdout.
11. Print: `Saved importance to {csv_path}` and `Saved figure to {png_path}`

Include `if __name__ == "__main__": main()` block.

### 6D.3 Acceptance criteria

- `python scripts/feature_importance.py` exits 0.
- `reports/feature_importance.csv` exists with columns: `feature`, `coefficient`, `abs_coefficient`.
- CSV is sorted by `abs_coefficient` descending.
- Row count equals the number of sensor columns in the processed data.
- `reports/figures/feature_importance.png` exists and has size > 0.

### 6D.4 Tests

**File: `tests/test_importance.py`**

Create a synthetic fitted pipeline fixture: 3 features, fit a `StandardScaler` + `LogisticRegression(class_weight="balanced")` on small random data. Use a fixed seed for determinism.

```
class TestExtractLrCoefficients:

    test_returns_all_features
        Assert len(result) == len(feature_names).

    test_sorted_by_abs_coefficient_descending
        Assert abs_coefficient column is monotonically non-increasing.

    test_has_required_columns
        Assert columns are exactly {"feature", "coefficient", "abs_coefficient"}.


class TestPlotTopFeatures:

    test_creates_file(tmp_path)
        Call plot_top_features with top_n=2.
        Assert output file exists and has size > 0.
```

Total: 4 tests.

---

## 7. Notebook: `notebooks/03_baseline_results.ipynb`

Create a Jupyter notebook with the following cells. The notebook is a read-only report — it loads pre-computed artifacts from Stages C and D and displays them.

| Cell | Type | Content |
|---|---|---|
| 1 | Markdown | `# Baseline Model Results` — one-paragraph summary of the model |
| 2 | Code | Load `reports/baseline_metrics.json`, print all 5 scalar metrics |
| 3 | Markdown | `## Confusion Matrix` |
| 4 | Code | `display(Image("reports/figures/confusion_matrix.png"))` |
| 5 | Markdown | `## ROC Curve` |
| 6 | Code | `display(Image("reports/figures/roc_curve.png"))` |
| 7 | Markdown | `## Precision-Recall Curve` |
| 8 | Code | `display(Image("reports/figures/precision_recall_curve.png"))` |
| 9 | Markdown | `## Feature Importance (Top 20)` |
| 10 | Code | Load `reports/feature_importance.csv`, display top 20 rows + importance PNG |

---

## 8. Integration: End-to-End Pipeline

The four stages run sequentially. Each stage depends on the outputs of the previous stage:

```bash
python scripts/preprocess_data.py
python scripts/train_baseline.py
python scripts/evaluate_model.py
python scripts/feature_importance.py
```

### Disk artifacts after full run

```
data/processed/
  train.csv
  test.csv
models/
  baseline_lr.joblib
reports/
  baseline_metrics.json
  feature_importance.csv
  figures/
    confusion_matrix.png
    roc_curve.png
    precision_recall_curve.png
    feature_importance.png
notebooks/
  03_baseline_results.ipynb
```

---

## 9. Definition of Done

All commands must pass with zero errors:

```bash
ruff check src/ tests/
mypy src/yield_risk
pytest
```

```bash
python scripts/preprocess_data.py
python scripts/train_baseline.py
python scripts/evaluate_model.py
python scripts/feature_importance.py
```

After the pipeline run, verify all disk artifacts listed in Section 8 exist.

### Constraints

- Every public function and class gets a Google-style docstring (Args, Returns, Raises where applicable).
- All function signatures are fully type-annotated.
- Line length <= 88 characters.
- Do not modify config.yaml or any preprocessing threshold.
- `val_size` in config is unused in this spec (reserved for future hyperparameter tuning).
- Use `matplotlib.use("Agg")` before importing `pyplot` in all modules that produce plots, so scripts work headlessly.
- All plot functions must call `plt.close()` after `plt.savefig()`.
- Scripts follow the existing pattern in `scripts/download_data.py`: minimal `main()` with `if __name__ == "__main__": main()` guard.
