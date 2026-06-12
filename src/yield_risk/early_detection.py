"""Early detection module: sensor access primitives and scoring utilities."""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from yield_risk.config import CostMatrix
from yield_risk.preprocess import (
    drop_high_correlation,
    drop_high_missing,
    drop_low_variance,
)
from yield_risk.thresholding import expected_cost_at_threshold

_VALID_ACCESS_TYPES = frozenset({"prefix", "window"})


@dataclass(frozen=True)
class SensorAccess:
    """Describes which sensors are observable at a given production stage.

    Args:
        access_type: Either ``"prefix"`` or ``"window"``.
        prefix_end: Exclusive end index for prefix access. Required when
            ``access_type == "prefix"``.
        window_start: Inclusive start index for window access. Required when
            ``access_type == "window"``.
        window_size: Number of sensors in the window. Required when
            ``access_type == "window"``.

    Raises:
        ValueError: If fields are inconsistent with ``access_type`` or have
            invalid values.
    """

    access_type: str
    prefix_end: int | None = None
    window_start: int | None = None
    window_size: int | None = None

    def __post_init__(self) -> None:
        """Validate field consistency and individual field values.

        Raises:
            ValueError: If ``access_type`` is unknown, required fields are
                missing, disallowed fields are set, or values are out of range.
        """
        if self.access_type not in _VALID_ACCESS_TYPES:
            raise ValueError(
                f"access_type must be 'prefix' or 'window', got {self.access_type!r}"
            )

        if self.access_type == "prefix":
            if self.prefix_end is None:
                raise ValueError(
                    "prefix_end is required when access_type == 'prefix'"
                )
            if self.prefix_end < 1:
                raise ValueError(
                    f"prefix_end must be >= 1, got {self.prefix_end}"
                )
            if self.window_start is not None:
                raise ValueError(
                    "window_start must not be set when access_type == 'prefix'"
                )
            if self.window_size is not None:
                raise ValueError(
                    "window_size must not be set when access_type == 'prefix'"
                )

        else:  # access_type == "window"
            if self.window_start is None:
                raise ValueError(
                    "window_start is required when access_type == 'window'"
                )
            if self.window_size is None:
                raise ValueError(
                    "window_size is required when access_type == 'window'"
                )
            if self.window_start < 0:
                raise ValueError(
                    f"window_start must be >= 0, got {self.window_start}"
                )
            if self.window_size < 1:
                raise ValueError(
                    f"window_size must be >= 1, got {self.window_size}"
                )
            if self.prefix_end is not None:
                raise ValueError(
                    "prefix_end must not be set when access_type == 'window'"
                )


def latest_index(access: SensorAccess, n_sensors: int) -> int:
    """Return the exclusive end index of the observable window.

    For ``prefix`` access this equals ``prefix_end``. For ``window`` access
    this equals ``min(window_start + window_size, n_sensors)``.

    Args:
        access: The sensor access descriptor.
        n_sensors: Total number of raw sensor columns.

    Returns:
        The exclusive end index of the observable sensor window.
    """
    if access.access_type == "prefix":
        return access.prefix_end  # type: ignore[return-value]
    # window
    return min(access.window_start + access.window_size, n_sensors)  # type: ignore[operator]


def observation_fraction(access: SensorAccess, n_sensors: int) -> float:
    """Return the fraction of sensors observed so far.

    Computes ``latest_index(access, n_sensors) / n_sensors``, always in
    ``(0, 1]``.

    Args:
        access: The sensor access descriptor.
        n_sensors: Total number of raw sensor columns.

    Returns:
        A value in ``(0, 1]``.
    """
    return latest_index(access, n_sensors) / n_sensors


@dataclass(frozen=True)
class HyperparamConfig:
    """All hyperparameters for one early-detection scoring run.

    A pure frozen dataclass — no validation logic beyond immutability.
    Constructed directly by the caller (e.g. an Optuna objective).

    Attributes:
        access: Which sensors are observable at this production stage.
        missing_threshold: Maximum fraction of missing values allowed per
            column before that column is dropped.
        variance_threshold: Minimum variance a column must have to survive
            the variance filter.
        correlation_threshold: Absolute Pearson correlation above which one
            of a correlated pair is dropped.
        selection_method: Feature selection strategy.  One of ``"none"``,
            ``"univariate"``, ``"mutual_info"``, or ``"model_importance"``.
        max_features: Number of top features to keep after selection.
            ``None`` when ``selection_method == "none"``.
        model_family: Classifier family.  One of
            ``"logistic_regression"``, ``"random_forest"``, ``"xgboost"``,
            or ``"lightgbm"``.
        model_params: Hyperparameter overrides passed verbatim to
            ``build_estimator``.
        threshold_policy: Decision-threshold strategy.  Either ``"tune"``
            or ``"far_constraint"``.
        threshold: Decision threshold to apply.  Set iff
            ``threshold_policy == "tune"``.
        false_alarm_rate: Maximum tolerated false-alarm rate.  Set iff
            ``threshold_policy == "far_constraint"``.
    """

    access: SensorAccess
    missing_threshold: float
    variance_threshold: float
    correlation_threshold: float
    selection_method: str
    max_features: int | None
    model_family: str
    model_params: dict[str, object]
    threshold_policy: str
    threshold: float | None
    false_alarm_rate: float | None


def build_estimator(
    model_family: str,
    model_params: dict[str, object],
    random_seed: int,
) -> ClassifierMixin:
    """Return a seeded sklearn-compatible classifier for the given family.

    Applies the family's fixed defaults, then overlays everything in
    ``model_params`` (``model_params`` wins on conflict). ``random_seed``
    seeds the estimator.

    For ``xgboost`` and ``lightgbm``, ``scale_pos_weight`` is honoured if
    present in ``model_params``; otherwise the library default is used.

    Args:
        model_family: One of ``"logistic_regression"``, ``"random_forest"``,
            ``"xgboost"``, or ``"lightgbm"``.
        model_params: Hyperparameter overrides applied on top of family
            defaults. Any key valid for the underlying estimator is accepted.
        random_seed: Random state for the estimator.

    Returns:
        An unfitted sklearn-compatible classifier exposing ``fit`` and
        ``predict_proba``.

    Raises:
        ValueError: If ``model_family`` is not one of the four supported
            families.
    """
    if model_family == "logistic_regression":
        defaults: dict[str, Any] = {
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": random_seed,
        }
        defaults.update(model_params)
        return LogisticRegression(**defaults)

    if model_family == "random_forest":
        defaults = {
            "class_weight": "balanced",
            "random_state": random_seed,
            "n_jobs": 1,
        }
        defaults.update(model_params)
        return RandomForestClassifier(**defaults)

    if model_family == "xgboost":
        defaults = {
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": random_seed,
            "n_jobs": 1,
        }
        defaults.update(model_params)
        return XGBClassifier(**defaults)

    if model_family == "lightgbm":
        defaults = {
            "random_state": random_seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        defaults.update(model_params)
        return LGBMClassifier(**defaults)

    raise ValueError(
        f"Unknown model_family {model_family!r}. "
        "Expected one of 'logistic_regression', 'random_forest', "
        "'xgboost', 'lightgbm'."
    )


def select_ordered_sensors(
    x: pd.DataFrame,
    raw_sensor_cols: list[str],
    access: SensorAccess,
) -> tuple[pd.DataFrame, list[str]]:
    """Slice ``x`` to the columns defined by ``access``.

    Args:
        x: DataFrame containing at least all columns in ``raw_sensor_cols``.
        raw_sensor_cols: Ordered list of all raw sensor column names.
        access: The sensor access descriptor.

    Returns:
        A tuple ``(sliced_df, window_cols)`` where ``sliced_df`` is
        ``x[window_cols]`` (sensor columns only, original names) and
        ``window_cols`` is the ordered list of selected column names.

    Raises:
        ValueError: If ``raw_sensor_cols`` is empty, or if ``window_start``
            is >= ``n_sensors`` for a window access.
    """
    if not raw_sensor_cols:
        raise ValueError("No raw sensor_ columns found")

    n_sensors = len(raw_sensor_cols)

    if access.access_type == "window":
        window_start: int = access.window_start  # type: ignore[assignment]
        if window_start >= n_sensors:
            raise ValueError(
                f"window_start ({window_start}) must be < n_sensors ({n_sensors})"
            )
        end = min(window_start + access.window_size, n_sensors)  # type: ignore[operator]
        window_cols = raw_sensor_cols[window_start:end]
    else:  # prefix
        prefix_end: int = access.prefix_end  # type: ignore[assignment]
        window_cols = raw_sensor_cols[:prefix_end]

    return x[window_cols].copy(), window_cols


@dataclass
class FoldPreprocessor:
    """Fitted preprocessing state for a single cross-validation fold.

    All statistics are computed exclusively from the training fold, preventing
    any leakage from validation or test rows.

    Attributes:
        retained_cols: Ordered list of sensor column names that survived all
            preprocessing filters.
        medians: Mapping of column name to its train-fold median, used for
            imputation.
        scaler_mean: Per-column mean fitted by ``StandardScaler`` on the
            train fold (shape ``(n_retained,)``).
        scaler_scale: Per-column scale fitted by ``StandardScaler``.
            Zero-variance entries are replaced with ``1.0`` to prevent
            division by zero.
        selected_idx: Integer indices (into ``retained_cols``) of the columns
            kept by the feature selector.
    """

    retained_cols: list[str]
    medians: dict[str, float]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    selected_idx: np.ndarray  # indices into retained_cols kept by selection

    @classmethod
    def fit(
        cls,
        x_train_window: pd.DataFrame,
        y_train: np.ndarray,
        cfg: HyperparamConfig,
        random_seed: int,
    ) -> FoldPreprocessor:
        """Fit all preprocessing statistics on the training fold only.

        Pipeline order:
        1. ``drop_high_missing`` to choose surviving columns.
        2. Median imputation (compute and capture medians from train fold).
        3. ``drop_low_variance`` to further filter columns.
        4. ``drop_high_correlation`` to remove redundant columns.
        5. Fit ``StandardScaler``; replace zero-scale entries with 1.0.
        6. Fit the configured feature selector.

        Args:
            x_train_window: DataFrame of sensor columns for the training fold.
            y_train: Binary class labels aligned with ``x_train_window``.
            cfg: Hyperparameter configuration controlling thresholds, selector,
                and model family.
            random_seed: Seed for stochastic components (mutual_info, model).

        Returns:
            A fitted ``FoldPreprocessor`` capturing all statistics.
        """
        # --- Step 1: drop high-missing columns ----------------------------
        df = x_train_window.copy()
        df = drop_high_missing(df, cfg.missing_threshold)

        if df.shape[1] == 0:
            return cls(
                retained_cols=[],
                medians={},
                scaler_mean=np.empty(0, dtype=np.float64),
                scaler_scale=np.empty(0, dtype=np.float64),
                selected_idx=np.empty(0, dtype=int),
            )

        # --- Step 2: median imputation ------------------------------------
        medians: dict[str, float] = {
            col: float(df[col].median()) for col in df.columns
        }
        df = df.fillna(medians)

        # --- Step 3: drop low-variance ------------------------------------
        df = drop_low_variance(df, cfg.variance_threshold)

        if df.shape[1] == 0:
            return cls(
                retained_cols=[],
                medians={},
                scaler_mean=np.empty(0, dtype=np.float64),
                scaler_scale=np.empty(0, dtype=np.float64),
                selected_idx=np.empty(0, dtype=int),
            )

        # --- Step 4: drop high-correlation --------------------------------
        df = drop_high_correlation(df, cfg.correlation_threshold)

        if df.shape[1] == 0:
            return cls(
                retained_cols=[],
                medians={},
                scaler_mean=np.empty(0, dtype=np.float64),
                scaler_scale=np.empty(0, dtype=np.float64),
                selected_idx=np.empty(0, dtype=int),
            )

        retained_cols: list[str] = list(df.columns)
        # Restrict medians to only the retained columns
        retained_medians: dict[str, float] = {
            col: medians[col] for col in retained_cols
        }
        n_retained = len(retained_cols)

        # --- Step 5: fit StandardScaler -----------------------------------
        x_arr = df[retained_cols].values.astype(np.float64)
        scaler = StandardScaler()
        scaler.fit(x_arr)
        scaler_mean: np.ndarray = np.asarray(scaler.mean_, dtype=np.float64)
        scaler_scale: np.ndarray = np.asarray(scaler.scale_, dtype=np.float64)
        # Replace zero-scale entries with 1.0
        scaler_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)

        x_scaled = (x_arr - scaler_mean) / scaler_scale

        # --- Step 6: fit selector -----------------------------------------
        method = cfg.selection_method
        max_f = cfg.max_features

        if method == "none" or max_f is None:
            selected_idx: np.ndarray = np.arange(n_retained, dtype=int)

        elif method == "univariate":
            k = min(max_f, n_retained)
            selector = SelectKBest(f_classif, k=k)
            selector.fit(x_scaled, y_train)
            selected_idx = np.where(selector.get_support())[0].astype(int)

        elif method == "mutual_info":
            k = min(max_f, n_retained)
            mi_func = functools.partial(
                mutual_info_classif, random_state=random_seed
            )
            selector = SelectKBest(mi_func, k=k)
            selector.fit(x_scaled, y_train)
            selected_idx = np.where(selector.get_support())[0].astype(int)

        elif method == "model_importance":
            estimator = build_estimator(
                cfg.model_family, cfg.model_params, random_seed
            )
            sfm = SelectFromModel(estimator, max_features=max_f)
            sfm.fit(x_scaled, y_train)
            selected_idx = np.where(sfm.get_support())[0].astype(int)

        else:
            raise ValueError(
                f"Unknown selection_method {method!r}. "
                "Expected one of 'none', 'univariate', 'mutual_info', "
                "'model_importance'."
            )

        return cls(
            retained_cols=retained_cols,
            medians=retained_medians,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            selected_idx=selected_idx,
        )

    def transform(self, x_window: pd.DataFrame) -> np.ndarray:
        """Apply fitted preprocessing to a new (possibly val/test) DataFrame.

        Uses only statistics captured during ``fit`` — no recomputation from
        ``x_window``.

        Args:
            x_window: DataFrame of sensor columns (any split).

        Returns:
            Float64 array of shape ``(n_rows, n_selected)`` after imputation,
            scaling, and feature selection.  Returns shape ``(n_rows, 0)``
            when no columns were retained during fit.
        """
        n_rows = len(x_window)

        if not self.retained_cols or len(self.selected_idx) == 0:
            return np.empty((n_rows, 0), dtype=np.float64)

        # Subset to retained columns only
        df = x_window[self.retained_cols].copy()

        # Impute with train medians
        df = df.fillna(self.medians)

        x_arr = df.values.astype(np.float64)

        # Scale with fitted stats
        x_scaled = (x_arr - self.scaler_mean) / self.scaler_scale

        # Select features
        result: np.ndarray = x_scaled[:, self.selected_idx]
        return result


def resolve_threshold(
    y_val: np.ndarray,
    y_prob: np.ndarray,
    cfg: HyperparamConfig,
) -> float:
    """Resolve the decision threshold for the given policy.

    For ``"tune"`` policy the threshold stored in ``cfg.threshold`` is returned
    as-is.  For ``"far_constraint"`` policy the lowest threshold (highest
    recall) whose false-alarm rate (FAR = FP / (FP+TN)) is within
    ``cfg.false_alarm_rate`` is returned; ``1.0`` is returned when no
    candidate qualifies.

    Args:
        y_val: Ground-truth binary labels for the validation fold, shape (n,).
        y_prob: Predicted positive-class probabilities, shape (n,).
        cfg: Hyperparameter configuration carrying the threshold policy.

    Returns:
        A float decision threshold in [0, 1].
    """
    if cfg.threshold_policy == "tune":
        return float(cfg.threshold)  # type: ignore[arg-type]

    # "far_constraint" policy
    target_far: float = float(cfg.false_alarm_rate)  # type: ignore[arg-type]
    neg_mask: np.ndarray = y_val == 0
    n_neg = int(np.sum(neg_mask))

    candidates = np.sort(np.unique(y_prob))

    # If no true negatives, FAR is undefined → treat every threshold as valid;
    # return the smallest candidate (maximum recall).
    if n_neg == 0:
        return float(candidates[0]) if len(candidates) > 0 else 1.0

    best: float | None = None
    for t in candidates:
        fp = int(np.sum((y_prob >= t) & neg_mask))
        far = fp / n_neg
        if far <= target_far:
            # candidates are sorted ascending; first qualifying one is the
            # smallest threshold → maximum recall subject to FAR constraint.
            best = float(t)
            break

    return best if best is not None else 1.0


def compute_detection_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    metric_name: str,
    cost_matrix: CostMatrix | None = None,
) -> float:
    """Compute a named detection metric for the given predictions.

    Supported metric names:

    * ``"pr_auc"`` — average precision score (threshold-free).
    * ``"roc_auc"`` — ROC-AUC score (threshold-free).
    * ``"recall_at_far"`` — recall of the positive class at ``threshold``.
    * ``"neg_balanced_error"`` — ``balanced_accuracy_score - 1.0``.
    * ``"neg_expected_cost"`` — negated expected cost at ``threshold``.

    For the threshold-free AUC metrics (``"pr_auc"``, ``"roc_auc"``), if
    ``y_true`` contains only one class a ``UserWarning`` is emitted and
    ``float("nan")`` is returned instead of raising.

    Args:
        y_true: Ground-truth binary labels, shape (n,).
        y_prob: Predicted positive-class probabilities, shape (n,).
        threshold: Decision boundary used by thresholded metrics.
        metric_name: One of ``"pr_auc"``, ``"roc_auc"``, ``"recall_at_far"``,
            ``"neg_balanced_error"``, ``"neg_expected_cost"``.
        cost_matrix: Required when ``metric_name == "neg_expected_cost"``.

    Returns:
        Scalar float metric value.

    Raises:
        ValueError: If ``metric_name`` is unknown, or if ``metric_name`` is
            ``"neg_expected_cost"`` and ``cost_matrix`` is ``None``.
    """
    if metric_name == "pr_auc":
        if len(np.unique(y_true)) < 2:
            warnings.warn(
                "y_true has only one class; pr_auc is undefined — returning nan.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")
        return float(average_precision_score(y_true, y_prob))

    if metric_name == "roc_auc":
        if len(np.unique(y_true)) < 2:
            warnings.warn(
                "y_true has only one class; roc_auc is undefined — returning nan.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))

    if metric_name == "recall_at_far":
        n_pos = int(np.sum(y_true == 1))
        if n_pos == 0:
            return float("nan")
        tp = int(np.sum((y_prob >= threshold) & (y_true == 1)))
        return float(tp) / float(n_pos)

    if metric_name == "neg_balanced_error":
        y_pred = (y_prob >= threshold).astype(int)
        return float(balanced_accuracy_score(y_true, y_pred)) - 1.0

    if metric_name == "neg_expected_cost":
        if cost_matrix is None:
            raise ValueError(
                "cost_matrix must be provided when metric_name == 'neg_expected_cost'."
            )
        return -expected_cost_at_threshold(y_true, y_prob, threshold, cost_matrix)

    raise ValueError(
        f"Unknown metric_name {metric_name!r}. "
        "Expected one of 'pr_auc', 'roc_auc', 'recall_at_far', "
        "'neg_balanced_error', 'neg_expected_cost'."
    )


# ---------------------------------------------------------------------------
# ConfigScore and evaluate_config
# ---------------------------------------------------------------------------

EARLY_DETECTION_FLOOR: float = -1.0
"""Score returned for infeasible configs (< 2 valid CV folds)."""


@dataclass
class ConfigScore:
    """Aggregated result from one cross-validation evaluation of a HyperparamConfig.

    Attributes:
        penalized_score: ``detection_metric - alpha * observation_fraction``
            when feasible; ``EARLY_DETECTION_FLOOR`` (-1.0) when infeasible.
        detection_metric: Mean detection metric over valid CV folds; ``nan``
            when no valid fold exists.
        observation_fraction: Fraction of sensors observable at this stage
            (window-position-based, not fold-dependent).
        n_features_selected: Mean number of selected features across *valid*
            folds.  ``0.0`` when no valid fold exists.
        feasible: ``True`` when at least 2 folds produced a non-nan metric.
        per_fold: One dict per CV fold with keys ``metric``, ``n_features``,
            and ``threshold``.  Invalid folds record ``metric=nan``.
    """

    penalized_score: float
    detection_metric: float
    observation_fraction: float
    n_features_selected: float
    feasible: bool
    per_fold: list[dict[str, float]] = field(default_factory=list)


def evaluate_config(
    x: pd.DataFrame,
    y: np.ndarray,
    raw_sensor_cols: list[str],
    cfg: HyperparamConfig,
    *,
    inner_cv_folds: int,
    alpha: float,
    detection_metric: str,
    random_seed: int,
    cost_matrix: CostMatrix | None = None,
) -> ConfigScore:
    """Score a HyperparamConfig via stratified cross-validation.

    Runs ``inner_cv_folds``-fold stratified CV over ``x`` and ``y``.  In each
    fold, the training window is preprocessed with ``FoldPreprocessor``, a
    classifier is trained, probabilities are predicted on the val fold, and
    ``compute_detection_metric`` is called.  Folds where the preprocessor
    yields no features (empty ``retained_cols`` or empty ``selected_idx``) are
    marked invalid and do not contribute to the aggregate score.

    The trial is infeasible when fewer than 2 folds are valid, in which case
    ``penalized_score = EARLY_DETECTION_FLOOR`` and ``detection_metric = nan``
    are returned without raising.

    ``n_features_selected`` is the mean selected-feature count over *valid*
    folds only (invalid folds contribute ``0`` features but are excluded from
    this mean).

    Args:
        x: Full feature DataFrame.  Must contain all ``raw_sensor_cols``.
        y: Binary class labels aligned with ``x``, shape ``(n,)``.
        raw_sensor_cols: Ordered list of all raw sensor column names.
        cfg: Hyperparameter configuration for this trial.
        inner_cv_folds: Number of stratified CV folds.
        alpha: Earliness penalty weight.  ``penalized_score = detection_metric
            - alpha * observation_fraction``.
        detection_metric: Metric name passed to ``compute_detection_metric``.
        random_seed: Seed for ``StratifiedKFold``, ``FoldPreprocessor``, and
            ``build_estimator`` to ensure full determinism.
        cost_matrix: Required when ``detection_metric == "neg_expected_cost"``.

    Returns:
        A ``ConfigScore`` summarising the cross-validation result.

    Raises:
        ValueError: If ``raw_sensor_cols`` is empty, or if the sensor matrix
            (``x[raw_sensor_cols]``) contains infinite values.
    """
    # --- Up-front validation -------------------------------------------------
    if not raw_sensor_cols:
        raise ValueError("No raw sensor_ columns found")

    sensor_matrix = x[raw_sensor_cols]
    if np.any(np.isinf(sensor_matrix.values)):
        raise ValueError("Sensor matrix contains infinite values")

    # --- Observation fraction (window-position-based, not fold-dependent) ---
    obs_fraction = observation_fraction(cfg.access, len(raw_sensor_cols))

    # --- Cross-validation ----------------------------------------------------
    cv = StratifiedKFold(
        n_splits=inner_cv_folds, shuffle=True, random_state=random_seed
    )

    per_fold_records: list[dict[str, float]] = []

    for train_idx, val_idx in cv.split(x, y):
        y_train: np.ndarray = y[train_idx]
        y_val: np.ndarray = y[val_idx]

        x_train = x.iloc[train_idx]
        x_val = x.iloc[val_idx]

        # Slice to the observable window
        x_train_window, _ = select_ordered_sensors(x_train, raw_sensor_cols, cfg.access)
        x_val_window, _ = select_ordered_sensors(x_val, raw_sensor_cols, cfg.access)

        # Fit preprocessor on training window only
        preprocessor = FoldPreprocessor.fit(
            x_train_window, y_train, cfg, random_seed
        )

        # Invalid fold: empty preprocessor
        if not preprocessor.retained_cols or len(preprocessor.selected_idx) == 0:
            per_fold_records.append(
                {"metric": float("nan"), "n_features": 0.0, "threshold": float("nan")}
            )
            continue

        # Transform both splits
        x_train_arr = preprocessor.transform(x_train_window)
        x_val_arr = preprocessor.transform(x_val_window)

        # A single-class training fold cannot produce meaningful probability
        # estimates for the positive class — mark as invalid.
        if len(np.unique(y_train)) < 2:
            per_fold_records.append(
                {"metric": float("nan"), "n_features": 0.0, "threshold": float("nan")}
            )
            continue

        # Compute scale_pos_weight for xgboost/lightgbm when not explicitly set
        if cfg.model_family in {"xgboost", "lightgbm"} and (
            "scale_pos_weight" not in cfg.model_params
        ):
            n_pos = int(np.sum(y_train == 1))
            n_neg = int(np.sum(y_train == 0))
            spw: float = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
            model_params: dict[str, object] = dict(cfg.model_params)
            model_params["scale_pos_weight"] = spw
        else:
            model_params = dict(cfg.model_params)

        estimator = build_estimator(cfg.model_family, model_params, random_seed)
        estimator.fit(x_train_arr, y_train)
        proba_out: np.ndarray = estimator.predict_proba(x_val_arr)
        # Guard: if model only learned one class, treat fold as invalid
        if proba_out.shape[1] < 2:
            per_fold_records.append(
                {"metric": float("nan"), "n_features": 0.0, "threshold": float("nan")}
            )
            continue
        y_prob: np.ndarray = proba_out[:, 1]

        threshold = resolve_threshold(y_val, y_prob, cfg)
        metric = compute_detection_metric(
            y_val, y_prob, threshold, detection_metric, cost_matrix
        )

        n_selected = float(len(preprocessor.selected_idx))
        per_fold_records.append(
            {"metric": metric, "n_features": n_selected, "threshold": threshold}
        )

    # --- Aggregate -----------------------------------------------------------
    valid_metrics = [
        fd["metric"] for fd in per_fold_records if not math.isnan(fd["metric"])
    ]
    n_valid = len(valid_metrics)
    feasible = n_valid >= 2

    if feasible:
        mean_metric = float(np.mean(valid_metrics))
        # n_features_selected: mean over valid folds only
        valid_n_features = [
            fd["n_features"]
            for fd in per_fold_records
            if not math.isnan(fd["metric"])
        ]
        mean_n_features = float(np.mean(valid_n_features)) if valid_n_features else 0.0
        penalized = mean_metric - alpha * obs_fraction
        return ConfigScore(
            penalized_score=penalized,
            detection_metric=mean_metric,
            observation_fraction=obs_fraction,
            n_features_selected=mean_n_features,
            feasible=True,
            per_fold=per_fold_records,
        )
    else:
        return ConfigScore(
            penalized_score=EARLY_DETECTION_FLOOR,
            detection_metric=float("nan"),
            observation_fraction=obs_fraction,
            n_features_selected=0.0,
            feasible=False,
            per_fold=per_fold_records,
        )
