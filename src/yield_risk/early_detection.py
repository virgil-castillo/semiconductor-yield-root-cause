"""Early detection module: sensor access primitives and scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

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
