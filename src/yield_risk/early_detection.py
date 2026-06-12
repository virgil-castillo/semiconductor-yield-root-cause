"""Early detection module: sensor access primitives and scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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
