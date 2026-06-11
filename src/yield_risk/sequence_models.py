"""Experimental PyTorch GRU sequence model for SECOM pass/fail prediction.

This module contains the leakage-free raw-to-sequence preprocessing pipeline,
the ``SecomGRU`` model and its factory, and the in-memory dataset/loader. It is
fully isolated from the tabular pipeline and only mirrors the cleaning decisions
of ``yield_risk.preprocess`` (it never imports or mutates that module's state).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class RawSensorCleaner:
    """Replay-able raw SECOM cleaning decisions fit on a train sub-split.

    The cleaner reproduces the strict-threshold semantics of
    ``yield_risk.preprocess`` (high-missing drop with ``>``, low-variance drop
    with ``<``, and upper-triangle high-correlation drop with ``>`` removing the
    later column) so the GRU sees exactly the retained sensors the tabular
    pipeline would keep.

    Attributes:
        raw_sensor_cols: Raw sensor column names in fit-time order.
        sensor_cols: Retained model sensor names after all filtering.
        medians: Per-retained-sensor medians, shape ``(n_sensors,)`` float32.
        dropped_high_missing: Raw columns dropped for high missingness.
        dropped_low_variance: Columns dropped for low variance.
        dropped_high_correlation: Columns dropped for high correlation.
        missing_threshold: Strict upper bound on the missing fraction.
        variance_threshold: Strict lower bound on post-impute variance.
        correlation_threshold: Strict upper bound on absolute correlation.
    """

    raw_sensor_cols: list[str]
    sensor_cols: list[str]
    medians: np.ndarray
    dropped_high_missing: list[str]
    dropped_low_variance: list[str]
    dropped_high_correlation: list[str]
    missing_threshold: float
    variance_threshold: float
    correlation_threshold: float

    @classmethod
    def fit(
        cls,
        x_raw: np.ndarray,
        raw_sensor_cols: list[str],
        missing_threshold: float,
        variance_threshold: float,
        correlation_threshold: float,
    ) -> RawSensorCleaner:
        """Fit all raw cleaning decisions on a train sub-split matrix.

        Args:
            x_raw: Raw sensor matrix, shape ``(n_rows, n_raw_sensors)``. ``NaN``
                is allowed; ``+/-inf`` is rejected.
            raw_sensor_cols: Raw sensor names aligned to ``x_raw`` columns.
            missing_threshold: Strict upper bound on the missing fraction.
            variance_threshold: Strict lower bound on post-impute variance.
            correlation_threshold: Strict upper bound on absolute correlation.

        Returns:
            A fitted ``RawSensorCleaner``.

        Raises:
            ValueError: If the matrix is not 2-D, empty, mismatched with
                ``raw_sensor_cols``, contains infinite values, drops every
                sensor, or yields non-finite imputed values.
        """
        x = _validate_raw_matrix(x_raw, raw_sensor_cols)

        missing_frac = np.isnan(x).mean(axis=0)
        keep_missing = missing_frac <= missing_threshold
        n_raw = len(raw_sensor_cols)
        dropped_high_missing = [
            raw_sensor_cols[i] for i in range(n_raw) if not keep_missing[i]
        ]
        surviving = [raw_sensor_cols[i] for i in range(n_raw) if keep_missing[i]]
        x_surv = x[:, keep_missing]

        medians_surv = np.nanmedian(x_surv, axis=0).astype(np.float32)
        x_imp = _impute(x_surv, medians_surv)

        variances = x_imp.var(axis=0, ddof=1) if x_imp.shape[0] > 1 else np.zeros(
            x_imp.shape[1], dtype=np.float64
        )
        keep_var = variances >= variance_threshold
        dropped_low_variance = [
            surviving[i] for i in range(len(surviving)) if not keep_var[i]
        ]
        var_cols = [surviving[i] for i in range(len(surviving)) if keep_var[i]]
        var_medians = medians_surv[keep_var]
        x_var = x_imp[:, keep_var]

        dropped_high_correlation = _high_correlation_drops(
            x_var, var_cols, correlation_threshold
        )
        keep_corr = np.array(
            [c not in dropped_high_correlation for c in var_cols], dtype=bool
        )
        sensor_cols = [var_cols[i] for i in range(len(var_cols)) if keep_corr[i]]
        if len(sensor_cols) == 0:
            raise ValueError("No sensor columns remain after preprocessing")
        medians = var_medians[keep_corr].astype(np.float32)

        x_final = x_var[:, keep_corr]
        if not np.all(np.isfinite(x_final)):
            raise ValueError("Median imputation produced non-finite values")
        if not np.all(np.isfinite(medians)):
            raise ValueError("Median imputation produced non-finite values")

        return cls(
            raw_sensor_cols=list(raw_sensor_cols),
            sensor_cols=sensor_cols,
            medians=medians,
            dropped_high_missing=dropped_high_missing,
            dropped_low_variance=dropped_low_variance,
            dropped_high_correlation=dropped_high_correlation,
            missing_threshold=float(missing_threshold),
            variance_threshold=float(variance_threshold),
            correlation_threshold=float(correlation_threshold),
        )

    def transform_full(
        self, x_raw: np.ndarray, raw_sensor_cols: list[str]
    ) -> np.ndarray:
        """Replay cleaning on a full raw row matrix.

        Args:
            x_raw: Raw matrix, shape ``(n_rows, n_raw_sensors)``.
            raw_sensor_cols: Raw names; must equal the fitted raw schema.

        Returns:
            Finite retained sensor values, float32, shape
            ``(n_rows, n_sensors)`` in ``sensor_cols`` order.

        Raises:
            ValueError: On a raw schema mismatch, infinite values, or
                non-finite imputed values.
        """
        if list(raw_sensor_cols) != self.raw_sensor_cols:
            raise ValueError("Raw sensor columns differ from fitted pipeline")
        x = _validate_raw_matrix(x_raw, raw_sensor_cols)
        index = {c: i for i, c in enumerate(raw_sensor_cols)}
        cols = [index[c] for c in self.sensor_cols]
        x_sel = x[:, cols]
        x_imp = _impute(x_sel, self.medians)
        if not np.all(np.isfinite(x_imp)):
            raise ValueError("Median imputation produced non-finite values")
        return x_imp.astype(np.float32)

    def transform_window(
        self, x_raw: np.ndarray, window_sensor_cols: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Replay cleaning on a raw current window.

        Args:
            x_raw: Raw matrix, shape ``(n_rows, raw_window_size)``.
            window_sensor_cols: Raw names for each supplied column.

        Returns:
            A tuple ``(x_imp, retained_cols)`` where ``x_imp`` is float32 finite
            retained values, shape ``(n_rows, retained_window_size)``, and
            ``retained_cols`` are the retained names in caller window order.

        Raises:
            ValueError: On an empty window, unknown sensor, width mismatch, an
                all-dropped window, infinite values, or non-finite imputation.
        """
        if len(window_sensor_cols) == 0:
            raise ValueError("Window sensor columns must be non-empty")
        raw_set = set(self.raw_sensor_cols)
        for col in window_sensor_cols:
            if col not in raw_set:
                raise ValueError(f"Unknown window sensor column: {col}")
        x = np.asarray(x_raw, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("SequenceSensorPipeline.fit expects a 2-D array")
        if x.shape[1] != len(window_sensor_cols):
            raise ValueError(
                f"Expected {len(window_sensor_cols)} window columns, "
                f"got {x.shape[1]}"
            )
        if np.isinf(x).any():
            raise ValueError("Sensor matrix contains infinite values")

        retained_set = set(self.sensor_cols)
        median_index = {c: i for i, c in enumerate(self.sensor_cols)}
        keep_positions = [
            i for i, c in enumerate(window_sensor_cols) if c in retained_set
        ]
        if len(keep_positions) == 0:
            raise ValueError(
                "Window contains no retained sensor columns after preprocessing"
            )
        retained_cols = [window_sensor_cols[i] for i in keep_positions]
        x_sel = x[:, keep_positions]
        medians = np.array(
            [self.medians[median_index[c]] for c in retained_cols], dtype=np.float32
        )
        x_imp = _impute(x_sel, medians)
        if not np.all(np.isfinite(x_imp)):
            raise ValueError("Median imputation produced non-finite values")
        return x_imp.astype(np.float32), retained_cols


@dataclass
class SequenceSensorPipeline:
    """Checkpointed raw-to-sequence preprocessing for the GRU experiment.

    Wraps a :class:`RawSensorCleaner` with train-only ``StandardScaler``
    statistics and maps retained sensor names to zero-based sensor IDs.

    Attributes:
        cleaner: The fitted raw cleaning stage.
        mean: Scaler means, shape ``(n_sensors,)`` float32.
        scale: Scaler scales, shape ``(n_sensors,)`` float32; zeros are 1.0.
    """

    cleaner: RawSensorCleaner
    mean: np.ndarray
    scale: np.ndarray

    @property
    def raw_sensor_cols(self) -> list[str]:
        """Raw sensor names in fit-time order."""
        return self.cleaner.raw_sensor_cols

    @property
    def sensor_cols(self) -> list[str]:
        """Retained model sensor names in fit-time order."""
        return self.cleaner.sensor_cols

    @classmethod
    def fit(
        cls,
        x_raw: np.ndarray,
        raw_sensor_cols: list[str],
        missing_threshold: float,
        variance_threshold: float,
        correlation_threshold: float,
    ) -> SequenceSensorPipeline:
        """Fit the cleaner then the scaler on the train sub-split only.

        Args:
            x_raw: Raw matrix, shape ``(n_rows, n_raw_sensors)``.
            raw_sensor_cols: Raw names aligned to ``x_raw`` columns.
            missing_threshold: Strict upper bound on the missing fraction.
            variance_threshold: Strict lower bound on post-impute variance.
            correlation_threshold: Strict upper bound on absolute correlation.

        Returns:
            A fitted ``SequenceSensorPipeline``.

        Raises:
            ValueError: Propagated from :meth:`RawSensorCleaner.fit`.
        """
        cleaner = RawSensorCleaner.fit(
            x_raw,
            raw_sensor_cols,
            missing_threshold,
            variance_threshold,
            correlation_threshold,
        )
        x_clean = cleaner.transform_full(x_raw, raw_sensor_cols)
        scaler = StandardScaler()
        scaler.fit(x_clean)
        mean = np.asarray(scaler.mean_, dtype=np.float32)
        scale = np.asarray(scaler.scale_, dtype=np.float32).copy()
        scale[scale == 0.0] = 1.0
        return cls(cleaner=cleaner, mean=mean, scale=scale)

    def transform_full(
        self, x_raw: np.ndarray, raw_sensor_cols: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clean, scale, and emit tiled sensor IDs for a full raw row matrix.

        Args:
            x_raw: Raw matrix, shape ``(n_rows, n_raw_sensors)``.
            raw_sensor_cols: Raw names; must equal the fitted raw schema.

        Returns:
            A tuple ``(x_norm, sensor_ids)`` where ``x_norm`` is float32 shape
            ``(n_rows, n_sensors)`` and ``sensor_ids`` is int64 shape
            ``(n_rows, n_sensors)`` of tiled ``arange(n_sensors)``.

        Raises:
            ValueError: On a raw schema mismatch or non-finite values.
        """
        x_clean = self.cleaner.transform_full(x_raw, raw_sensor_cols)
        x_norm = ((x_clean - self.mean) / self.scale).astype(np.float32)
        n_rows = x_norm.shape[0]
        n_sensors = len(self.sensor_cols)
        sensor_ids = np.tile(np.arange(n_sensors, dtype=np.int64), (n_rows, 1))
        return x_norm, sensor_ids

    def transform_window(
        self, x_raw: np.ndarray, window_sensor_cols: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clean, scale, and emit sensor IDs for a raw current window.

        Args:
            x_raw: Raw matrix, shape ``(n_rows, raw_window_size)``.
            window_sensor_cols: Raw names for each supplied column.

        Returns:
            A tuple ``(x_norm, sensor_ids)`` of float32 normalized values and
            int64 sensor IDs, shape ``(n_rows, retained_window_size)``, in the
            caller's raw window order after dropped columns are removed.

        Raises:
            ValueError: Propagated from :meth:`RawSensorCleaner.transform_window`.
        """
        x_clean, retained_cols = self.cleaner.transform_window(
            x_raw, window_sensor_cols
        )
        id_index = {c: i for i, c in enumerate(self.sensor_cols)}
        ids = [id_index[c] for c in retained_cols]
        mean = self.mean[ids]
        scale = self.scale[ids]
        x_norm = ((x_clean - mean) / scale).astype(np.float32)
        n_rows = x_norm.shape[0]
        sensor_ids = np.tile(np.array(ids, dtype=np.int64), (n_rows, 1))
        return x_norm, sensor_ids

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-comparable dict of plain lists.

        Returns:
            A portable metadata dict with all column lists, arrays, and
            thresholds expressed as plain Python lists/floats.
        """
        return {
            "raw_sensor_cols": list(self.cleaner.raw_sensor_cols),
            "sensor_cols": list(self.cleaner.sensor_cols),
            "medians": self.cleaner.medians.astype(np.float32).tolist(),
            "dropped_high_missing": list(self.cleaner.dropped_high_missing),
            "dropped_low_variance": list(self.cleaner.dropped_low_variance),
            "dropped_high_correlation": list(self.cleaner.dropped_high_correlation),
            "missing_threshold": float(self.cleaner.missing_threshold),
            "variance_threshold": float(self.cleaner.variance_threshold),
            "correlation_threshold": float(self.cleaner.correlation_threshold),
            "mean": self.mean.astype(np.float32).tolist(),
            "scale": self.scale.astype(np.float32).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> SequenceSensorPipeline:
        """Reconstruct a pipeline from :meth:`to_dict` output.

        Args:
            d: A metadata dict produced by :meth:`to_dict`.

        Returns:
            The reconstructed ``SequenceSensorPipeline``.
        """
        cleaner = RawSensorCleaner(
            raw_sensor_cols=list(_as_str_list(d["raw_sensor_cols"])),
            sensor_cols=list(_as_str_list(d["sensor_cols"])),
            medians=np.asarray(d["medians"], dtype=np.float32),
            dropped_high_missing=list(_as_str_list(d["dropped_high_missing"])),
            dropped_low_variance=list(_as_str_list(d["dropped_low_variance"])),
            dropped_high_correlation=list(_as_str_list(d["dropped_high_correlation"])),
            missing_threshold=float(d["missing_threshold"]),  # type: ignore[arg-type]
            variance_threshold=float(d["variance_threshold"]),  # type: ignore[arg-type]
            correlation_threshold=float(
                d["correlation_threshold"]  # type: ignore[arg-type]
            ),
        )
        return cls(
            cleaner=cleaner,
            mean=np.asarray(d["mean"], dtype=np.float32),
            scale=np.asarray(d["scale"], dtype=np.float32),
        )


def _as_str_list(value: object) -> list[str]:
    """Coerce a stored sequence into a list of strings.

    Args:
        value: A list-like object of column names.

    Returns:
        A list of strings.
    """
    return [str(v) for v in value]  # type: ignore[attr-defined]


def _validate_raw_matrix(x_raw: np.ndarray, raw_sensor_cols: list[str]) -> np.ndarray:
    """Validate and cast a raw sensor matrix to float32.

    Args:
        x_raw: Candidate raw matrix.
        raw_sensor_cols: Raw names aligned to columns.

    Returns:
        The matrix cast to float32.

    Raises:
        ValueError: If not 2-D, empty, mismatched, or containing infinities.
    """
    x = np.asarray(x_raw, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("SequenceSensorPipeline.fit expects a 2-D array")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(
            "SequenceSensorPipeline.fit requires at least 1 row and 1 column"
        )
    if len(raw_sensor_cols) != x.shape[1]:
        raise ValueError("raw_sensor_cols length must match x columns")
    if np.isinf(x).any():
        raise ValueError("Sensor matrix contains infinite values")
    return x


def _impute(x: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Impute ``NaN`` entries column-wise with the given medians.

    Args:
        x: Float matrix, shape ``(n_rows, n_cols)``.
        medians: Per-column medians, shape ``(n_cols,)``.

    Returns:
        A float32 copy with ``NaN`` entries filled.
    """
    out = x.astype(np.float32).copy()
    mask = np.isnan(out)
    if mask.any():
        rows = np.where(mask)
        out[rows] = medians[rows[1]]
    return out


def _high_correlation_drops(
    x: np.ndarray, cols: list[str], threshold: float
) -> list[str]:
    """Return columns to drop using upper-triangle absolute Pearson correlation.

    Mirrors ``yield_risk.preprocess.drop_high_correlation``: for each column,
    drop it if its absolute correlation with any earlier column strictly exceeds
    ``threshold``.

    Args:
        x: Imputed matrix, shape ``(n_rows, n_cols)``.
        cols: Column names aligned to ``x``.
        threshold: Strict upper bound on absolute correlation.

    Returns:
        The list of column names to drop, in column order.
    """
    n = x.shape[1]
    if n < 2:
        return []
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(x, rowvar=False)
    corr = np.abs(np.atleast_2d(corr))
    to_drop: list[str] = []
    for j in range(n):
        for i in range(j):
            val = corr[i, j]
            if np.isfinite(val) and val > threshold:
                to_drop.append(cols[j])
                break
    return to_drop


def build_gru(
    n_sensors: int,
    emb_dim: int = 16,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
) -> SecomGRU:
    """Construct a :class:`SecomGRU` after validating hyperparameters.

    Args:
        n_sensors: Number of retained model sensors (embedding cardinality).
        emb_dim: Sensor-ID embedding dimension.
        hidden_size: GRU hidden size.
        num_layers: Number of stacked GRU layers.
        dropout: Inter-layer dropout in ``[0.0, 1.0)``.

    Returns:
        An initialized ``SecomGRU``.

    Raises:
        ValueError: For any out-of-range hyperparameter.
    """
    if n_sensors < 1:
        raise ValueError("n_sensors must be >= 1")
    if emb_dim < 1:
        raise ValueError("emb_dim must be >= 1")
    if hidden_size < 1:
        raise ValueError("hidden_size must be >= 1")
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("dropout must be in [0.0, 1.0)")
    return SecomGRU(
        n_sensors=n_sensors,
        emb_dim=emb_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


class SecomGRU(nn.Module):
    """GRU over an ordered SECOM sensor window with learned sensor-ID embeddings.

    Each timestep feeds ``[normalized_value, sensor_id_embedding]`` into the GRU.
    The model returns one final-window logit per wafer after reading the
    available sensor window.
    """

    def __init__(
        self,
        n_sensors: int,
        emb_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Initialize submodules and persist hyperparameters.

        Args:
            n_sensors: Embedding cardinality.
            emb_dim: Embedding dimension.
            hidden_size: GRU hidden size.
            num_layers: Number of GRU layers.
            dropout: User-requested inter-layer dropout (persisted as-is).
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_sensors, embedding_dim=emb_dim)
        self.gru = nn.GRU(
            input_size=1 + emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.head = nn.Linear(hidden_size, 1)
        self.n_sensors = n_sensors
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x: Tensor, sensor_ids: Tensor) -> Tensor:
        """Run the GRU sequence model.

        Args:
            x: Normalized sensor window values, shape ``(B, W)``, float32, on
                the model's device.
            sensor_ids: Sensor identity indices, shape ``(B, W)``, int64, on the
                model's device.

        Returns:
            Final-window logits, shape ``(B,)`` float32.

        Raises:
            ValueError: If ``x``/``sensor_ids`` are not aligned 2-D tensors, if
                the window is empty, or if any sensor ID is out of range.
        """
        if x.dim() != 2:
            raise ValueError("forward expects x as a 2-D (batch, window_size) tensor")
        if sensor_ids.dim() != 2:
            raise ValueError(
                "forward expects sensor_ids as a 2-D (batch, window_size) tensor"
            )
        if x.shape != sensor_ids.shape:
            raise ValueError("x and sensor_ids must have the same shape")
        if x.size(1) < 1:
            raise ValueError("window_size must be >= 1")
        if sensor_ids.min() < 0 or sensor_ids.max() >= self.n_sensors:
            raise ValueError("sensor_ids contain values outside [0, n_sensors)")

        vals = x.unsqueeze(-1)
        emb = self.embedding(sensor_ids)
        feats = torch.cat([vals, emb], dim=-1)
        out, _ = self.gru(feats)
        last = out[:, -1, :]
        logit: Tensor = self.head(last).squeeze(-1)
        return logit


def predict_logits(model: SecomGRU, x: Tensor, sensor_ids: Tensor) -> Tensor:
    """Return one logit per wafer regardless of mode, shape ``(B,)`` float32.

    Args:
        model: The GRU model.
        x: Normalized window values, shape ``(B, W)`` float32.
        sensor_ids: Sensor IDs, shape ``(B, W)`` int64.

    Returns:
        Final-decision logits, shape ``(B,)`` float32.
    """
    logits: Tensor = model(x, sensor_ids)
    return logits


class SecomSequenceDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """In-memory dataset of normalized sensor windows and float labels.

    Attributes:
        x: ``(n_examples, window_size)`` float32 CPU tensor.
        sensor_ids: ``(n_examples, window_size)`` int64 CPU tensor.
        y: ``(n_examples,)`` float32 CPU tensor (0.0/1.0).
    """

    def __init__(self, x: np.ndarray, sensor_ids: np.ndarray, y: np.ndarray) -> None:
        """Validate inputs and store contiguous CPU tensors.

        Args:
            x: Normalized values, shape ``(n_examples, window_size)``.
            sensor_ids: Sensor IDs, shape ``(n_examples, window_size)``.
            y: Labels, shape ``(n_examples,)``.

        Raises:
            ValueError: On wrong dimensionality, shape mismatch, row mismatch,
                or an empty window.
        """
        if x.ndim != 2:
            raise ValueError("x must be 2-D")
        if sensor_ids.ndim != 2:
            raise ValueError("sensor_ids must be 2-D")
        if x.shape != sensor_ids.shape:
            raise ValueError("x and sensor_ids must have the same shape")
        if y.ndim != 1:
            raise ValueError("y must be 1-D")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x has {x.shape[0]} rows but y has {y.shape[0]}")
        if x.shape[1] == 0:
            raise ValueError("window_size must be >= 1")
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.sensor_ids = torch.from_numpy(
            np.ascontiguousarray(sensor_ids, dtype=np.int64)
        )
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self) -> int:
        """Return the number of examples."""
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(x, sensor_ids, y)`` for one example.

        Args:
            idx: Example index.

        Returns:
            A tuple of the row's values, sensor IDs, and 0-D float32 label.
        """
        return self.x[idx], self.sensor_ids[idx], self.y[idx]


def make_loader(
    dataset: SecomSequenceDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    """Build a deterministic DataLoader.

    Args:
        dataset: The sequence dataset.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle each epoch.
        seed: Seed for the shuffle generator and worker seeding.
        num_workers: Worker process count; ``0`` loads in the main process.

    Returns:
        A configured ``DataLoader`` with ``drop_last=False`` and
        ``pin_memory=False``.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init_fn(worker_id: int) -> None:
        import random

        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        generator=generator if shuffle else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


def compute_pos_weight(y_train: np.ndarray) -> float:
    """Compute BCEWithLogitsLoss ``pos_weight = n_neg / n_pos`` on TRAIN only.

    Args:
        y_train: Integer label array of 0/1 values.

    Returns:
        The ratio ``n_neg / n_pos`` as a float.

    Raises:
        ValueError: If ``y_train`` is empty, has no positives, or no negatives.
    """
    if len(y_train) == 0:
        raise ValueError("pos_weight undefined: y_train is empty")
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0:
        raise ValueError("pos_weight undefined: train split has no positive samples")
    if n_neg == 0:
        raise ValueError("pos_weight undefined: train split has no negative samples")
    return float(n_neg) / float(n_pos)
