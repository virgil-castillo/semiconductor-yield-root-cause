# SDD Spec: Experimental PyTorch GRU Sequence Model for SECOM Wafer Pass/Fail Prediction

> Synthesized from 3 independent drafts (simplicity / extensibility / ML-correctness lenses)
> via sdd-brainstorm-spec. Three flagged decisions were resolved by the maintainer:
> config-driven hyperparameters (a committed `configs/sequence_config.yaml`), **no** early
> stopping (fixed epochs), and the larger default model size. See §16.

## 0. Scope, lens, non-goals

**Scope.** Add an ISOLATED PyTorch GRU experiment to the `yield_risk` package (src layout, Python >= 3.12). The model treats ordered SECOM sensor observations as a pseudo-sequence with learned per-sensor-ID embeddings, and predicts wafer pass/fail from either a complete sensor row or the current available sensor window. It reuses `data/processed/{train,test}.csv`, `yield_risk.config`, and `yield_risk.evaluate`'s plot/metric helpers, but does not modify the tabular pipeline.

**Non-goals (MUST NOT modify).** `src/yield_risk/evaluate.py` (`ClassificationMetrics` shape is frozen — tabular pipeline depends on it), `src/yield_risk/model.py`, `src/yield_risk/config.py`, `src/yield_risk/__init__.py`, `scripts/train_models.py`, `scripts/evaluate_model.py`, `reports/model_comparison.json`, `models/model_metadata.json`, or any existing artifact. The GRU writes only to namespaced artifacts (§9).

**New files (repo-relative paths):**
- `src/yield_risk/sequence_models.py`
- `src/yield_risk/sequence_train.py`
- `scripts/train_sequence_model.py`
- `scripts/evaluate_sequence_model.py`
- `configs/sequence_config.yaml`
- `tests/test_sequence_models.py`
- `tests/test_sequence_train.py`
- `docs/sequence_model.md`

**Dev contract (enforced on all new files).** `from __future__ import annotations`; Google-style docstrings with Args/Returns/Raises; full type annotations; `mypy --strict` clean; ruff `E,F,W,I,UP,ANN` @ line length 88.

---

## 1. Dependency and imports

### 1.1 `pyproject.toml`

PyTorch is a required dependency for this experiment. Add torch to the project
dependencies, not as an optional guarded import path:

```toml
[project]
dependencies = [
    # existing dependencies...
    "torch>=2.2",
]

[[tool.mypy.overrides]]
module = "torch.*"
ignore_missing_imports = true
```

CPU-only torch is sufficient. No CUDA requirement.

### 1.2 Torch import strategy - direct imports

Torch is installed in the supported development environment, so the sequence
modules import it directly. Do not add optional-import guard machinery or a
custom missing-torch exception path.

At the top of `sequence_models.py`:

```python
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
```

Rules:
- `sequence_models.py` and `sequence_train.py` import torch directly.
- Torch-typed signatures use normal annotations (`Tensor`, `SecomGRU`) under
  `from __future__ import annotations`.

### 1.3 Exception scope

No custom torch exception is used. Validation errors raise a specific builtin
(`ValueError`/`FileNotFoundError`/`KeyError`) with the exact message specified
inline. If torch is missing, Python raises the normal `ImportError` at import
time; the supported environment is expected to install the required dependency.

---

## 2. Data contract and pinned dtypes

Source files: `cfg.paths.processed_dir / "train.csv"` and `.../ "test.csv"`.

- Sensor columns: `sensor_cols = [c for c in df.columns if c.startswith("sensor_")]`, in DataFrame column order. Length = `n_sensors` (~474 in real data; never hard-coded).
- A model input is a **current sensor window**, not necessarily a complete row.
  A window is represented by normalized sensor values plus the matching
  zero-based sensor-ID indices from `sensor_cols`.
- Full-row scoring is the special case where `window_sensor_cols == sensor_cols`.
- Prefix/current-progress scoring is the primary use case: for a wafer where
  only the first `k` ordered sensors are available, pass those `k` values and
  sensor IDs. Contiguous or sparse windows are also valid as long as every
  window sensor name exists in the training-time `sensor_cols`.
- `df["label"]`: int 0/1 (0=pass, 1=fail).
- `df["timestamp"]`: string, ignored by the GRU.

**Canonical dtypes (pinned at every boundary):**
- Raw sensor values: `np.float32` in numpy.
- Normalized window values: `np.float32` in numpy → `torch.float32` tensors,
  shape `(batch, window_size)`.
- Labels: `np.int64` in numpy; loss targets are `torch.float32` (BCEWithLogitsLoss requires float targets).
- Sensor-ID indices: `np.int64` in numpy → `torch.int64` tensors, shape
  `(batch, window_size)` (required by `nn.Embedding`).
- Scaler mean/scale: `np.float32`, shape `(n_sensors,)`, persisted with the
  sequence preprocessing pipeline.
- `y_prob` returned to sklearn metrics: `np.float64` (metric stability).

Extraction idiom: `df[sensor_cols].to_numpy(dtype=np.float32)`, `df["label"].to_numpy(dtype=np.int64)`.

---

## 3. Sequence preprocessing pipeline (leakage-free) and pos_weight

### 3.1 `SequenceSensorPipeline`

The GRU must never receive raw, unnormalized sensor magnitudes. Fit one
sequence-specific preprocessing pipeline on the training sub-split, persist it
with the checkpoint, and reload it for every inference/evaluation path.

Do not reuse the existing tabular model pipeline directly: tabular pipelines
include a classifier, and only some tabular model families include a
`StandardScaler`. The sequence pipeline reuses the same sklearn
`StandardScaler` semantics but owns its fit statistics and sensor ordering for
the GRU experiment.

Plain `@dataclass` in `sequence_models.py` (no torch needed to fit):

```python
@dataclass
class SequenceSensorPipeline:
    sensor_cols: list[str]
    mean: np.ndarray   # shape (n_sensors,), float32
    scale: np.ndarray  # shape (n_sensors,), float32, zeros replaced by 1.0

    @classmethod
    def fit(cls, x: np.ndarray, sensor_cols: list[str]) -> SequenceSensorPipeline: ...
    def transform_full(self, x: np.ndarray, sensor_cols: list[str]) -> tuple[np.ndarray, np.ndarray]: ...
    def transform_window(self, x: np.ndarray, window_sensor_cols: list[str]) -> tuple[np.ndarray, np.ndarray]: ...
    def to_dict(self) -> dict[str, object]: ...
    @classmethod
    def from_dict(cls, d: dict[str, object]) -> SequenceSensorPipeline: ...
```

Math and dtypes:
- Fit with `sklearn.preprocessing.StandardScaler` on the TRAIN sub-split matrix
  after the validation carve-out. Persist `scaler.mean_` and `scaler.scale_`
  as `np.float32` arrays rather than the sklearn object itself.
- **scale==0 guard:** StandardScaler sets zero-variance feature scale to `1.0`;
  preserve that behavior. A constant column maps to all-zeros after centering.
- `transform_full` validates that `sensor_cols == self.sensor_cols`, applies
  `(x - mean) / scale`, returns `(x_norm, sensor_ids)` where `x_norm` is
  `np.float32` shape `(n_rows, n_sensors)` and `sensor_ids` is `np.int64`
  shape `(n_rows, n_sensors)` containing tiled `np.arange(n_sensors)`.
- `transform_window` accepts raw window values shape `(n_rows, window_size)` and
  the exact `window_sensor_cols` for those columns. It looks up each window
  column in the training-time `sensor_cols`, applies that sensor's saved
  `mean`/`scale`, and returns `(x_norm, sensor_ids)` with shape
  `(n_rows, window_size)`.
- `to_dict`/`from_dict` use lists for `sensor_cols`, `mean`, and `scale`
  (portable, JSON-comparable in tests).

Errors:
- `fit`: `ValueError("SequenceSensorPipeline.fit expects a 2-D array")` if `x.ndim != 2`; `ValueError("SequenceSensorPipeline.fit requires at least 1 row and 1 column")` if `x.shape[0] == 0 or x.shape[1] == 0`; `ValueError("sensor_cols length must match x columns")` if `len(sensor_cols) != x.shape[1]`.
- `transform_full`: `ValueError("Sensor columns differ from fitted pipeline")`
  if the column list differs in membership or order.
- `transform_window`: `ValueError("Window sensor columns must be non-empty")`
  for an empty window; `ValueError(f"Unknown window sensor column: {col}")`
  for any sensor not seen during fit; `ValueError(f"Expected {len(window_sensor_cols)} window columns, got {x.shape[1]}")` on width mismatch.

**Leakage rule (fit AFTER val carve-out).** The preprocessing pipeline is fit
ONLY on the train sub-split matrix produced after the val carve-out (§10.1).
Val and test are transformed with these train statistics. The val set is used
for per-epoch monitoring so it must not contribute to fit statistics. Persisted
in the checkpoint so eval reloads the training-fitted preprocessing pipeline
and never re-fits.

### 3.2 pos_weight — RAISE on degenerate labels

A train split with no positives or no negatives means the experiment is misconfigured; silently substituting `1.0` produces a quietly meaningless model. Tests construct valid two-class synthetic data and separately assert the raise on degenerate input.

```python
def compute_pos_weight(y_train: np.ndarray) -> float:
    """Compute BCEWithLogitsLoss pos_weight = n_neg / n_pos on TRAIN only.

    Raises:
        ValueError: If y_train is empty, has no positives, or no negatives.
    """
```
- `n_pos = int((y_train == 1).sum())`, `n_neg = int((y_train == 0).sum())`.
- Returns `float(n_neg) / float(n_pos)`. For SECOM ≈ 14.2.
- `ValueError("pos_weight undefined: y_train is empty")` if `len(y_train) == 0`; `ValueError("pos_weight undefined: train split has no positive samples")` if `n_pos == 0`; `ValueError("pos_weight undefined: train split has no negative samples")` if `n_neg == 0`.
- At loss-construction time: `pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)`, shape `(1,)`.

---

## 4. Model: `SecomGRU` (file `sequence_models.py`)

### 4.1 Factory and validation

```python
def build_gru(
    n_sensors: int,
    emb_dim: int = 16,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    early_prediction: bool = False,
) -> SecomGRU:
    """Construct a SecomGRU after validating hyperparameters.

    Raises:
        ValueError: For invalid hyperparameters (see below).
    """
```

`build_gru` validates and raises `ValueError` with exact messages:
- `n_sensors < 1` → `"n_sensors must be >= 1"`
- `emb_dim < 1` → `"emb_dim must be >= 1"`
- `hidden_size < 1` → `"hidden_size must be >= 1"`
- `num_layers < 1` → `"num_layers must be >= 1"`
- `dropout < 0.0 or dropout >= 1.0` → `"dropout must be in [0.0, 1.0)"`

`nn.GRU` receives `dropout=(dropout if num_layers > 1 else 0.0)` (avoids torch's single-layer dropout warning) while the user-requested `dropout` is persisted in the checkpoint.

### 4.2 `__init__` submodules

- `self.embedding = nn.Embedding(num_embeddings=n_sensors, embedding_dim=emb_dim)`.
- `self.gru = nn.GRU(input_size=1 + emb_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))`.
- `self.head = nn.Linear(hidden_size, 1)` (shared across timesteps in early mode).
- `self.early_prediction = early_prediction`.
- Store `self.n_sensors, self.emb_dim, self.hidden_size, self.num_layers, self.dropout` as plain attributes for checkpointing.

### 4.3 `forward` — exact shape walk

```python
def forward(self, x: Tensor, sensor_ids: Tensor) -> Tensor:
    """Run the GRU sequence model.

    Args:
        x: Normalized sensor window values, shape (B, W), float32, on the
            model's device.
        sensor_ids: Sensor identity indices for the same window, shape
            (B, W), int64, on the model's device.

    Returns:
        early_prediction=False: logits shape (B,), float32 (final-timestep).
        early_prediction=True: logits shape (B, W), float32.

    Raises:
        ValueError: If x/sensor_ids are not aligned 2-D tensors, if the
            window is empty, or if any sensor ID is outside the embedding
            range.
    """
```

1. Input `x`: `(B, W)` float32, `W = window_size`. `sensor_ids`: `(B, W)` int64. Guards: `if x.dim() != 2: raise ValueError("forward expects x as a 2-D (batch, window_size) tensor")`; `if sensor_ids.dim() != 2: raise ValueError("forward expects sensor_ids as a 2-D (batch, window_size) tensor")`; `if x.shape != sensor_ids.shape: raise ValueError("x and sensor_ids must have the same shape")`; `if x.size(1) < 1: raise ValueError("window_size must be >= 1")`; `if sensor_ids.min() < 0 or sensor_ids.max() >= self.n_sensors: raise ValueError("sensor_ids contain values outside [0, n_sensors)")`.
2. `vals = x.unsqueeze(-1)` → `(B, W, 1)` float32.
3. `emb = self.embedding(sensor_ids)` → `(B, W, emb_dim)` float32.
4. `feats = torch.cat([vals, emb], dim=-1)` → `(B, W, 1 + emb_dim)` float32. (Value first, then embedding — matches brief's `[normalized_value, sensor_id_embedding]`.)
5. `out, _ = self.gru(feats)` → `out`: `(B, W, hidden_size)` float32.
6. **Standard:** `last = out[:, -1, :]` → `(B, hidden_size)`; `logit = self.head(last)` → `(B, 1)`; `return logit.squeeze(-1)` → `(B,)`.
7. **Early:** `logits_seq = self.head(out)` → `(B, W, 1)`; `return logits_seq.squeeze(-1)` → `(B, W)`.

### 4.4 Mode-unifying prediction helper

```python
def predict_logits(model: SecomGRU, x: Tensor, sensor_ids: Tensor) -> Tensor:
    """Return one logit per wafer regardless of mode, shape (B,), float32.

    Standard mode: forward(x, sensor_ids). Early mode:
    forward(x, sensor_ids)[:, -1] (current-window final timestep).
    """
```
`y_prob = torch.sigmoid(logit)` is computed by the caller (eval loop), never inside the model.

---

## 5. Dataset and DataLoader (file `sequence_models.py`)

```python
class SecomSequenceDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """In-memory dataset of normalized sensor windows and float labels.

    Attributes:
        x: (n_examples, window_size) float32 CPU tensor.
        sensor_ids: (n_examples, window_size) int64 CPU tensor.
        y: (n_examples,) float32 CPU tensor (0.0/1.0).
    """
    def __init__(self, x: np.ndarray, sensor_ids: np.ndarray, y: np.ndarray) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]: ...
```
- `__init__` sets `self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))`, `self.sensor_ids = torch.from_numpy(np.ascontiguousarray(sensor_ids, dtype=np.int64))`, `self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))`.
- Validation: `ValueError("x must be 2-D")` if `x.ndim != 2`; `ValueError("sensor_ids must be 2-D")` if `sensor_ids.ndim != 2`; `ValueError("x and sensor_ids must have the same shape")` if shapes differ; `ValueError("y must be 1-D")` if `y.ndim != 1`; `ValueError(f"x has {x.shape[0]} rows but y has {y.shape[0]}")` on row mismatch; `ValueError("window_size must be >= 1")` if `x.shape[1] == 0`.
- `__getitem__` returns `(self.x[idx], self.sensor_ids[idx], self.y[idx])` (label a 0-D float32 tensor). Default collate → batch `x: (B, W)` float32, `sensor_ids: (B, W)` int64, `y: (B,)` float32.

```python
def make_loader(dataset: SecomSequenceDataset, batch_size: int, shuffle: bool,
                seed: int, num_workers: int = 0) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    """Build a deterministic DataLoader."""
```
- `generator = torch.Generator(); generator.manual_seed(seed)`, passed to `DataLoader(generator=generator)` when `shuffle=True`.
- `drop_last=False`, `pin_memory=False`, `num_workers` default 0. When `>0`, `worker_init_fn` sets `np.random.seed(seed + worker_id)` and `random.seed(seed + worker_id)`.

---

## 6. Checkpoint schema and save/load (file `sequence_train.py`) — flat schema

Flat keys are simpler to validate key-by-key and compare in tests. Saved via `torch.save` (not JSON; contains `model_state_dict`).

### 6.1 Exact `ckpt: dict[str, object]` — EXACTLY these keys

| Key | Type | Notes |
|---|---|---|
| `format_version` | `int` | `1` |
| `model_state_dict` | `dict[str, Tensor]` | `model.state_dict()` |
| `n_sensors` | `int` | from train.csv |
| `emb_dim` | `int` | |
| `hidden_size` | `int` | |
| `num_layers` | `int` | |
| `dropout` | `float` | user-requested value |
| `early_prediction` | `bool` | |
| `timestep_weighting` | `str` | `"none"`/`"linear"`/`"sqrt"`; stored even when `early_prediction=False` (ignored on load then) |
| `preprocessor` | `dict[str, object]` | `SequenceSensorPipeline.to_dict()` with `sensor_cols`, `mean`, `scale` |
| `sensor_cols` | `list[str]` | ordered fit-time names |
| `pos_weight` | `float` | train-only n_neg/n_pos |
| `random_seed` | `int` | |
| `epochs` | `int` | epochs actually trained |
| `lr` | `float` | |
| `batch_size` | `int` | |
| `torch_version` | `str` | `torch.__version__` |
| `cuda_available` | `bool` | `torch.cuda.is_available()` at save |
| `device` | `str` | `"cpu"` or `"cuda"` used for training |
| `created_at` | `str` | ISO-8601 UTC timestamp (`datetime.now(timezone.utc).isoformat()`) |

### 6.2 Signatures

```python
def save_checkpoint(path: Path, model: SecomGRU, preprocessor: SequenceSensorPipeline,
                    sensor_cols: list[str], pos_weight: float,
                    timestep_weighting: str, random_seed: int, epochs: int,
                    lr: float, batch_size: int, device: str) -> None:
    """Serialize model + preprocessing pipeline for leakage-free eval.

    Raises:
        ValueError: If len(preprocessor.mean) != model.n_sensors or
            len(sensor_cols) != model.n_sensors.
    """

@dataclass
class LoadedCheckpoint:
    model: SecomGRU            # eval() mode, on device, weights loaded
    preprocessor: SequenceSensorPipeline
    sensor_cols: list[str]
    pos_weight: float
    early_prediction: bool
    timestep_weighting: str
    raw: dict[str, object]

def load_checkpoint(path: Path, device: str = "cpu") -> LoadedCheckpoint:
    """Load checkpoint, rebuild model via build_gru, restore weights/stats.

    Uses torch.load(path, map_location=device, weights_only=False).

    Raises:
        FileNotFoundError: If path does not exist.
        KeyError: If a required schema key is missing.
        ValueError: If format_version != 1 or preprocessor metadata is invalid.
    """
```
- Preprocessor reconstructed via `SequenceSensorPipeline.from_dict(ckpt["preprocessor"])`.
- Missing key → `KeyError(f"Checkpoint missing required key: {key}")`.
- `format_version != 1` → `ValueError(f"Unsupported checkpoint format_version: {v}")`.
- Round-trip guarantee (acceptance-tested): after load, `predict_logits(model, x, sensor_ids)` equals the pre-save model's output bit-for-bit (`torch.equal`) under `eval()`/`no_grad`.

---

## 7. Loss math — timestep weights normalized to **sum 1**, schemes `{none, linear, sqrt}`

Sum-to-1 keeps the loss scale identical across weighting schemes and identical to the unweighted mean baseline, so loss values are directly comparable across runs.

### 7.1 Standard mode
- `criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)`.
- `logits = model(x, sensor_ids)` → `(B,)`; `loss = criterion(logits, y)` (`y` `(B,)` float32), default `"mean"` reduction → scalar.

### 7.2 Early-prediction mode
- `logits = model(x, sensor_ids)` → `(B, W)`.
- `targets = y.unsqueeze(1).expand(B, W)` → `(B, W)` float32 (wafer label repeated across window timesteps).
- `elem = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight_tensor, reduction="none")` → `(B, W)`. `pos_weight` shape `(1,)` broadcasts over the last dim.
- `w = build_timestep_weights(W, scheme, device)` → `(W,)` float32, `w.sum() == 1`.
- `per_sample = (elem * w.unsqueeze(0)).sum(dim=1)` → `(B,)`; `loss = per_sample.mean()` → scalar.
- With `"none"`, this equals `F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight_tensor, reduction="mean")` within `1e-6`.

```python
def build_timestep_weights(window_size: int, scheme: str, device: str) -> Tensor:
    """Return timestep weights, shape (window_size,), float32, sum == 1.

    Schemes:
        "none": w_t = 1 / W (uniform).
        "linear": raw_t = (t + 1); w = raw / raw.sum(). Strictly increasing.
        "sqrt": raw_t = sqrt(t + 1); w = raw / raw.sum(). Gentler increase.

    Raises:
        ValueError: If window_size < 1, or scheme not in {none, linear, sqrt}.
    """
```
- `ValueError("window_size must be >= 1")` if `window_size < 1`; `ValueError(f"Unknown timestep_weighting scheme: {scheme}")` otherwise.
- `timestep_weighting` is ignored in standard mode (still persisted for record-keeping).

---

## 8. Edge cases — exact behavior

| Case | Behavior |
|---|---|
| `train.csv`/`test.csv` missing | `prepare_data` (§10.1) raises `FileNotFoundError(f"Processed data not found: {path}")`; CLI → `exit(1)`. |
| Non-finite values in sensor matrix | After float32 cast, assert `np.isfinite(x).all()`, else `ValueError("Sensor matrix contains non-finite values; expected median-imputed data")`. |
| `n_sensors == 0` (no sensor cols) | `prepare_data` raises `ValueError("No sensor_ columns found")`; `build_gru`/`SequenceSensorPipeline.fit` also raise `ValueError`. CLI → `exit(1)`. |
| `n_sensors == 1` | Allowed; full row and any valid window have `W=1`; early-mode final timestep == only timestep. |
| `window_size == 0` | Dataset/model/preprocessor raise `ValueError("window_size must be >= 1")` or `ValueError("Window sensor columns must be non-empty")`. |
| Unknown window sensor | `SequenceSensorPipeline.transform_window` raises `ValueError(f"Unknown window sensor column: {col}")`. |
| zero-variance column | scale element → 1.0 (§3.1); column all zeros after centering; no NaN/div-by-zero. |
| Sensor-column mismatch (train vs test) | `prepare_data` raises `ValueError("Train and test sensor columns differ")`. |
| Sensor-column mismatch (checkpoint vs eval data) | `evaluate` (§10.3) raises `ValueError(f"Sensor columns mismatch: checkpoint has {len(a)}, eval data has {len(b)}")` when lists differ in membership or order. CLI → `exit(1)`. |
| Single-class minibatch (train) | pos_weight is computed once on the full train sub-split, not per-batch; BCE on a one-class batch is valid. No special handling. |
| `compute_pos_weight` 0-pos / 0-neg / empty | Raises `ValueError` (§3.2). CLI → `exit(1)`. |
| Single-class `y_true` at metric time | `roc_auc`/`pr_auc` → `nan` with `warnings.warn`; confusion matrix kept 2x2 via `labels=[0,1]`; precision/recall/f1/balanced_accuracy still computed (§11). |
| Baseline file missing | `load_tabular_baseline` returns `None` (after fallback); comparison still written with `baseline: null`, `verdict: "no_baseline"`. No exception. |
| Baseline malformed (bad JSON / no selected row / missing field) | Returns `None`, warns to stderr; tries fallback (§12), then `None`. No exception. |
| Bad checkpoint `format_version` | `load_checkpoint` raises `ValueError`. |
| Missing checkpoint file | `load_checkpoint` raises `FileNotFoundError`. |
| CUDA requested but unavailable | `resolve_device` raises `ValueError("CUDA requested but not available")`. CLI → `exit(1)`. (§10.2.) |
| `val_size` rounds to 0 val rows | Empty val arrays; per-epoch validation skipped (log `"val split empty; skipping validation"`); preprocessing pipeline fit on all train.csv rows; training/checkpoint proceed. |
| `epochs == 0` | Loop runs zero epochs; checkpoint saved with seeded-initialized weights and `epochs=0`; no exception (smoke-test friendly). |

---

## 9. Namespaced artifact paths (no tabular collision)

Relative to `cfg.paths`:
- Checkpoint: `models_dir / "sequence_gru.pt"` (CLI-overridable).
- Test metrics JSON: `reports_dir / "sequence_model_metrics.json"`.
- Baseline comparison: `reports_dir / "sequence_model_comparison.json"` and `... .csv`.
- Training history: `reports_dir / "sequence_train_history.json"`.
- Figures: `figures_dir / "sequence_confusion_matrix.png"`, `sequence_roc_curve.png`, `sequence_precision_recall_curve.png`.

None overwrite tabular outputs.

---

## 10. Orchestration (file `sequence_train.py`)

### 10.0 Config — config-driven via `configs/sequence_config.yaml` (maintainer decision)

Hyperparameters live in a committed YAML config so each experiment run is auditable and reproducible from a versioned file (matching the repo's `config.yaml` / `model_config.yaml` convention). CLI flags override individual config values for one-off sweeps.

`TrainConfig` is the in-memory carrier of all hyperparameters and defaults:

```python
@dataclass
class TrainConfig:
    emb_dim: int = 16
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 30
    early_prediction: bool = False
    timestep_weighting: str = "none"
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 0
    val_size: float = 0.10
    window_sizes: tuple[int, ...] | None = (64, 128, 256, 512)
```

```python
def load_sequence_config(
    path: Path | str = "configs/sequence_config.yaml",
) -> TrainConfig:
    """Load TrainConfig from YAML, falling back to dataclass defaults.

    If the file does not exist, returns TrainConfig() with all defaults (the
    experiment must be runnable out-of-the-box; this intentionally differs from
    load_config/load_cost_config, which raise on a missing file). If it exists,
    parses YAML to a dict and constructs TrainConfig(**known_keys).

    Args:
        path: Path to the sequence config YAML.

    Returns:
        A TrainConfig with file values overlaid on defaults.

    Raises:
        ValueError: If the YAML contains keys not present on TrainConfig
            (message lists the unknown keys), or a value cannot be coerced to
            its field type, or timestep_weighting is not in {none, linear, sqrt}.
    """
```

Behavior:
- Missing file → `TrainConfig()` (all defaults).
- Present → `yaml.safe_load`; unknown keys → `ValueError(f"Unknown sequence config keys: {sorted(unknown)}")`; type-coercion failure → `ValueError` naming the field; `timestep_weighting` validated against `{"none","linear","sqrt"}`; `window_sizes` must be null or a non-empty list/tuple of positive integers.
- A `device` value of `"cuda"`/`"auto"`/`"cpu"` is permitted in the file but the CLI `--device` flag (default unset) overrides it; final device is resolved by `resolve_device` (§10.2).

**`configs/sequence_config.yaml` ships with the defaults written explicitly (as documentation):**

```yaml
# Hyperparameters for the experimental GRU sequence model.
# Override any value at the CLI, e.g. --epochs 50 --early-prediction.
emb_dim: 16
hidden_size: 64
num_layers: 1
dropout: 0.0
lr: 0.001
batch_size: 32
epochs: 30
early_prediction: false
timestep_weighting: none   # one of: none | linear | sqrt
seed: 42
device: cpu                # one of: cpu | cuda | auto
num_workers: 0
val_size: 0.10
window_sizes: [64, 128, 256, 512]  # prefix sizes; values above n_sensors clip to full row
```

`window_sizes` controls training-time prefix augmentation. `null` preserves the
full-row baseline for ablation. The default expands each wafer into
current-progress prefix windows using the first `k` sensors in `sensor_cols`
for each listed `k`; values larger than `n_sensors` are clipped to
`n_sensors`, so the default includes the full row on the real SECOM data.
Duplicates are removed, and at least one valid window size must remain.
Inference is not limited to these sizes; it can score any current window whose
sensor names exist in the fitted preprocessing pipeline.

**CLI override semantics.** Both scripts parse overridable hyperparameters with `default=None`. After `load_sequence_config(args.config)`, each non-`None` CLI value replaces the corresponding field via `dataclasses.replace`. `--early-prediction` is a `store_true` whose presence sets `True`; its absence leaves the config value unchanged (argparse default `None` sentinel — do not use `BooleanOptionalAction`). `--window-sizes` parses a comma-separated list of positive integers into `tuple[int, ...]`.

### 10.1 `prepare_data`

```python
@dataclass
class WindowArrays:
    x: np.ndarray          # (n_examples, W) float32, normalized
    sensor_ids: np.ndarray # (n_examples, W) int64
    y: np.ndarray          # (n_examples,) int64
    window_sensor_cols: list[str]

@dataclass
class PreparedData:
    train_windows: list[WindowArrays]
    val_windows: list[WindowArrays]    # may be empty
    x_test_raw: np.ndarray       # (n_te, n_sensors) float32, RAW (un-normalized)
    y_test: np.ndarray           # (n_te,) int64
    sensor_cols: list[str]
    preprocessor: SequenceSensorPipeline
    pos_weight: float

def prepare_data(
    processed_dir: Path,
    val_size: float,
    random_seed: int,
    window_sizes: tuple[int, ...] | None = None,
) -> PreparedData:
    """Load CSVs, split train/val, fit preprocessing, and build windows.

    Raises:
        FileNotFoundError: If a CSV is missing.
        ValueError: For empty sensor set, non-finite values, train/test
            sensor-column mismatch, or undefined pos_weight.
    """
```
Pinned order (leakage controls):
1. Read `train.csv`, `test.csv` (FileNotFoundError if absent).
2. `sensor_cols` from train column order; validate test has identical `sensor_cols` (same order) else `ValueError("Train and test sensor columns differ")`. Empty set → `ValueError("No sensor_ columns found")`.
3. Assert finiteness on all sensor matrices.
4. Stratified split of TRAIN rows into train/val via sklearn `train_test_split(test_size=val_size, stratify=label, random_state=random_seed)`. `test.csv` is the held-out test set, never split. If stratify is infeasible (a class has <2 rows) or `val_size <= 0`, val is empty and the preprocessing pipeline is fit on all train rows.
5. `SequenceSensorPipeline.fit` on the TRAIN sub-split only (after val carve-out).
6. Build train/val `WindowArrays` blocks:
   - If `window_sizes is None`, call `preprocessor.transform_full(...)` and
     create one full-row block.
   - If `window_sizes` is provided, create one fixed-width prefix block per
     valid size. Each block has shape `(n_wafers, W)` and can use normal
     DataLoader collation without padding. Training iterates all blocks.
7. `x_test_raw` kept RAW (normalized later by `evaluate` using the checkpoint preprocessor — guarantees eval-time leakage-freedom even if the caller never saw train stats).
8. `compute_pos_weight` on the train sub-split labels before window expansion, so class weighting reflects wafer counts rather than augmented window counts.

### 10.2 Training loop — fixed epochs, no early stopping (maintainer decision)

With ~21 test positives and a tiny val split, validation PR-AUC is too noisy to drive reliable early stopping; fixed `epochs` is fully deterministic and bit-reproducible. `epochs` is the sole duration control; history enables manual inspection.

```python
def resolve_device(requested: str) -> str:
    """Resolve a device string.

    "cpu" -> "cpu"; "auto" -> "cuda" if available else "cpu";
    "cuda" -> "cuda" if available else ValueError("CUDA requested but not available").
    """

def set_global_determinism(seed: int) -> None:
    """Seed random, numpy, torch (CPU+CUDA); enable deterministic algorithms.

    Calls random.seed, np.random.seed, torch.manual_seed,
    torch.cuda.manual_seed_all, torch.use_deterministic_algorithms(True,
    warn_only=True); sets torch.backends.cudnn.deterministic=True,
    torch.backends.cudnn.benchmark=False, and
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8".
    """

def train(data: PreparedData, config: TrainConfig
          ) -> tuple[SecomGRU, list[dict[str, float]]]:
    """Train a SecomGRU; return (model in eval() mode, per-epoch history).

    Raises:
        ValueError: From resolve_device on unavailable CUDA.
    """
```
Behavior:
- `set_global_determinism(config.seed)` first; `device = resolve_device(config.device)`.
- `build_gru(...)` → `.to(device)`.
- `optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)`. (Adam + lr default 1e-3: standard reproducible GRU-classification default; brief did not specify.)
- Loss per mode (§7).
- For each `WindowArrays` block in `data.train_windows`, build a
  `SecomSequenceDataset` and deterministic DataLoader. Use `seed + block_idx`
  for each loader so shuffle order is reproducible.
- For `epoch in range(config.epochs)`: `model.train()`, iterate every training
  window loader, accumulate mean train loss over `(x, sensor_ids, y)` batches;
  if val windows are non-empty, `model.eval()` + `no_grad`, evaluate every val
  window block and compute aggregate val loss / val PR-AUC (via
  `predict_logits(model, x, sensor_ids)`→sigmoid→`compute_sequence_metrics`;
  nan-guarded). Append `{"epoch", "train_loss", "val_loss", "val_pr_auc"}`
  (`val_*` = nan when val empty/single-class).
- `model.eval()`; return `(model, history)`. No early stopping; the final-epoch weights are the returned/checkpointed weights.

**CUDA-unavailable → raise.** A silent device swap can mask a misconfigured run. `--device auto` is provided for users who want best-effort selection without an error.

### 10.3 Evaluation loop

```python
def evaluate(checkpoint: LoadedCheckpoint, x_window_raw: np.ndarray, y_true: np.ndarray,
             window_sensor_cols: list[str], threshold: float = 0.5,
             device: str = "cpu") -> tuple[np.ndarray, SequenceMetrics]:
    """Compute probabilities and metrics for a raw sensor window.

    Steps:
      1. Normalize RAW x_window_raw with checkpoint.preprocessor and
         window_sensor_cols.
      2. Batch through predict_logits(model, x, sensor_ids) -> sigmoid ->
         y_prob (n_rows,) float64.
      3. metrics = compute_sequence_metrics(y_true, y_prob, threshold).

    Returns:
        (y_prob float64 (n_rows,), SequenceMetrics).

    Raises:
        ValueError: On unknown window sensor columns or shape mismatch.
    """
```

For standard held-out evaluation, the CLI passes the complete test matrix and
`checkpoint.sensor_cols`, so metrics remain comparable to the tabular baseline.
For current-window inference, callers pass only the currently available sensor
columns and raw values; the loaded preprocessor applies the training-time
normalization for exactly those sensors.

**Decision threshold — 0.5 default, CLI-overridable.** ROC-AUC/PR-AUC (the primary comparison metrics) are threshold-free, so the headline comparison is fair at any threshold. `pos_weight` rebalances the loss, so GRU logits are NOT calibrated to the tabular cost matrix; borrowing the tabular cost-optimal threshold (0.08) would be misleading. Report thresholded metrics at the neutral 0.5 operating point; expose `--threshold`. Documented in `docs/sequence_model.md`.

---

## 11. Metrics (file `sequence_train.py`; reuses `evaluate.compute_metrics`, does NOT modify `ClassificationMetrics`)

```python
@dataclass
class SequenceMetrics:
    roc_auc: float            # nan if y_true single-class
    pr_auc: float             # nan if y_true single-class
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    confusion_matrix: list[list[int]]   # [[TN, FP], [FN, TP]]
    threshold: float
    n_pos: int
    n_neg: int

def compute_sequence_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                             threshold: float = 0.5) -> SequenceMetrics:
    """Compute the full GRU metric set with a single-class guard."""
```
Behavior:
- `y_pred = (y_prob >= threshold).astype(int)`; `n_pos`/`n_neg` from `y_true`.
- If `y_true` has < 2 distinct classes: `roc_auc = pr_auc = float("nan")` and `warnings.warn("y_true single-class; ROC/PR-AUC undefined")` (do not let sklearn raise). Otherwise delegate AUCs to `yield_risk.evaluate.compute_metrics(y_true, y_prob, threshold)` and copy `roc_auc`/`pr_auc`.
- `precision`/`recall`/`f1` via sklearn with `zero_division=0` (matching `compute_metrics`).
- `confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()` — always 2x2 even when single-class.
- `balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)` — the new metric the brief requires (imbalance-robust mean of per-class recall), added without touching `ClassificationMetrics`.

```python
def save_sequence_metrics(metrics: SequenceMetrics, path: Path) -> None:
    """Write the dataclass as JSON; nan serialized as JSON null."""
```
- A `_json_safe` helper maps `nan` → `None`.
- Figures via `yield_risk.evaluate.plot_confusion_matrix`/`plot_roc_curve`/`plot_precision_recall_curve`; ROC and PR figures skipped (with a logged note) when their AUC is nan (undefined for single-class data); confusion-matrix figure always produced.

---

## 12. Baseline comparison — `model_comparison.json` PRIMARY + `model_metadata.json` FALLBACK

```python
@dataclass
class TabularBaseline:
    source: str                 # "model_comparison.json" or "model_metadata.json"
    model: str | None
    test_pr_auc: float | None
    test_roc_auc: float | None
    test_recall: float | None
    test_precision: float | None

def load_tabular_baseline(reports_dir: Path, models_dir: Path
                          ) -> TabularBaseline | None:
    """Load the existing best tabular baseline, gracefully handling absence.

    Resolution order (first success wins):
      1. reports_dir/"model_comparison.json": JSON list; pick the dict with
         selected == True; read test_pr_auc, test_roc_auc, test_recall,
         test_precision, model. source="model_comparison.json".
      2. models_dir/"model_metadata.json": read metrics.pr_auc, metrics.roc_auc,
         metrics.recall, metrics.precision and model_version (as `model`).
         source="model_metadata.json".
    Returns None if neither exists, JSON malformed, no/multiple selected rows,
    or required fields missing/non-numeric. Warns to stderr; never raises.
    """
```
Verified field mapping (ground truth): comparison row uses `test_pr_auc`/`test_roc_auc`/`test_recall`/`test_precision`/`model`; metadata uses nested `metrics.{pr_auc,roc_auc,recall,precision}` and top-level `model_version`. The `selected == true` row is the production winner (currently `random_forest`, `test_pr_auc ≈ 0.193`, `test_roc_auc ≈ 0.758`).

**Comparison output — delta + verdict.**

```python
def write_baseline_comparison(seq_metrics: SequenceMetrics,
                              baseline: TabularBaseline | None,
                              reports_dir: Path) -> None:
    """Write sequence_model_comparison.{json,csv}."""
```
JSON shape:
```json
{
  "sequence": {"pr_auc": 0.0, "roc_auc": 0.0, "recall": 0.0, "precision": 0.0,
               "balanced_accuracy": 0.0, "threshold": 0.5, "n_pos": 0, "n_neg": 0},
  "baseline": null,
  "delta_pr_auc": null,
  "delta_roc_auc": null,
  "verdict": "no_baseline"
}
```
- When baseline present, `"baseline"` is `{"source", "model", "pr_auc", "roc_auc", "recall", "precision"}`.
- `delta_pr_auc = sequence.pr_auc - baseline.pr_auc` (null if no baseline or either side nan); same for `delta_roc_auc`.
- `verdict` (on pr_auc): `"sequence_better"` / `"baseline_better"` / `"tie"` (`|delta| < 1e-6`) / `"no_baseline"` (baseline None or sequence pr_auc nan).
- CSV: one flattened row; `baseline_*` columns blank when null.
- nan → JSON null throughout.

---

## 13. CLI scripts

Both use `argparse`, fully typed, `main() -> None`, `if __name__ == "__main__": main()`. Exit codes: `FileNotFoundError`/`ValueError`/`KeyError` → stderr + `exit(1)`; baseline missing → warn + `exit(0)`; success → `exit(0)`. Both resolve paths via `load_config()`.

### 13.1 `scripts/train_sequence_model.py`

Loads `train.csv` only (never `test.csv`). Loads `TrainConfig` from `--config`, then applies CLI overrides (§10.0 semantics).

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--config` | Path | `configs/sequence_config.yaml` | Config YAML (missing → dataclass defaults) |
| `--emb-dim` | int | None→config | embedding dim |
| `--hidden-size` | int | None→config | GRU hidden size |
| `--num-layers` | int | None→config | GRU layers |
| `--dropout` | float | None→config | inter-layer dropout |
| `--lr` | float | None→config | Adam learning rate |
| `--batch-size` | int | None→config | batch size |
| `--epochs` | int | None→config | training epochs |
| `--early-prediction` | store_true | None→config | per-timestep mode |
| `--timestep-weighting` | choice{none,linear,sqrt} | None→config | early-mode weighting |
| `--seed` | int | None→config | global seed |
| `--device` | choice{cpu,cuda,auto} | None→config | compute device |
| `--val-size` | float | None→config | val carve-out fraction |
| `--window-sizes` | comma-separated ints | None→config | prefix window sizes for training augmentation |
| `--checkpoint` | Path | `models_dir/sequence_gru.pt` | output checkpoint |

Behavior: `load_config()`, `load_sequence_config(args.config)`, apply overrides, `prepare_data(processed_dir, config.val_size, config.seed, config.window_sizes)`, `train`, `save_checkpoint`, write `sequence_train_history.json`. Prints final-epoch train/val loss and the checkpoint path.

### 13.2 `scripts/evaluate_sequence_model.py`

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--checkpoint` | Path | `models_dir/sequence_gru.pt` | input checkpoint |
| `--threshold` | float | 0.5 | decision threshold |
| `--device` | choice{cpu,cuda,auto} | cpu | compute device |
| `--metrics-out` | Path | `reports_dir/sequence_model_metrics.json` | metrics JSON |
| `--comparison-out` | Path | `reports_dir/sequence_model_comparison.json` | comparison JSON (csv sibling auto) |
| `--figures-dir` | Path | `cfg.paths.figures_dir` | figure output dir |
| `--window-size` | int | None | score the first N fitted sensor columns instead of the full row |
| `--window-sensors` | comma-separated names | None | explicit sensor window; mutually exclusive with `--window-size` |
| `--no-figures` | store_true | False | skip figure generation |

Behavior: `load_checkpoint`, read `test.csv` (raw), choose a scoring window
(full fitted `sensor_cols` by default, first `--window-size` columns, or
explicit `--window-sensors`), call `evaluate` (normalizes via checkpoint
preprocessor), `save_sequence_metrics`, generate the three `sequence_*` figures
(unless `--no-figures`; ROC/PR skipped when AUC nan), `load_tabular_baseline`,
`write_baseline_comparison`. Prints a `format_report`-style summary plus a
`Balanced Acc:` line and the verdict.

---

## 14. Docs

Create `docs/sequence_model.md`:
- **Sequence assumption.** SECOM sensors have no intrinsic temporal order; we impose DataFrame column order as a pseudo-sequence and let `nn.Embedding` learn a per-sensor identity. State plainly this is an experimental hypothesis test (positional, not temporal), not a claim of true temporal structure.
- **Install:** `pip install -e .` in an environment where the required torch dependency is available.
- **Configure:** `configs/sequence_config.yaml` holds all hyperparameters; CLI flags override per run.
- **Train:** `python scripts/train_sequence_model.py --epochs 30`.
- **Evaluate:** `python scripts/evaluate_sequence_model.py`.
- **Current-window scoring:** explain that the model accepts raw sensor values
  plus the matching sensor names for the current window; the loaded checkpoint
  reuses the training-fitted `SequenceSensorPipeline` to normalize those exact
  sensors before inference.
- **Normalization:** sensor values are standardized with training-only
  `StandardScaler` statistics persisted in the checkpoint; inference never
  re-fits normalization on eval/current data.
- **Interpreting results:** read `sequence_model_comparison.json` `delta_pr_auc`/`verdict`; emphasize PR-AUC and balanced accuracy over raw accuracy given ~6.6% fail rate; note the small test positive count (~21) makes ROC-AUC noisy and PR-AUC the primary signal; explain the 0.5 threshold choice (§10.3) and the `pos_weight` rationale.
- **Early-prediction mode:** what it does, the three timestep-weighting schemes (all sum-to-1), and that eval uses the final-timestep logit.
- **Reproducibility/limitations:** `set_global_determinism` details and the cuDNN GRU determinism caveat.

Add a short "Experimental: GRU sequence model" subsection to `README.md` linking to `docs/sequence_model.md`. Do not alter existing README content.

---

## 15. Tests — two files, split by module

`tests/test_sequence_models.py` (model/dataset/preprocessing/loss/metrics core) and `tests/test_sequence_train.py` (config/prepare_data/train/checkpoint/evaluate/baseline/determinism). Class-based, pytest. Two files mirror the two source modules and the repo's per-module test convention. No real SECOM files; all synthetic and fast.

Torch is a required dependency. Do not add missing-torch skip logic or
missing-torch test classes.

Shared synthetic fixture (in `tests/conftest.py` or local): `make_synthetic(n_rows=40, n_sensors=6, n_pos=8, seed=0) -> tuple[np.ndarray, np.ndarray]` → `x` float32 `(40, 6)` with a constant column at index 0 (exercises std==0) and a signal column correlated with `y` int64 `(40,)`.

`test_sequence_models.py`:
- **TestForwardShapes:** standard `x=(4,6)`, `sensor_ids=(4,6)` → `(4,)` float32; early → `(4,6)`; shorter current window `x=(4,3)`, `sensor_ids=(4,3)` → `(4,)`; wrong ndim / mismatched shapes / empty window / out-of-range sensor ID → `ValueError`.
- **TestSequenceSensorPipeline:** constant column → scale 1.0, transformed column all zeros, no NaN/inf; full-row transform returns values and tiled IDs; window transform over a subset returns expected sensor IDs and uses saved train stats; unknown window sensor / column-count mismatch → `ValueError`; **no-leakage:** fit on a train slice, mutate val rows, transform val — mean/scale unchanged and equal train stats.
- **TestPosWeight:** `n_neg=32, n_pos=8` → `4.0`; 0-pos / 0-neg / empty → `ValueError`.
- **TestTimestepWeights:** `none` all `1/W` sum 1; `linear` strictly increasing, `w[-1] > w[0]`, sum ≈ 1; `sqrt` increasing, sum ≈ 1; unknown scheme → `ValueError`; `window_size < 1` → `ValueError`.
- **TestEarlyPredictionLoss:** target = label broadcast to `(B,W)`; switching `none`→`linear` on fixed logits/targets changes the loss; `none` equals torch mean-reduction within `1e-6`.
- **TestMetrics:** two-class synthetic → finite AUCs, `balanced_accuracy` present, 2x2 confusion matrix; single-class `y_true` → AUCs `nan` (via `math.isnan`), no exception, 2x2 matrix (via `labels=[0,1]`), precision/recall/f1/balanced_accuracy finite; JSON serialization maps nan→null; `ClassificationMetrics` still has exactly its 6 fields (guards against accidental modification).

`test_sequence_train.py`:
- **TestLoadConfig:** missing file → defaults; valid YAML overlays values including `window_sizes`; unknown key → `ValueError`; bad `timestep_weighting` or malformed `window_sizes` → `ValueError`; CLI-override merge via `dataclasses.replace` produces expected `TrainConfig`.
- **TestPrepareData:** synthetic CSVs in tmp dir → correct window block shapes, train/test sensor-col match enforced, leakage rule (preprocessor fit only on train sub-split); `window_sizes=None` produces one full-row block; `window_sizes=(2,4)` produces two fixed-width prefix blocks; missing CSV → `FileNotFoundError`; mismatched sensor cols → `ValueError`; empty sensor set → `ValueError`.
- **TestCheckpointRoundTrip:** train 1 epoch on synthetic, save, load → loaded `predict_logits(x, sensor_ids)` equals pre-save bit-for-bit (`torch.equal`) for both modes; ckpt dict has every §6.1 key with correct types and `preprocessor.sensor_cols`/`mean`/`scale` lengths == n_sensors; loaded preprocessor can normalize a shorter current window; missing-key → `KeyError`; bad `format_version` → `ValueError`; missing file → `FileNotFoundError`.
- **TestEvaluate / TestSensorMismatch:** `evaluate` with an unknown `window_sensor_cols` entry → `ValueError`; standard full-row end-to-end and shorter current-window end-to-end both produce `SequenceMetrics`.
- **TestResolveDevice:** `cpu`→`cpu`; `auto`→cpu when no CUDA; `cuda` without CUDA → `ValueError`.
- **TestBaselineComparison:** valid `model_comparison.json` with a selected row → populated `TabularBaseline` (source `model_comparison.json`), comparison JSON has numeric `delta_pr_auc` + allowed verdict; comparison absent but valid `model_metadata.json` present → fallback `TabularBaseline` (source `model_metadata.json`); both absent → `None`, `verdict: "no_baseline"`; malformed JSON → `None`, no exception.
- **TestSmoke:** end-to-end on synthetic arrays (no files): `train` 2 epochs (standard and early), `save_checkpoint`, `load_checkpoint`, `evaluate`, `compute_sequence_metrics`, `write_baseline_comparison` (no baseline) — completes, metrics finite-or-nan-guarded.
- **TestDeterminism:** two `train` runs with identical `TrainConfig` (fixed seed) on identical synthetic data → bit-identical `model_state_dict` (`torch.equal` over each parameter).
- **TestEdgeCases:** `epochs == 0` → checkpoint saved with `epochs=0`, no exception; `val_size` → 0 val rows → training proceeds, val metrics nan.

**Acceptance:** `ruff check src/ tests/ scripts/`, `mypy --strict` (with `torch.*` override), `pytest tests/test_sequence_models.py tests/test_sequence_train.py`, and the full `pytest` suite all pass with no tabular regressions (nothing existing is modified).

---

## 16. Resolved decisions and assumptions

**Maintainer-resolved (2026-06-09):**
1. **Config delivery:** config-driven via a committed `configs/sequence_config.yaml` + `load_sequence_config`, with CLI overrides (§10.0).
2. **Stopping policy:** fixed `epochs`, no early stopping (§10.2).
3. **Default model size:** `emb_dim=16`, `hidden_size=64`, `epochs=30`, `batch_size=32` (§10.0 / §4.1).
4. **Torch dependency:** torch is required for this experiment; no optional
   import guard or custom missing-torch path (§1).
5. **Current-window support:** model inputs are normalized values plus sensor
   IDs for the current window; full rows are only one supported case (§2, §4).
6. **Preprocessing reuse:** sequence inference reloads the training-fitted
   preprocessing pipeline from the checkpoint and never re-fits on eval/current
   data (§3, §10.3).

**Assumptions (brief genuinely ambiguous; resolved on the merits):**
1. Threshold = 0.5 for thresholded metrics (§10.3); `pos_weight` decorrelates GRU logits from the tabular cost matrix; AUC comparison is threshold-free.
2. Sequence preprocessing fit AFTER val carve-out, train sub-split only (§3.1, §10.1) — strictest no-leakage reading of "fit on TRAIN only."
3. Baseline = `selected==True` row of `model_comparison.json`, fallback `model_metadata.json` (§12).
4. Optimizer = Adam, lr default 1e-3 (not specified by brief).
5. Value-first feature order `[normalized_value, sensor_id_embedding]` (§4.3), per brief wording.
6. Primary comparison metric = PR-AUC (matches the tabular pipeline's imbalanced-data selection convention); recall is the documented secondary metric.
