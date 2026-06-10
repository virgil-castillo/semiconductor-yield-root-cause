# Experimental GRU Sequence Model

This document covers the experimental PyTorch GRU model added to the
`yield_risk` package. The experiment is **isolated**: it does not modify the
tabular pipeline, its trained artifacts, or any existing scripts.

---

## Sequence assumption

SECOM sensors carry no intrinsic temporal ordering — one row is a snapshot of
all process measurements for a single wafer, not a time series. This experiment
imposes **DataFrame column order** as a pseudo-sequence and trains an
`nn.Embedding` layer to learn a per-sensor identity representation for each
position.

This is an **experimental hypothesis test**: the model explores whether a
positional (not temporal) sequence structure, combined with learned sensor
embeddings, adds predictive signal beyond flat tabular features. It makes no
claim of true temporal structure in the data. Treat all results as
exploratory.

---

## Installation

Install the package in an environment where PyTorch is already available:

```bash
pip install -e .
```

PyTorch (`torch>=2.2`) is a required dependency; CPU-only torch is sufficient.
No CUDA is needed.

---

## Configuration

All hyperparameters live in `configs/sequence_config.yaml`. The file ships
with the defaults written explicitly:

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

The config is loaded by `load_sequence_config` at the start of both CLI
scripts. Any flag passed at the command line overrides the corresponding config
value for that run only; the YAML file on disk is never modified.

---

## Training

```bash
python scripts/train_sequence_model.py --epochs 30
```

The script:

1. Loads raw SECOM data and runs a leakage-free stratified train/test split
   (same `test_size` as the tabular pipeline, from `configs/config.yaml`).
2. Carves a validation sub-split from train rows (`--val-size`, default 0.10).
3. Fits the `SequenceSensorPipeline` on the **train sub-split only**.
4. Builds prefix window blocks per `window_sizes` and trains for fixed `epochs`
   (no early stopping).
5. Writes the checkpoint to `models/sequence_gru.pt` and the per-epoch history
   to `reports/sequence_train_history.json`.

### Key training flags

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `configs/sequence_config.yaml` | Config YAML (missing file → all defaults) |
| `--epochs N` | 30 | Number of training epochs |
| `--emb-dim N` | 16 | Sensor-ID embedding dimension |
| `--hidden-size N` | 64 | GRU hidden size |
| `--num-layers N` | 1 | Number of stacked GRU layers |
| `--dropout F` | 0.0 | Inter-layer dropout |
| `--lr F` | 0.001 | Adam learning rate |
| `--batch-size N` | 32 | Mini-batch size |
| `--early-prediction` | off | Enable per-timestep early-prediction mode |
| `--timestep-weighting` | `none` | Weighting scheme for early mode: `none`, `linear`, or `sqrt` |
| `--seed N` | 42 | Global RNG seed |
| `--device` | `cpu` | Compute device: `cpu`, `cuda`, or `auto` |
| `--val-size F` | 0.10 | Validation carve-out fraction |
| `--window-sizes N[,N...]` | `64,128,256,512` | Prefix window sizes for training augmentation |
| `--checkpoint PATH` | `models/sequence_gru.pt` | Output checkpoint path |

---

## Evaluation

```bash
python scripts/evaluate_sequence_model.py
```

The script loads the checkpoint, reproduces the identical held-out test split,
scores test wafers through the checkpoint's preprocessing pipeline, and writes
metrics, figures, and a baseline comparison report.

### Key evaluation flags

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint PATH` | `models/sequence_gru.pt` | Input checkpoint |
| `--threshold F` | 0.5 | Decision threshold for thresholded metrics |
| `--device` | `cpu` | Compute device: `cpu`, `cuda`, or `auto` |
| `--metrics-out PATH` | `reports/sequence_model_metrics.json` | Metrics JSON output |
| `--comparison-out PATH` | `reports/sequence_model_comparison.json` | Comparison JSON output |
| `--figures-dir PATH` | `reports/figures/` | Directory for figure PNGs |
| `--window-size N` | — | Score the first N raw sensor columns instead of the full row |
| `--window-sensors NAME[,NAME...]` | — | Explicit raw sensor window (mutually exclusive with `--window-size`) |
| `--no-figures` | off | Skip figure generation |

### Output artifacts

| Artifact | Path |
|---|---|
| Checkpoint | `models/sequence_gru.pt` |
| Test metrics | `reports/sequence_model_metrics.json` |
| Baseline comparison | `reports/sequence_model_comparison.json` and `.csv` |
| Training history | `reports/sequence_train_history.json` |
| Confusion matrix | `reports/figures/sequence_confusion_matrix.png` |
| ROC curve | `reports/figures/sequence_roc_curve.png` |
| PR curve | `reports/figures/sequence_precision_recall_curve.png` |

---

## Current-window scoring

The model is not limited to scoring full rows. For a wafer where only the first
`k` sensors in the process flow are available, pass those `k` raw values and
their names:

```bash
# Score using the first 128 raw sensor columns
python scripts/evaluate_sequence_model.py --window-size 128

# Score using a specific set of raw sensors
python scripts/evaluate_sequence_model.py --window-sensors sensor_000,sensor_001,sensor_059
```

The loaded checkpoint owns its own `SequenceSensorPipeline`. When evaluating a
current window, the pipeline:

1. **Drops** any supplied sensor that was rejected during training (high
   missingness, low variance, or high correlation).
2. **Imputes** `NaN` values for retained sensors using the train-sub-split
   medians stored in the checkpoint.
3. **Normalizes** retained sensor values using the train-sub-split
   `StandardScaler` statistics stored in the checkpoint.

The GRU then scores the retained window. Sensor names outside the training
`raw_sensor_cols` raise a `ValueError`; supplying a window where every sensor
was dropped during training also raises a `ValueError`. Full-row evaluation is
the special case where `window_sensors == raw_sensor_cols`.

---

## Raw preprocessing pipeline

The sequence checkpoint owns its own end-to-end preprocessing; it does **not**
delegate to the tabular pipeline. All decisions are fitted exclusively on the
**train sub-split** (after the validation carve-out) and never re-fitted on
evaluation or current-window data.

The preprocessing steps, in order:

1. **High-missing filtering** — drop raw sensor columns whose missing fraction
   strictly exceeds `missing_threshold` (default from `configs/config.yaml`).
2. **Median imputation** — impute `NaN` values for surviving columns using
   per-column medians computed on the train sub-split.
3. **Low-variance filtering** — drop columns whose post-imputation variance
   is strictly below `variance_threshold`.
4. **High-correlation filtering** — drop one column from each highly correlated
   pair (upper-triangle absolute Pearson correlation strictly above
   `correlation_threshold`; the later column is dropped).
5. **StandardScaler normalization** — fit mean and scale on the final retained
   train sub-split matrix; zero-variance scale elements are replaced by 1.0.

All five steps — drop lists, medians, scaler mean, and scaler scale — are
persisted inside the checkpoint and reloaded at inference time. The ~590 raw
SECOM sensors reduce to approximately 474 retained sensors with default
thresholds (the exact count is never hard-coded and may vary with the random
seed).

---

## Interpreting results

### Primary artifact: `sequence_model_comparison.json`

The key fields are `delta_pr_auc` and `verdict`:

```json
{
  "sequence": { "pr_auc": 0.182, "roc_auc": 0.712, ... },
  "baseline":  { "source": "model_comparison.json", "model": "random_forest",
                 "pr_auc": 0.193, ... },
  "delta_pr_auc": -0.011,
  "delta_roc_auc": -0.046,
  "verdict": "baseline_better"
}
```

`verdict` is one of `sequence_better`, `baseline_better`, `tie`, or
`no_baseline`. A positive `delta_pr_auc` means the GRU outperforms the tabular
baseline on PR-AUC.

### Metric priorities

With a ~6.6% fail rate, raw accuracy is a misleading signal (the majority-class
baseline achieves ~93.4% by always predicting pass). Prioritize metrics in this
order:

1. **PR-AUC** — the primary comparison metric. Threshold-free and sensitive to
   performance on the minority positive class. This is the metric used for
   `verdict`.
2. **Balanced accuracy** — mean of per-class recall; accounts for imbalance
   without requiring a cost ratio.
3. **Recall (fail class)** — fraction of actual fails correctly flagged. Missing
   a fail is expensive.
4. **ROC-AUC** — threshold-free but less informative under severe imbalance.
   The test set has approximately 21 positive examples; ROC-AUC is sensitive to
   random variation at this scale and should be treated as a noisy secondary
   signal.

### Threshold choice

Thresholded metrics (precision, recall, F1, balanced accuracy, confusion matrix)
are reported at **0.5** by default. This is intentional: `pos_weight` in
`BCEWithLogitsLoss` rebalances the loss so the GRU trains effectively under
class imbalance, but it does **not** calibrate the output logits to the tabular
pipeline's cost matrix. Borrowing the tabular cost-optimal threshold (0.08) for
the GRU logits would be misleading. Use `--threshold` to explore other operating
points; AUC comparisons are threshold-free and remain the primary signal.

---

## Early-prediction mode

Standard mode feeds the full retained sensor window through the GRU and returns
one logit per wafer from the final hidden state.

Early-prediction mode (`--early-prediction`) emits a logit at **every
timestep**. At training time, each timestep's logit is compared to the wafer
label, and the per-timestep losses are combined using a weighted sum. This
allows the model to learn to predict failure risk from partial windows, which
mirrors the current-window scoring use case.

### Timestep weighting schemes

The `--timestep-weighting` flag controls how per-timestep losses are weighted.
All three schemes are normalized to sum to 1, so loss values remain on the same
scale regardless of window size or scheme:

| Scheme | Weight at timestep `t` (1-indexed) | Effect |
|---|---|---|
| `none` (default) | `1 / W` (uniform) | Each timestep contributes equally |
| `linear` | `t / sum(1..W)` | Later timesteps weighted more strongly; strictly increasing |
| `sqrt` | `sqrt(t) / sum(sqrt(1)..sqrt(W))` | Later timesteps up-weighted more gently; increasing but sub-linear |

At **evaluation time**, both modes use the **final-timestep logit** for all
scored metrics and threshold comparisons, so results are directly comparable
between standard and early-prediction runs.

---

## Reproducibility and limitations

### Determinism: `set_global_determinism`

At the start of every training run, `set_global_determinism(seed)` is called.
It applies the following in order:

1. `random.seed(seed)`
2. `numpy.random.seed(seed)`
3. `torch.manual_seed(seed)`
4. `torch.cuda.manual_seed_all(seed)`
5. `torch.use_deterministic_algorithms(True, warn_only=True)`
6. `torch.backends.cudnn.deterministic = True`
7. `torch.backends.cudnn.benchmark = False`
8. `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`

With the same `seed`, `configs/sequence_config.yaml`, and raw data, two
training runs on CPU produce bit-identical checkpoints.

### cuDNN GRU determinism caveat

`torch.use_deterministic_algorithms` is called with `warn_only=True` because
some cuDNN GRU kernels do not have a fully deterministic CUDA implementation.
On GPU, complete bit-reproducibility is not guaranteed even with the above
settings. CPU runs are fully deterministic.

### Other limitations

- The positional pseudo-sequence hypothesis has not been validated against a
  true temporal ordering of sensor readings. Column order is a proxy, not a
  ground truth.
- With ~21 test positives, all reported metrics have wide confidence intervals.
  Treat any single-run comparison as indicative rather than conclusive.
- The model is isolated from the tabular pipeline and does not incorporate
  cost-sensitive threshold optimization or SHAP explanations.
