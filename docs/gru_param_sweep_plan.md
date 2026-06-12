# GRU Parameter Sweep Plan

## Goal

Run a small, reproducible sweep of the experimental GRU sequence model to find
a stronger default configuration than the current `configs/sequence_config.yaml`
baseline.

## What to Sweep

Keep the first sweep intentionally small so it can run on CPU and produce a
clear comparison table.

| Parameter | Values |
|---|---|
| `emb_dim` | `8`, `16`, `32` |
| `hidden_size` | `32`, `64`, `128` |
| `num_layers` | `1`, `2` |
| `dropout` | `0.0` for one layer; `0.1`, `0.3` for two layers |
| `lr` | `0.001`, `0.0003` |
| `batch_size` | `32`, `64` |

Hold `epochs`, `seed`, `val_size`, and `window_sizes` fixed for the first pass.
Use the existing validation PR-AUC from training history as the primary tuning
signal, then evaluate only the best few trials on the held-out test split.

## Deliverable

Add a sweep runner that trains each configuration, stores per-trial artifacts,
and writes a ranked summary:

- `reports/sequence_sweep_results.csv`
- `reports/sequence_sweep_results.json`
- `models/sequence_sweep/<trial_id>/sequence_gru.pt`
- `reports/sequence_sweep/<trial_id>/sequence_train_history.json`

The summary should include the swept parameters, final validation loss,
validation PR-AUC, validation ROC-AUC if available, checkpoint path, and run
status.

## Selection Rule

Choose the best trial by validation PR-AUC. Break ties by lower validation loss,
then smaller model size. After selecting the winner, run the existing sequence
evaluation script against that checkpoint and compare it to the current tabular
baseline.

## Guardrails

- Reuse the existing `prepare_data`, `train`, `save_checkpoint`, and evaluation
  functions.
- Keep artifacts namespaced under `sequence_sweep` so the current
  `models/sequence_gru.pt` and `reports/sequence_train_history.json` are not
  overwritten.
- Do not add Optuna, W&B, MLflow, or another tracking dependency for the first
  sweep.
- Record failed trials in the summary instead of aborting the entire sweep.

## Done When

- A single command can run the sweep from the repo root.
- Results are ranked in CSV and JSON.
- The best checkpoint can be evaluated with the existing GRU evaluation flow.
- `ruff check src/ tests/`, `mypy src/yield_risk`, and the sequence tests pass.
