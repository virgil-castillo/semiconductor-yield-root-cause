# GRU Sequence Sweep — Results & Analysis

Analysis of the first GRU hyperparameter sweep defined in
[`gru_param_sweep_plan.md`](gru_param_sweep_plan.md). All 108 trials completed
successfully (`status == ok`, zero failures). Trials are ranked by **validation
PR-AUC**; the winner was then evaluated on the held-out **test** split with the
existing `evaluate_sequence_model.py` flow.

Source artifacts:

- Ranked summary: `reports/sequence_sweep_results.csv` (and `.json`)
- Per-trial checkpoints: `models/sequence_sweep/<trial_id>/sequence_gru.pt`
- Per-trial history: `reports/sequence_sweep/<trial_id>/sequence_train_history.json`
- Winner test eval: `reports/sequence_model_metrics.json`,
  `reports/sequence_model_comparison.json`

---

## Headline

The sweep found a configuration that **maximizes validation PR-AUC (0.332)** but
that gain **does not survive on the test split**: the winner scores test PR-AUC
**0.185**, essentially identical to the untuned production GRU baseline (0.185)
and still below every non-trivial tabular model. Tuning improved the noisy
validation signal but bought **no measurable test-set improvement**, and the GRU
remains the weakest of the real models on this problem.

This is the expected outcome of optimizing a small, noisy validation split (~21
positives) — see [Validation→test gap](#validationtest-generalization-gap).

---

## Sweep setup (recap)

| Parameter | Values swept |
|---|---|
| `emb_dim` | 8, 16, 32 |
| `hidden_size` | 32, 64, 128 |
| `num_layers` | 1, 2 |
| `dropout` | 0.0 (1-layer only); 0.1, 0.3 (2-layer only) |
| `lr` | 0.001, 0.0003 |
| `batch_size` | 32, 64 |

`epochs`, `seed`, `val_size`, and `window_sizes` were held fixed. **Note the
confound:** `dropout` is tied to `num_layers` by design (dropout 0.0 appears
only on 1-layer trials, 0.1/0.3 only on 2-layer trials), so the marginal effects
of those two parameters cannot be cleanly separated.

Validation PR-AUC across the 108 trials: min **0.081**, median **0.133**, mean
**0.143**, max **0.332**.

---

## Winning trial

**`trial_025`** — selected by the plan's rule (highest val PR-AUC, ties broken by
lower val loss then smaller model).

| Hyperparameter | Value |
|---|---|
| `emb_dim` | 8 |
| `hidden_size` | 128 |
| `num_layers` | 1 |
| `dropout` | 0.0 |
| `lr` | 0.001 |
| `batch_size` | 64 |
| Parameters | 55,057 |

| Metric | Validation | Test |
|---|---|---|
| PR-AUC | **0.332** | **0.185** |
| ROC-AUC | 0.725 | 0.610 |
| Recall (fail) | — | 0.238 |
| Precision (fail) | — | 0.119 |
| Balanced accuracy | — | 0.556 |

Top 10 trials by validation PR-AUC:

| Rank | Trial | emb | hidden | layers | dropout | lr | batch | val PR-AUC | val ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | trial_025 | 8 | 128 | 1 | 0.0 | 0.001 | 64 | 0.332 | 0.725 |
| 2 | trial_028 | 8 | 128 | 2 | 0.1 | 0.001 | 32 | 0.318 | 0.794 |
| 3 | trial_032 | 8 | 128 | 2 | 0.3 | 0.001 | 32 | 0.303 | 0.715 |
| 4 | trial_029 | 8 | 128 | 2 | 0.1 | 0.001 | 64 | 0.285 | 0.703 |
| 5 | trial_061 | 16 | 128 | 1 | 0.0 | 0.001 | 64 | 0.259 | 0.700 |
| 6 | trial_100 | 32 | 128 | 2 | 0.1 | 0.001 | 32 | 0.258 | 0.686 |
| 7 | trial_096 | 32 | 128 | 1 | 0.0 | 0.001 | 32 | 0.251 | 0.718 |
| 8 | trial_068 | 16 | 128 | 2 | 0.3 | 0.001 | 32 | 0.250 | 0.647 |
| 9 | trial_065 | 16 | 128 | 2 | 0.1 | 0.001 | 64 | 0.208 | 0.637 |
| 10 | trial_031 | 8 | 128 | 2 | 0.1 | 0.0003 | 64 | 0.206 | 0.729 |

The top 10 is dominated by two settings: **`hidden_size = 128`** (9 of 10) and
**`lr = 0.001`** (9 of 10).

---

## Hyperparameter trends

Marginal validation PR-AUC (mean / median / max) over each parameter's swept
values, and the Spearman rank correlation with val PR-AUC across all 108 trials.

### `hidden_size` — strongest lever (Spearman +0.24)

| hidden_size | mean | median | max |
|---|---|---|---|
| 32 | 0.134 | 0.135 | 0.186 |
| 64 | 0.119 | 0.115 | 0.177 |
| 128 | **0.175** | **0.154** | **0.332** |

128 is clearly best and owns every top trial. Note 64 is slightly *worse* than
32 on the mean — the relationship is not monotonic, but the widest capacity wins
both on average and at the top.

### `lr` — second lever (Spearman +0.17)

| lr | mean | median | max |
|---|---|---|---|
| 0.0003 | 0.130 | 0.126 | 0.206 |
| 0.001 | **0.155** | 0.139 | **0.332** |

The higher learning rate (0.001) is uniformly better here; with fixed epochs and
no early stopping, the slower 0.0003 runs likely under-train. Worth pushing the
LR ceiling higher in a follow-up.

### `num_layers` / `dropout` — confounded (Spearman +0.19 / +0.16)

| num_layers | mean | median | max |
|---|---|---|---|
| 1 | 0.134 | 0.126 | **0.332** |
| 2 | 0.147 | 0.139 | 0.318 |

| dropout | mean | median | max |
|---|---|---|---|
| 0.0 (1-layer) | 0.134 | 0.126 | **0.332** |
| 0.1 (2-layer) | 0.149 | 0.142 | 0.318 |
| 0.3 (2-layer) | 0.145 | 0.137 | 0.303 |

Two layers edges out one on the mean, but the single best trial is one layer.
Because dropout 0.0 only ever runs on 1-layer trials, the dropout and depth
columns above are the *same split viewed twice* and cannot be disentangled. A
follow-up should cross dropout with depth to separate them.

### `emb_dim` — weak, slightly favors small (Spearman −0.04)

| emb_dim | mean | median | max |
|---|---|---|---|
| 8 | **0.149** | 0.125 | **0.332** |
| 16 | 0.142 | 0.138 | 0.259 |
| 32 | 0.137 | 0.131 | 0.258 |

Smallest embedding (8) holds the top trial and the best mean. A larger sensor-ID
embedding does not help — consistent with the embedding learning little beyond a
compact per-position identity.

### `batch_size` — negligible (Spearman −0.06)

| batch_size | mean | median | max |
|---|---|---|---|
| 32 | 0.145 | 0.136 | 0.318 |
| 64 | 0.140 | 0.128 | **0.332** |

Essentially a wash.

**Takeaway:** capacity (`hidden_size = 128`) and a healthy learning rate
(`lr = 0.001`) are the only two parameters that move the validation needle.
Everything else is within noise.

---

## Cross-model comparison (test split)

All numbers are on the **same held-out test split** (~21 positives / 293
negatives). Tabular numbers from `reports/model_comparison.csv`; GRU from the
winner's test eval.

| Model | test PR-AUC | test ROC-AUC | recall (fail) | precision (fail) |
|---|---|---|---|---|
| **xgboost** | **0.261** | **0.802** | 0.667 | 0.222 |
| logistic_regression | 0.210 | 0.669 | 0.571 | 0.130 |
| random_forest *(selected)* | 0.193 | 0.758 | 0.571 | 0.171 |
| **GRU winner (trial_025)** | **0.185** | **0.610** | **0.238** | 0.119 |
| dummy | 0.066 | 0.490 | 0.048 | 0.048 |

The tuned GRU is the **weakest non-trivial model** on test PR-AUC and ROC-AUC,
and its recall (0.238) is dramatically lower than every tree/linear model
(0.57–0.67) at the 0.5 threshold. The official comparison verdict is
`baseline_better` (Δ PR-AUC −0.008 vs random_forest, Δ ROC-AUC −0.149).

> Caveat: the GRU threshold metrics are reported at 0.5 (its logits are not
> cost-calibrated to the tabular threshold), so the recall gap partly reflects
> operating point, not just ranking quality. The threshold-free AUCs are the
> fair comparison — and the GRU loses on both.

---

## Validation→test generalization gap

The winner's validation PR-AUC (0.332) collapses to 0.185 on test — a drop of
**−0.147**. Three points explain this and bound how much to trust the sweep:

1. **Tiny, noisy selection signal.** With ~21 positives in each split, PR-AUC has
   a wide confidence interval. Ranking 108 trials on such a noisy metric makes
   the top result partly a draw of the variance — we are selecting for
   validation-split luck as much as for genuine configuration quality.
2. **No test-set gain from tuning.** The tuned winner (test PR-AUC 0.185) matches
   the *untuned* production GRU baseline (0.185) almost exactly. Whatever the
   sweep optimized did not transfer.
3. **The architecture, not the hyperparameters, is the ceiling.** The spread of
   the whole sweep tops out well below xgboost. No point in this grid closes the
   gap to the tabular models.

---

## Known limitation: windowing was held fixed and never evaluated

This first sweep tuned the architecture while holding the most sequence-specific
knob constant. Two consequences:

- **`window_sizes` was not swept.** All 108 trials used the fixed augmentation
  set `(64, 128, 256, 512)` (clipping to ~474 retained sensors). The grid never
  explored whether a different prefix-augmentation set trains a better model — a
  lever plausibly as impactful as `hidden_size`.
- **Every trial — including the winner's test eval — was scored on the full
  row.** The prefix design's headline capability (*current-window scoring*:
  predicting from only the first *k* sensors, per
  [`sequence_model.md`](sequence_model.md)) was never measured. We do not know
  how gracefully these models degrade on partial sensor windows.

The full-row test numbers above remain a fair architecture-vs-architecture
comparison (the tabular models also use all features). But the windowing question
is open, so we ran a follow-up **eval-time window sweep**.

### Window sweep results

`scripts/run_window_sweep.py` scores selected top checkpoints across a grid of
`--window-size` values (first *N* raw sensor columns) on the test split and
writes `reports/sequence_window_sweep_results.csv`. No model is retrained. Three
distinct hyperparameter families were used: `trial_025` (shallow + fast),
`trial_028` (deep + fast), `trial_031` (deep + slow, least overfit).

Test PR-AUC by window (bold = best window per family; `full` is the ~474-sensor
retained row used everywhere in the main sweep):

| Family | 64 | 128 | 256 | 384 | 512 | full |
|---|---|---|---|---|---|---|
| trial_025 (shallow+fast) | 0.076 | 0.211 | **0.214** | 0.167 | 0.190 | 0.185 |
| trial_028 (deep+fast) | 0.088 | **0.184** | 0.123 | 0.117 | 0.129 | 0.156 |
| trial_031 (deep+slow) | 0.095 | 0.107 | **0.174** | 0.165 | 0.124 | 0.136 |

Findings:

1. **Window is a real lever.** For `trial_025` it swings test PR-AUC from 0.076
   (at 64) to 0.214 (at 256) — a larger range than any *hyperparameter* moved in
   the main sweep.
2. **Full-row scoring is sub-optimal for all three families.** Every family peaks
   at a *partial* window (256, 128, 256), not the full row. The headline 0.185
   for the winner understates what its weights can do — at window 256 it reaches
   0.214 and edges past random_forest's test PR-AUC (0.193).
3. **Very short windows starve the model.** 64 sensors is uniformly poor
   (PR-AUC ~0.08, ROC-AUC ~0.51–0.60); the sweet spot is mid-range (128–256).
4. **`trial_031` is the better-behaved model.** Despite a lower PR-AUC, the
   deep + slow (least overfit) family has by far the best recall (0.76–0.81 at
   256+ vs ~0.24–0.62 for the others) and the highest GRU ROC-AUC (0.722 at 256,
   approaching random_forest's 0.758). PR-AUC ranking alone hid this.

**Critical caveat — do not read "256 is best" as a result.** Picking the window
by its *test* PR-AUC is selection on the test set (the same multiplicity trap as
choosing a model by test). The honest conclusion is narrower: *full-row is
demonstrably not the optimal operating point, and window is a first-class lever
the main sweep ignored.* To actually choose a window, select it on
validation/CV and confirm on test once.

A larger *training*-augmentation sweep (varying `window_sizes` and retraining
each family, with window selected on validation) is the natural next step and
remains a separate, heavier follow-up.

---

## Recommendation

- **Do not promote** the swept GRU to production. The selected tabular model
  (random_forest) and the strongest tabular model (xgboost) both beat it on the
  test metrics that matter, and tuning produced no real test-set lift.
- **Keep the experiment isolated.** The sweep confirms the positional
  pseudo-sequence hypothesis adds no signal beyond flat tabular features on
  SECOM, consistent with [`sequence_model.md`](sequence_model.md).
- **If pursued further**, the only directions the data supports are *more
  capacity* (`hidden_size` > 128) and a *higher learning rate* search, plus a
  redesigned grid that **decouples dropout from depth** and uses
  cross-validation or repeated seeds instead of a single noisy validation split
  for selection. Manage expectations: the cross-model gap suggests architecture,
  not tuning, is the limiting factor.
