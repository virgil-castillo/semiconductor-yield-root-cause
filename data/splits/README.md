# Data Splits

Raw stratified train/test splits produced by `scripts/split_data.py`.

These hold RAW rows: NaNs are intact and every sensor column is retained.
Imputation and feature selection happen later, inside the model pipeline, so
they refit per CV fold.

| File | Description |
|------|-------------|
| `train.csv` | Training rows (stratified split, raw) |
| `test.csv` | Test rows (stratified split, raw) |

Run `python scripts/split_data.py` to regenerate.
