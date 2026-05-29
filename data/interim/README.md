# Interim Data

Intermediate artifacts produced by `scripts/make_dataset.py`. Not committed.

| File | Description |
|------|-------------|
| `secom_combined.parquet` | Features joined with labels and timestamp |
| `secom_validated.parquet` | After schema validation and feature renaming |
| `secom_featured.parquet` | After preprocessing and feature engineering |

Run `make data` to regenerate.
