# Raw Data

Raw SECOM dataset files are not committed to this repository.

## Acquisition

1. Visit the UCI Machine Learning Repository:
   https://archive.ics.uci.edu/ml/datasets/SECOM

2. Download:
   - `secom.data` — 1567 × 590 sensor measurements (space-delimited, NaN for missing)
   - `secom_labels.data` — 1567 × 2 (label: -1 = pass / 1 = fail; timestamp)

3. Place both files in this directory, then run:
   ```bash
   make data
   ```

## Citation

> McCann, M. & Johnston, A. (2008). SECOM Dataset. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5SG6K
