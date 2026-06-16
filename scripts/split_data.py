"""Load raw SECOM data and split it into raw train/test artifacts."""
from __future__ import annotations

from yield_risk.config import load_config
from yield_risk.data import load_secom
from yield_risk.preprocess import split_train_test
from yield_risk.validation import validate_secom


def main() -> None:
    """Load raw SECOM data and write raw stratified train/test splits.

    The splits are written with NaNs intact and all sensor columns retained.
    Imputation and feature selection happen later, inside the model pipeline,
    so they refit per CV fold and stay leak-free.
    """
    cfg = load_config()
    raw = load_secom(cfg.paths.raw_dir)
    validate_secom(raw)
    train, test = split_train_test(raw, cfg.run)
    cfg.paths.splits_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(cfg.paths.splits_dir / "train.csv", index=False)
    test.to_csv(cfg.paths.splits_dir / "test.csv", index=False)
    sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
    print(f"Features retained: {len(sensor_cols)}")
    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Train class distribution: {train['label'].value_counts().to_dict()}")
    print(f"Test class distribution: {test['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
