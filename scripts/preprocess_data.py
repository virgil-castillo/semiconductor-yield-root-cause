"""Load raw SECOM data, preprocess, and save train/test splits."""
from __future__ import annotations

from yield_risk.config import load_config
from yield_risk.data import load_secom
from yield_risk.preprocess import run_preprocessing
from yield_risk.validation import validate_secom


def main() -> None:
    """Load raw SECOM data, preprocess, and save train/test splits."""
    cfg = load_config()
    raw = load_secom(cfg.paths.raw_dir)
    validate_secom(raw)
    train, test = run_preprocessing(raw, cfg.run)
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(cfg.paths.processed_dir / "train.csv", index=False)
    test.to_csv(cfg.paths.processed_dir / "test.csv", index=False)
    sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
    print(f"Features retained: {len(sensor_cols)}")
    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Train class distribution: {train['label'].value_counts().to_dict()}")
    print(f"Test class distribution: {test['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
