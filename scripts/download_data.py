"""Download SECOM dataset files from UCI Machine Learning Repository."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

_BASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
)
_FILES = ["secom.data", "secom_labels.data"]


def download_file(url: str, dest: Path) -> None:
    """Download a single file, skipping if it already exists.

    Args:
        url: Source URL to fetch.
        dest: Local destination path.
    """
    if dest.exists():
        print(f"Skipping {dest.name} (already exists)")
        return
    print(f"Downloading {dest.name} ...")
    urlretrieve(url, dest)
    print(f"Saved {dest.name} -> {dest}")


def download_secom(raw_dir: Path = Path("data/raw")) -> None:
    """Download all SECOM data files into *raw_dir*.

    Skips any file that already exists on disk.

    Args:
        raw_dir: Directory to save downloaded files (created if absent).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in _FILES:
        download_file(_BASE_URL + filename, raw_dir / filename)


def main() -> None:
    """Entry point: download SECOM data files to data/raw/."""
    download_secom()


if __name__ == "__main__":
    main()
