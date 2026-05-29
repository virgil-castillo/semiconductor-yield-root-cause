"""Tests for scripts/download_data.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from download_data import download_file, download_secom


class TestDownloadFile:
    """Tests for download_file."""

    def test_skips_existing_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        existing = tmp_path / "secom.data"
        existing.touch()
        with patch("download_data.urlretrieve") as mock_retrieve:
            download_file("http://example.com/secom.data", existing)
        mock_retrieve.assert_not_called()
        assert "Skipping" in capsys.readouterr().out

    def test_downloads_missing_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "secom.data"
        with patch("download_data.urlretrieve") as mock_retrieve:
            download_file("http://example.com/secom.data", dest)
        mock_retrieve.assert_called_once_with("http://example.com/secom.data", dest)

    def test_prints_progress_for_download(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dest = tmp_path / "secom.data"
        with patch("download_data.urlretrieve"):
            download_file("http://example.com/secom.data", dest)
        out = capsys.readouterr().out
        assert "secom.data" in out


class TestDownloadSecom:
    """Tests for download_secom."""

    def test_creates_raw_dir(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        with patch("download_data.urlretrieve"):
            download_secom(raw_dir)
        assert raw_dir.exists()

    def test_downloads_both_files(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        with patch("download_data.urlretrieve") as mock_retrieve:
            download_secom(raw_dir)
        assert mock_retrieve.call_count == 2

    def test_skips_already_downloaded_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "secom.data").touch()
        (raw_dir / "secom_labels.data").touch()
        with patch("download_data.urlretrieve") as mock_retrieve:
            download_secom(raw_dir)
        mock_retrieve.assert_not_called()
        out = capsys.readouterr().out
        assert out.count("Skipping") == 2

    def test_correct_urls_requested(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        with patch("download_data.urlretrieve") as mock_retrieve:
            download_secom(raw_dir)
        urls = [c.args[0] for c in mock_retrieve.call_args_list]
        assert any("secom.data" in u for u in urls)
        assert any("secom_labels.data" in u for u in urls)

    def test_files_saved_to_raw_dir(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        with patch("download_data.urlretrieve") as mock_retrieve:
            download_secom(raw_dir)
        dest_paths = [call_args.args[1] for call_args in mock_retrieve.call_args_list]
        assert all(str(raw_dir) in str(p) for p in dest_paths)
