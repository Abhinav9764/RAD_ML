"""
Tests for KaggleCollector without making network calls.
"""
import sys
import shutil
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "Data_Collection_Agent"))

from collectors.kaggle_collector import KaggleCollector


TMP_ROOT = Path(__file__).parent / ".tmp_kaggle_tests"


def _make_workspace_dir(name: str) -> Path:
    path = TMP_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cfg(raw_dir: str) -> dict:
    return {
        "kaggle": {"username": "demo-user", "key": "demo-key"},
        "collection": {
            "max_kaggle_results": 5,
            "raw_data_dir": raw_dir,
        },
    }


def test_download_converts_tsv_from_archive():
    tmp = _make_workspace_dir("convert_tsv")
    collector = KaggleCollector(_cfg(str(tmp)))
    api = MagicMock()

    def fake_download(dataset_ref, path, unzip=False, quiet=True):
        archive_path = Path(path) / "dataset.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("ratings.tsv", "userId\tmovieId\trating\n1\t10\t4.5\n")

    api.dataset_download_files.side_effect = fake_download
    collector._get_api = MagicMock(return_value=api)

    paths = collector.download("demo/movies")

    assert len(paths) == 1
    assert paths[0].suffix == ".csv"
    df = pd.read_csv(paths[0])
    assert list(df.columns) == ["userId", "movieId", "rating"]
    assert df.iloc[0]["movieId"] == 10


def test_download_surfaces_auth_failure_reason():
    tmp = _make_workspace_dir("auth_failure")
    collector = KaggleCollector(_cfg(str(tmp)))

    def fake_get_api():
        collector.last_error = "Kaggle authentication failed."
        return None

    collector._get_api = fake_get_api

    paths = collector.download("demo/movies")

    assert paths == []
    assert "authentication failed" in collector.last_error.lower()
