"""
tests/test_openml_collector.py
================================
Tests for OpenMLCollector — mocks openml.datasets so no network needed.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Data_Collection_Agent"))

from collectors.openml_collector import OpenMLCollector

CFG = {
    "openml": {"api_key": "", "enabled": True},
    "collection": {
        "max_openml_results": 5,
        "min_row_threshold": 500,
        "raw_data_dir": tempfile.mkdtemp(),
    },
}


def _make_mock_df():
    """Return a mock dataframe like openml.datasets.list_datasets returns."""
    import pandas as pd
    return pd.DataFrame({
        "name": ["housing prices", "movie lens", "iris dataset"],
        "NumberOfInstances": [1000, 800, 150],
        "NumberOfFeatures":  [10,   20,  4],
        "NumberOfMissingValues": [0, 5, 0],
    }, index=[42, 1001, 61])


@patch("openml.datasets.list_datasets", return_value=_make_mock_df())
def test_search_returns_metadata(mock_list):
    col = OpenMLCollector(CFG)
    results = col.search("housing price prediction")
    assert len(results) > 0
    assert results[0]["source"] == "openml"
    assert "ref" in results[0]
    assert "title" in results[0]
    assert "num_instances" in results[0]


@patch("openml.datasets.list_datasets", return_value=_make_mock_df())
def test_search_ranks_by_keyword_match(mock_list):
    col = OpenMLCollector(CFG)
    results = col.search("housing price")
    # "housing prices" dataset should rank first (most keyword hits)
    assert "housing" in results[0]["title"].lower()


@patch("openml.datasets.list_datasets", return_value=_make_mock_df())
def test_search_respects_max_results(mock_list):
    col = OpenMLCollector(CFG)
    results = col.search("data")
    assert len(results) <= CFG["collection"]["max_openml_results"]


@patch("openml.datasets.list_datasets", side_effect=Exception("network error"))
def test_search_handles_failure_gracefully(mock_list):
    col = OpenMLCollector(CFG)
    results = col.search("anything")
    assert results == []


def test_download_uses_cache():
    """If cached CSV exists, download should return it without calling openml."""
    with tempfile.TemporaryDirectory() as d:
        cfg = dict(CFG)
        cfg["collection"] = dict(CFG["collection"])
        cfg["collection"]["raw_data_dir"] = d

        col = OpenMLCollector(cfg)
        # Pre-create the cache file
        dest = Path(d) / "openml_42" / "data.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(dest, index=False)

        with patch("openml.datasets.get_dataset") as mock_get:
            paths = col.download("42")
            mock_get.assert_not_called()  # should use cache
        assert len(paths) == 1
        assert paths[0] == dest


def test_metadata_fields_complete():
    """Every metadata dict must have fields required by DatasetScorer."""
    import pandas as pd
    mock_df = pd.DataFrame({
        "name": ["test dataset"],
        "NumberOfInstances": [600],
        "NumberOfFeatures": [8],
        "NumberOfMissingValues": [0],
    }, index=[999])

    with patch("openml.datasets.list_datasets", return_value=mock_df):
        col = OpenMLCollector(CFG)
        results = col.search("test")

    required = {"source", "ref", "title", "url", "size_mb",
                "vote_count", "num_instances", "num_features"}
    for meta in results:
        assert required.issubset(set(meta.keys())), \
            f"Missing fields: {required - set(meta.keys())}"
