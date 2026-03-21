"""
tests/test_dataset_merger.py
==============================
Tests for DatasetMerger — shared-column merge, union merge, threshold logic.

Note: test CSVs use non-overlapping row values so drop_duplicates()
does not collapse the merged result.
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Data_Collection_Agent"))

from utils.dataset_merger import DatasetMerger

# Use a temp dir for processed output — recreated fresh per test via fixture
import pytest

@pytest.fixture()
def proc_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d

def _cfg(proc_dir: str) -> dict:
    return {
        "collection": {
            "min_row_threshold":  500,
            "processed_data_dir": proc_dir,
        }
    }


def _make_csv(rows: int, cols: list[str], tmpdir: Path,
              start: int = 0, name: str | None = None) -> Path:
    """Write a CSV with non-overlapping integer values starting at `start`."""
    fname = name or f"test_{rows}_{start}.csv"
    p = tmpdir / fname
    pd.DataFrame({c: range(start, start + rows) for c in cols}).to_csv(p, index=False)
    return p


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_single_dataset_above_threshold(proc_dir, tmp_path):
    p = _make_csv(600, ["a", "b", "price"], tmp_path, start=0)
    result = DatasetMerger(_cfg(proc_dir)).build_final_dataset([(p, 0.9, {})], "job1")
    assert result["row_count"] == 600
    assert result["merged"] is False
    assert result["source_count"] == 1


def test_merge_two_below_threshold(proc_dir, tmp_path):
    """Two 300-row CSVs with distinct values → merged to 600 rows."""
    p1 = _make_csv(300, ["a", "b", "price"], tmp_path, start=0,   name="f1.csv")
    p2 = _make_csv(300, ["a", "b", "price"], tmp_path, start=1000, name="f2.csv")
    result = DatasetMerger(_cfg(proc_dir)).build_final_dataset(
        [(p1, 0.8, {}), (p2, 0.7, {})], "job2"
    )
    assert result["row_count"] == 600
    assert result["merged"] is True
    assert result["source_count"] == 2


def test_union_merge_no_shared_columns(proc_dir, tmp_path):
    """Two 300-row CSVs with different columns → union concat gives 600 rows."""
    p1 = _make_csv(300, ["col_x", "col_y"], tmp_path, start=0,   name="f3.csv")
    p2 = _make_csv(300, ["col_z", "col_w"], tmp_path, start=1000, name="f4.csv")
    result = DatasetMerger(_cfg(proc_dir)).build_final_dataset(
        [(p1, 0.8, {}), (p2, 0.7, {})], "job3"
    )
    # Union concat: 300 + 300 = 600 rows; 4 columns (half NaN each side)
    assert result["row_count"] == 600
    assert result["merged"] is True


def test_no_double_load_of_same_path(proc_dir, tmp_path):
    """Passing the same path twice must not inflate row count."""
    p = _make_csv(300, ["a", "b"], tmp_path, start=0)
    result = DatasetMerger(_cfg(proc_dir)).build_final_dataset(
        [(p, 0.8, {}), (p, 0.7, {})], "job4"   # same path twice
    )
    # Deduplication by path → only one file loaded → 300 rows
    assert result["row_count"] == 300


def test_columns_normalised(proc_dir, tmp_path):
    p = _make_csv(600, ["Column A", "Column-B", "Price ($)"], tmp_path)
    result = DatasetMerger(_cfg(proc_dir)).build_final_dataset([(p, 0.9, {})], "job5")
    for col in result["columns"]:
        assert col == col.lower(), f"Not lowercase: {col}"
        assert " " not in col, f"Space in col: {col}"


def test_preview_rows_ten(proc_dir, tmp_path):
    p = _make_csv(600, ["x", "y"], tmp_path)
    result = DatasetMerger(_cfg(proc_dir)).build_final_dataset([(p, 0.9, {})], "job6")
    assert len(result["preview_rows"]) == 10


def test_output_csv_exists(proc_dir, tmp_path):
    p = _make_csv(600, ["x", "y"], tmp_path)
    result = DatasetMerger(_cfg(proc_dir)).build_final_dataset([(p, 0.9, {})], "job7")
    assert Path(result["path"]).exists()


def test_raises_on_empty_input(proc_dir):
    with pytest.raises(ValueError, match="No scored CSVs"):
        DatasetMerger(_cfg(proc_dir)).build_final_dataset([], "job8")


def test_three_way_merge(proc_dir, tmp_path):
    """Three datasets all below threshold → merged to combined total."""
    p1 = _make_csv(200, ["a", "b"], tmp_path, start=0,    name="g1.csv")
    p2 = _make_csv(200, ["a", "b"], tmp_path, start=1000,  name="g2.csv")
    p3 = _make_csv(200, ["a", "b"], tmp_path, start=2000,  name="g3.csv")
    result = DatasetMerger(_cfg(proc_dir)).build_final_dataset(
        [(p1, 0.8, {}), (p2, 0.7, {}), (p3, 0.6, {})], "job9"
    )
    assert result["row_count"] == 600
    assert result["source_count"] == 3
