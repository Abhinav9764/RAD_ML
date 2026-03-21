"""
tests/test_dataset_scorer.py
==============================
Tests for the scoring formula — no network, no Kaggle/UCI calls.
"""
import sys
from pathlib import Path
import tempfile
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Data_Collection_Agent"))

from utils.dataset_scorer import DatasetScorer

CFG = {
    "scoring": {
        "keyword_match_weight":  0.40,
        "row_count_weight":      0.30,
        "column_match_weight":   0.20,
        "recency_weight":        0.10,
    },
    "collection": {"min_row_threshold": 500},
}
SCORER = DatasetScorer(CFG)

SPEC = {
    "keywords":     ["housing", "price", "bedrooms"],
    "input_params": ["bedrooms", "bathrooms", "location"],
    "target_param": "price",
}


def test_keyword_score_full_match():
    meta = {"title": "housing price bedrooms dataset", "ref": "", "vote_count": 0}
    score = SCORER.score_metadata(meta, SPEC)
    assert score > 0.3


def test_keyword_score_no_match():
    meta = {"title": "random unrelated data", "ref": "", "vote_count": 0}
    score = SCORER.score_metadata(meta, SPEC)
    assert score < 0.5


def test_row_score_above_threshold():
    meta = {"title": "housing", "ref": "", "vote_count": 100,
            "num_instances": 1000}
    score = SCORER.score_metadata(meta, SPEC)
    assert score > 0.4


def test_row_score_below_threshold():
    meta = {"title": "housing", "ref": "", "vote_count": 0,
            "num_instances": 50}
    score = SCORER.score_metadata(meta, SPEC)
    # row score pulls total down
    assert score < 0.9


def test_csv_score_with_matching_columns():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df = pd.DataFrame({
            "bedrooms": range(600),
            "bathrooms": range(600),
            "location": ["city"] * 600,
            "price": range(600),
        })
        df.to_csv(f.name, index=False)
        p = Path(f.name)

    meta = {"title": "housing price", "ref": "", "vote_count": 200}
    score = SCORER.score_csv(p, meta, SPEC)
    assert score > 0.5
    assert meta["row_count"] >= 600


def test_csv_score_missing_columns():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df = pd.DataFrame({"col_a": range(600), "col_b": range(600)})
        df.to_csv(f.name, index=False)
        p = Path(f.name)

    meta = {"title": "housing", "ref": "", "vote_count": 0}
    score = SCORER.score_csv(p, meta, SPEC)
    # column score should drag total down
    assert score < 0.8
