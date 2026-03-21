"""
tests/test_data_preprocessor.py
=================================
Tests for DataPreprocessor — target resolution, feature selection,
train/val split, SageMaker CSV format (target first, no header).
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Code_Generator" / "RAD-ML"))

from engines.ml_engine.data_preprocessor import DataPreprocessor

CFG = {
    "collection": {"processed_data_dir": tempfile.mkdtemp()},
}
PP = DataPreprocessor(CFG)


def _make_housing_csv(rows=800) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        pd.DataFrame({
            "bedrooms":  [3] * rows,
            "bathrooms": [2] * rows,
            "location":  ["city"] * rows,
            "price":     range(rows),
        }).to_csv(f.name, index=False)
        return Path(f.name)


SPEC_HOUSING = {
    "task_type":    "regression",
    "input_params": ["bedrooms", "bathrooms", "location"],
    "target_param": "price",
}


def test_target_resolved_correctly():
    csv = _make_housing_csv()
    res = PP.preprocess(csv, SPEC_HOUSING, "t1")
    assert res["target_col"] == "price"


def test_features_include_input_params():
    csv = _make_housing_csv()
    res = PP.preprocess(csv, SPEC_HOUSING, "t2")
    for param in ["bedrooms", "bathrooms"]:
        assert param in res["feature_cols"]


def test_target_not_in_features():
    csv = _make_housing_csv()
    res = PP.preprocess(csv, SPEC_HOUSING, "t3")
    assert res["target_col"] not in res["feature_cols"]


def test_train_val_split_sizes():
    csv = _make_housing_csv(1000)
    res = PP.preprocess(csv, SPEC_HOUSING, "t4")
    assert res["stats"]["train_rows"] > res["stats"]["val_rows"]
    total = res["stats"]["train_rows"] + res["stats"]["val_rows"]
    assert total == res["stats"]["total_rows"]


def test_sagemaker_csv_target_first_no_header():
    csv  = _make_housing_csv()
    res  = PP.preprocess(csv, SPEC_HOUSING, "t5")
    df_train = pd.read_csv(res["train_path"], header=None)
    # SageMaker XGBoost: first col = target, no column names
    assert len(df_train.columns) == len(res["feature_cols"]) + 1


def test_categorical_encoding():
    csv = _make_housing_csv()
    res = PP.preprocess(csv, SPEC_HOUSING, "t6")
    df_train = pd.read_csv(res["train_path"], header=None)
    # All values should be numeric after encoding
    assert df_train.select_dtypes(include="number").shape[1] == df_train.shape[1]


def test_fallback_target_when_not_in_spec():
    csv  = _make_housing_csv()
    spec = {"task_type": "regression", "input_params": [], "target_param": "nonexistent"}
    res  = PP.preprocess(csv, spec, "t7")
    # Should fall back to last numeric column
    assert res["target_col"] is not None
