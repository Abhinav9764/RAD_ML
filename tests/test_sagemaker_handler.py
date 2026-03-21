"""
tests/test_sagemaker_handler.py
=================================
Tests SageMaker handler in MOCK mode (no AWS credentials required).
"""
import sys
from pathlib import Path
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Code_Generator" / "RAD-ML"))

import engines.ml_engine.sagemaker_handler as sm_mod
from engines.ml_engine.sagemaker_handler import SageMakerHandler

CFG = {
    "aws": {
        "region":          "us-east-1",
        "s3_bucket":       "rad-ml-test",
        "s3_prefix":       "test",
        "sagemaker_role":  "arn:aws:iam::000000000000:role/FakeRole",
    },
    "sagemaker": {
        "instance_type":       "ml.m5.large",
        "max_runtime_seconds": 60,
        "framework_version":   "1.7-1",
        "output_path_prefix":  "models",
    },
}


def test_mock_training_returns_metadata(monkeypatch):
    monkeypatch.setattr(sm_mod, "BOTO3_AVAILABLE", False)
    handler = SageMakerHandler(CFG)
    meta = handler.run_training(
        s3_input_uri      = "s3://rad-ml-test/collected_data/job1/train/train.csv",
        target_column     = "price",
        preprocess_result = {"task_type": "regression"},
    )
    assert meta["endpoint_name"]
    assert meta["model_name"]
    assert meta["job_name"]
    assert meta["status"] == "mock_completed"


def test_mock_upload_returns_uris(monkeypatch, tmp_path):
    monkeypatch.setattr(sm_mod, "BOTO3_AVAILABLE", False)
    handler = SageMakerHandler(CFG)

    train = tmp_path / "train.csv"
    val   = tmp_path / "val.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(train, index=False)
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(val,   index=False)

    t_uri, v_uri = handler.upload_data(train, val, "job1")
    assert t_uri.startswith("s3://")
    assert v_uri.startswith("s3://")
