"""
Code_Generator/RAD-ML/engines/ml_engine/sagemaker_handler.py
=============================================================
Uploads train/val CSVs to S3 and launches a SageMaker XGBoost training job.
Falls back to MOCK mode when boto3 is unavailable (CI / local dev).
"""
from __future__ import annotations
import logging
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import boto3
    import sagemaker
    from sagemaker.xgboost import XGBoost
    from sagemaker.inputs import TrainingInput
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class SageMakerHandler:
    def __init__(self, config: dict):
        self._cfg   = config
        aws         = config.get("aws", {})
        sm_cfg      = config.get("sagemaker", {})
        self._role          = aws.get("sagemaker_role", "")
        self._region        = aws.get("region", "us-east-1")
        self._bucket        = aws.get("s3_bucket", "rad-ml-datasets")
        self._prefix        = aws.get("s3_prefix", "collected_data")
        self._instance_type = sm_cfg.get("instance_type", "ml.m5.large")
        self._max_run       = int(sm_cfg.get("max_runtime_seconds", 3600))
        self._fw_version    = sm_cfg.get("framework_version", "1.7-1")
        self._output_prefix = sm_cfg.get("output_path_prefix", "models")

    # ── public ────────────────────────────────────────────────────────────────
    def upload_data(self, train_path: Path, val_path: Path,
                    job_id: str) -> tuple[str, str]:
        """Upload train/val CSVs to S3, return (train_s3_uri, val_s3_uri)."""
        if not BOTO3_AVAILABLE:
            logger.warning("boto3 not available — mock S3 URIs returned.")
            return (f"s3://{self._bucket}/{self._prefix}/{job_id}/train/train.csv",
                    f"s3://{self._bucket}/{self._prefix}/{job_id}/val/val.csv")

        s3 = boto3.client("s3", region_name=self._region)
        train_key = f"{self._prefix}/{job_id}/train/{train_path.name}"
        val_key   = f"{self._prefix}/{job_id}/val/{val_path.name}"

        logger.info("Uploading train CSV to s3://%s/%s", self._bucket, train_key)
        s3.upload_file(str(train_path), self._bucket, train_key)

        logger.info("Uploading val CSV to s3://%s/%s", self._bucket, val_key)
        s3.upload_file(str(val_path), self._bucket, val_key)

        return (f"s3://{self._bucket}/{train_key}",
                f"s3://{self._bucket}/{val_key}")

    def run_training(self, s3_input_uri: str, target_column: str,
                     preprocess_result: dict | None = None) -> dict:
        """
        Launch a SageMaker XGBoost training job.
        Returns metadata: job_name, model_name, endpoint_name, s3_output.
        """
        job_name     = f"radml-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        model_name   = f"model-{job_name}"
        endpoint_name = f"ep-{job_name}"
        task         = (preprocess_result or {}).get("task_type", "regression")

        if not BOTO3_AVAILABLE:
            logger.warning("MOCK mode: SageMaker training simulated.")
            return {
                "job_name":      job_name,
                "model_name":    model_name,
                "endpoint_name": endpoint_name,
                "s3_output":     f"s3://{self._bucket}/{self._output_prefix}/{job_name}/",
                "status":        "mock_completed",
                "task_type":     task,
            }

        try:
            sm_session = sagemaker.Session(
                boto_session=boto3.Session(region_name=self._region)
            )
            output_path = f"s3://{self._bucket}/{self._output_prefix}/{job_name}/"

            # XGBoost objective
            objective = {
                "regression":     "reg:squarederror",
                "classification": "binary:logistic",
                "clustering":     "reg:squarederror",
            }.get(task, "reg:squarederror")

            xgb = XGBoost(
                entry_point="train.py",   # built-in, no custom script needed
                role=self._role,
                instance_count=1,
                instance_type=self._instance_type,
                framework_version=self._fw_version,
                output_path=output_path,
                hyperparameters={
                    "objective":       objective,
                    "num_round":       100,
                    "max_depth":       6,
                    "eta":             0.2,
                    "subsample":       0.8,
                    "colsample_bytree": 0.8,
                    "alpha":           0.1,
                    "lambda":          1.0,
                    "early_stopping_rounds": 10,
                },
                sagemaker_session=sm_session,
                max_run=self._max_run,
            )

            train_input = TrainingInput(s3_input_uri, content_type="text/csv")
            xgb.fit({"train": train_input}, job_name=job_name, wait=True)

            # Deploy endpoint
            predictor = xgb.deploy(
                initial_instance_count=1,
                instance_type=self._instance_type,
                endpoint_name=endpoint_name,
            )

            logger.info("Endpoint deployed: %s", endpoint_name)
            return {
                "job_name":       job_name,
                "model_name":     model_name,
                "endpoint_name":  endpoint_name,
                "s3_output":      output_path,
                "status":         "deployed",
                "task_type":      task,
            }
        except Exception as exc:
            logger.error("SageMaker training failed: %s", exc)
            return {
                "job_name":       job_name,
                "model_name":     model_name,
                "endpoint_name":  endpoint_name,
                "s3_output":      "",
                "status":         f"error: {exc}",
                "task_type":      task,
            }
