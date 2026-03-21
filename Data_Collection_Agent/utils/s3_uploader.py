"""
utils/s3_uploader.py
=====================
Uploads the final dataset CSV to S3.
Gracefully disabled when boto3 / credentials are absent.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import boto3
    _BOTO3 = True
except ImportError:
    _BOTO3 = False


class S3Uploader:
    def __init__(self, config: dict):
        aws = config.get("aws", {})
        self._bucket  = aws.get("s3_bucket", "rad-ml-datasets")
        self._prefix  = aws.get("s3_prefix", "collected_data")
        self._region  = aws.get("region", "us-east-1")
        self._client  = None
        self._enabled = _BOTO3

        if self._enabled:
            try:
                kw: dict = {"region_name": self._region}
                if aws.get("access_key_id"):
                    kw["aws_access_key_id"]     = aws["access_key_id"]
                    kw["aws_secret_access_key"] = aws["secret_access_key"]
                self._client = boto3.client("s3", **kw)
                logger.info("S3Uploader ready — bucket: %s", self._bucket)
            except Exception as exc:
                logger.warning("S3 init failed (%s) — uploads disabled.", exc)
                self._enabled = False

    # ── public ────────────────────────────────────────────────────────────────
    def upload_dataset(self, local_path: Path, job_id: str) -> str | None:
        """Upload the final CSV and return its s3:// URI, or None if disabled."""
        if not self._enabled:
            logger.info("S3 disabled — dataset not uploaded.")
            return None
        key = f"{self._prefix}/datasets/{job_id}/{local_path.name}"
        return self._upload(local_path, key)

    def upload_results_json(self, payload: dict, job_id: str) -> str | None:
        if not self._enabled:
            return None
        key = f"{self._prefix}/{job_id}/db_results.json"
        try:
            self._client.put_object(
                Bucket=self._bucket, Key=key,
                Body=json.dumps(payload, indent=2, default=str).encode(),
                ContentType="application/json",
            )
            uri = f"s3://{self._bucket}/{key}"
            logger.info("Uploaded results JSON → %s", uri)
            return uri
        except Exception as exc:
            logger.error("JSON upload failed: %s", exc)
            return None

    def _upload(self, path: Path, key: str) -> str | None:
        try:
            self._client.upload_file(str(path), self._bucket, key)
            uri = f"s3://{self._bucket}/{key}"
            logger.info("Uploaded %s → %s", path.name, uri)
            return uri
        except Exception as exc:
            logger.error("Upload failed for %s: %s", path, exc)
            return None
