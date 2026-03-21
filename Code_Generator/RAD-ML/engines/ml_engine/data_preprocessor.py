"""
Code_Generator/RAD-ML/engines/ml_engine/data_preprocessor.py
=============================================================
Cleans the combined CSV and produces train/validation splits for SageMaker.
Critically: it is AWARE of the spec's input_params and target_param, so
the generated model only uses those columns — matching what the user will
provide at inference time.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config: dict):
        self._proc_dir = Path(config.get("collection", {})
                              .get("processed_data_dir", "data/processed"))
        self._proc_dir.mkdir(parents=True, exist_ok=True)

    # ── public ────────────────────────────────────────────────────────────────
    def preprocess(self, csv_path: Path, spec: dict, job_id: str) -> dict:
        """
        Parameters
        ----------
        csv_path : path to the final merged CSV
        spec     : parsed prompt spec (contains input_params, target_param, task_type)
        job_id   : for output file naming

        Returns
        -------
        {
          "train_path":  Path,
          "val_path":    Path,
          "feature_cols": [str],
          "target_col":   str,
          "task_type":    str,
          "encoders":     {col: LabelEncoder},   # for code-gen to know dtypes
          "stats":        dict,
        }
        """
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        logger.info("Loaded CSV: %d rows × %d cols", *df.shape)

        # ── resolve target column ─────────────────────────────────────────────
        target_col = self._resolve_target(df, spec)
        logger.info("Target column resolved to: %s", target_col)

        # ── resolve feature columns ───────────────────────────────────────────
        feature_cols = self._resolve_features(df, spec, target_col)
        logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

        # Keep only the columns we need
        keep = feature_cols + [target_col]
        df   = df[keep].copy()

        # ── drop rows where target is null ────────────────────────────────────
        df.dropna(subset=[target_col], inplace=True)

        # ── encode categoricals, fill numeric NaNs ────────────────────────────
        encoders: dict = {}
        for col in feature_cols:
            # Treat anything that isn't purely numeric as categorical
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            if not is_numeric:
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            else:
                median_val = float(df[col].median())
                df[col] = df[col].fillna(median_val)

        # Encode target for classification
        task = spec.get("task_type", "regression")
        if task == "classification" and df[target_col].dtype == object:
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].astype(str))
            encoders[target_col] = le

        # ── train / validation split ──────────────────────────────────────────
        train_df, val_df = train_test_split(df, test_size=0.2,
                                             random_state=42, shuffle=True)

        train_path = self._proc_dir / f"{job_id}_train.csv"
        val_path   = self._proc_dir / f"{job_id}_val.csv"

        # SageMaker XGBoost expects target as the FIRST column
        cols_ordered = [target_col] + feature_cols
        train_df[cols_ordered].to_csv(train_path, index=False, header=False)
        val_df[cols_ordered].to_csv(val_path,   index=False, header=False)

        logger.info("Train: %d rows  Val: %d rows", len(train_df), len(val_df))

        return {
            "train_path":   train_path,
            "val_path":     val_path,
            "feature_cols": feature_cols,
            "target_col":   target_col,
            "task_type":    task,
            "encoders":     encoders,
            "stats": {
                "total_rows":  len(df),
                "train_rows":  len(train_df),
                "val_rows":    len(val_df),
                "num_features": len(feature_cols),
            },
        }

    # ── internals ─────────────────────────────────────────────────────────────
    def _resolve_target(self, df: pd.DataFrame, spec: dict) -> str:
        target_hint = (spec.get("target_param") or "").lower().replace(" ", "_")
        cols_lower  = {c.lower(): c for c in df.columns}

        # Exact match
        if target_hint in cols_lower:
            return cols_lower[target_hint]

        # Partial match
        for c_low, c_orig in cols_lower.items():
            if target_hint and (target_hint in c_low or c_low in target_hint):
                return c_orig

        # Smart fallback: last numeric column (common for tabular targets)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            return numeric_cols[-1]

        # Last column
        return df.columns[-1]

    def _resolve_features(self, df: pd.DataFrame, spec: dict,
                          target_col: str) -> list[str]:
        input_hints = [p.lower().replace(" ", "_") for p in spec.get("input_params", [])]
        cols_lower  = {c.lower(): c for c in df.columns if c != target_col}
        matched: list[str] = []

        for hint in input_hints:
            for c_low, c_orig in cols_lower.items():
                if hint in c_low or c_low in hint:
                    if c_orig not in matched:
                        matched.append(c_orig)

        # If no hints matched or no hints given, use all remaining columns
        if not matched:
            matched = [c for c in df.columns if c != target_col]

        # Drop columns with > 50% nulls
        clean = []
        for c in matched:
            null_frac = df[c].isnull().mean()
            if null_frac <= 0.5:
                clean.append(c)
            else:
                logger.debug("Dropped %s (%.0f%% null)", c, null_frac * 100)

        return clean
