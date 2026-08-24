"""
Data Validation and Quality Assurance Module.

Performs schema verification, missing value accounting, duplicate checking,
date parsing validation, and anomaly auditing across all PLAYHACK CSVs.
"""

import os
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from src.config import RAW_FILES
from src.utils import LOGGER


class DataValidator:
    """
    Validates data integrity, completeness, and value plausibility for all tables.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def validate_file_existence(self) -> Dict[str, bool]:
        """Checks if all required raw CSV files exist."""
        results = {}
        for key, filename in RAW_FILES.items():
            path = os.path.join(self.data_dir, filename)
            exists = os.path.exists(path)
            results[key] = exists
            if not exists:
                LOGGER.warning(f"Validation Check: Missing expected file '{filename}' at {path}")
            else:
                LOGGER.info(f"Validation Check: Found '{filename}'")
        return results

    @staticmethod
    def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates count and percentage of missing values per column.
        """
        missing_count = df.isnull().sum()
        total_rows = len(df)
        missing_pct = (missing_count / total_rows * 100).round(2) if total_rows > 0 else 0

        summary_df = pd.DataFrame({
            "column": df.columns,
            "missing_count": missing_count.values,
            "missing_percentage": missing_pct.values,
        })
        return summary_df[summary_df["missing_count"] > 0].sort_values(
            by="missing_count", ascending=False
        ).reset_index(drop=True)

    @staticmethod
    def check_duplicates(df: pd.DataFrame, subset: List[str]) -> int:
        """
        Counts duplicated rows based on given subset keys.
        """
        valid_subset = [col for col in subset if col in df.columns]
        if not valid_subset:
            return 0
        return int(df.duplicated(subset=valid_subset).sum())

    @staticmethod
    def check_numeric_anomalies(df: pd.DataFrame, non_negative_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Checks for infinite values, NaNs, and impossible negative numbers.
        """
        anomalies = {}
        for col in non_negative_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                min_val = float(series.min())
                max_val = float(series.max())
                neg_count = int((series < 0).sum())
                inf_count = int(np.isinf(series.to_numpy()).sum())

                if neg_count > 0 or inf_count > 0:
                    LOGGER.warning(
                        f"Anomaly in '{col}': {neg_count} negative values, {inf_count} infinite values. Min={min_val}, Max={max_val}"
                    )

                anomalies[col] = {
                    "negative_count": neg_count,
                    "infinite_count": inf_count,
                    "min_val": min_val,
                    "max_val": max_val,
                }
        return anomalies

    def run_full_validation(self, raw_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes end-to-end validation checks on all loaded tables and returns a report.
        """
        LOGGER.info("Starting comprehensive data validation...")
        report: Dict[str, Any] = {
            "file_existence": self.validate_file_existence(),
            "table_summaries": {},
        }

        for table_name, df in raw_data_dict.items():
            if df.empty:
                report["table_summaries"][table_name] = {"status": "EMPTY_OR_NOT_FOUND"}
                continue

            num_rows = len(df)
            num_cols = len(df.columns)
            unique_athletes = df["athlete_id"].nunique() if "athlete_id" in df.columns else 0
            missing_info = self.get_missing_summary(df)

            # Key duplication check
            dup_keys = ["athlete_id", "date"] if ("athlete_id" in df.columns and "date" in df.columns) else ["athlete_id"]
            if table_name == "training_sessions" and "session_id" in df.columns:
                dup_keys = ["session_id"]

            duplicates = self.check_duplicates(df, subset=dup_keys)

            # Numeric anomalies
            num_cols_list = df.select_dtypes(include=[np.number]).columns.tolist()
            anomalies = self.check_numeric_anomalies(df, num_cols_list)

            report["table_summaries"][table_name] = {
                "rows": num_rows,
                "columns": num_cols,
                "unique_athletes": unique_athletes,
                "duplicates": duplicates,
                "missing_columns_count": len(missing_info),
                "missing_summary": missing_info.to_dict(orient="records"),
                "anomalies": anomalies,
            }

            LOGGER.info(
                f"Validated '{table_name}': {num_rows} rows, {num_cols} cols, {unique_athletes} athletes, {duplicates} duplicates."
            )

        LOGGER.info("Data validation complete.")
        return report
