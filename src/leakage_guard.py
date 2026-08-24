"""
Structural Leakage Guard Module.

Enforces zero temporal leakage by:
1. Auditing feature column names against known future/risk keywords.
2. Enforcing strict temporal upper bounds (no data beyond obs_window_end).
3. Raising explicit, descriptive exceptions if any leakage violation is detected.
"""

from typing import List, Optional
import pandas as pd

from src.config import LEAKAGE_COLUMN_PATTERNS
from src.utils import LOGGER


class LeakageViolationError(ValueError):
    """Raised when data or columns violate the temporal leakage boundary."""
    pass


def leakage_guard(
    df: pd.DataFrame,
    allowed_targets: Optional[List[str]] = None,
    raise_exception: bool = True,
) -> List[str]:
    """
    Scans a DataFrame's columns for suspicious words indicating future knowledge.
    """
    if allowed_targets is None:
        allowed_targets = []

    suspicious_cols = []

    for col in df.columns:
        col_lower = str(col).lower()

        # Skip approved target columns or explicit observation date columns
        if col in allowed_targets or col in ["obs_window_end", "obs_end", "window_end"]:
            continue

        for pattern in LEAKAGE_COLUMN_PATTERNS:
            if pattern in col_lower:
                suspicious_cols.append(col)
                break

    if suspicious_cols:
        error_msg = (
            f"\n{'='*75}\n"
            f"CRITICAL LEAKAGE DETECTED!\n"
            f"The feature dataset contains columns with suspicious future/risk keywords:\n"
            + "\n".join([f"  - {col}" for col in suspicious_cols])
            + f"\n\nFeatures must ONLY use data from the 30-day observation window.\n"
            f"Please remove or rename these columns before training.\n"
            f"{'='*75}"
        )
        LOGGER.error(error_msg)
        if raise_exception:
            raise LeakageViolationError(error_msg)

    return suspicious_cols


def validate_date_boundary(
    df: pd.DataFrame,
    date_col: str,
    obs_end: str,
    table_name: str = "table",
) -> None:
    """
    Asserts that no record in the DataFrame has a timestamp after obs_end string 'YYYY-MM-DD'.
    """
    if df.empty or date_col not in df.columns:
        return

    obs_end_str = str(obs_end)[:10]
    future_mask = df[date_col] > obs_end_str
    future_count = int(future_mask.sum())

    if future_count > 0:
        max_found_date = df[date_col].max()
        error_msg = (
            f"\n{'='*75}\n"
            f"TEMPORAL LEAKAGE IN '{table_name}'!\n"
            f"Found {future_count} records with date > observation end ({obs_end_str}).\n"
            f"Latest date found: {max_found_date}.\n"
            f"Features must NEVER include records from the 30-day risk window.\n"
            f"{'='*75}"
        )
        LOGGER.error(error_msg)
        raise LeakageViolationError(error_msg)
