"""
Utility Functions for Logging, Column Resolution, and Data Processing.

Provides beginner-friendly helpers with detailed error explanations.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import COLUMN_ALIASES


# ==============================================================================
# 1. LOGGING SETUP
# ==============================================================================
def setup_logger(name: str = "playhack", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a formatted console logger for pipeline tracking.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


# ==============================================================================
# 2. REPRODUCIBILITY
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across standard Python, NumPy, and environments.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==============================================================================
# 3. ROBUST COLUMN FINDER & LIGHTNING-FAST DATE PARSER
# ==============================================================================
def find_col(
    df: pd.DataFrame,
    alias_key: str,
    required: bool = True,
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Finds a matching column in a DataFrame using alias keys from src/config.py.

    If the column cannot be found and required=True, raises a clear ValueError
    listing the expected aliases and available columns.
    """
    aliases = COLUMN_ALIASES.get(alias_key, [alias_key])

    # 1. Check exact match
    for candidate in aliases:
        if candidate in df.columns:
            return candidate

    # 2. Check case-insensitive match
    df_cols_lower = {col.lower(): col for col in df.columns}
    for candidate in aliases:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]

    if required:
        available_cols = list(df.columns)
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: Could not find the '{alias_key}' column.\n\n"
            f"Expected one of these aliases:\n"
            + "\n".join([f"  - {a}" for a in aliases])
            + f"\n\nAvailable columns in dataset:\n"
            + f"  {available_cols}\n\n"
            f"Please update COLUMN_ALIASES in 'src/config.py'.\n"
            f"{'='*70}"
        )
        raise KeyError(error_msg)

    return default


def parse_date_series(series: pd.Series) -> pd.Series:
    """
    Rapidly parses a pandas Series to datetime64[ns] using vectorized NumPy array conversion.
    """
    if series.empty or pd.api.types.is_datetime64_any_dtype(series):
        return series

    arr = series.fillna("").astype(str).to_numpy(dtype=str)
    sample = arr[0] if len(arr) > 0 else ""

    if len(sample) >= 10 and sample[4] == "-" and sample[7] == "-":
        if "T" in sample:
            parsed = pd.to_datetime(arr, format="%Y-%m-%dT%H:%M:%S", errors="coerce")
        else:
            parsed = pd.to_datetime(arr, format="%Y-%m-%d", errors="coerce")
    else:
        parsed = pd.to_datetime(arr, errors="coerce")

    return pd.Series(parsed, index=series.index)


# ==============================================================================
# 4. JSON & SERIALIZATION HELPERS
# ==============================================================================
class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy and Pandas types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, pd.Period)):
            return str(obj)
        return super().default(obj)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Saves a dictionary to JSON with formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NpEncoder)


def load_json(filepath: str) -> Dict[str, Any]:
    """Loads a dictionary from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
