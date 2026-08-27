"""
CLI Entrypoint for PLAYHACK Feature Engineering Pipeline.

Usage:
    python feature_engineering.py --data_dir ./data --out outputs/features.parquet
"""

import argparse
import os
import sys

# Ensure root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DEFAULT_OBS_END, FEATURES_PARQUET_PATH
from src.data_loader import load_all_raw_data
from src.data_validation import DataValidator
from src.feature_engineering import combine_features
from src.utils import LOGGER


def main():
    parser = argparse.ArgumentParser(description="PLAYHACK Leakage-Safe Feature Engineering Pipeline")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing the raw PLAYHACK CSV files.",
    )
    parser.add_argument(
        "--obs_end",
        type=str,
        default=DEFAULT_OBS_END,
        help=f"Observation window end date (default: {DEFAULT_OBS_END}).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=FEATURES_PARQUET_PATH,
        help=f"Output file path for features parquet (default: {FEATURES_PARQUET_PATH}).",
    )
    args = parser.parse_args()

    LOGGER.info("=" * 70)
    LOGGER.info("STARTING PLAYHACK FEATURE ENGINEERING PIPELINE")
    LOGGER.info(f"Data directory: {args.data_dir}")
    LOGGER.info(f"Observation window anchor: {args.obs_end}")
    LOGGER.info(f"Output path: {args.out}")
    LOGGER.info("=" * 70)

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    # [1-4/8] Load Raw Data
    LOGGER.info("[1/8] Ingesting and standardizing raw CSV datasets...")
    raw_data = load_all_raw_data(data_dir, default_obs_end=args.obs_end)

    # Validate Raw Data
    validator = DataValidator(data_dir)
    val_report = validator.run_full_validation(raw_data)

    # [5-6/8] Build Features & Run Leakage Guard
    features_df = combine_features(
        raw_data=raw_data,
        obs_end=None,  # Handled automatically via raw_data or config
        include_labels=True,
    )

    # [7/8] Save Parquet
    LOGGER.info(f"[7/8] Saving feature dataset to {args.out}...")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    features_df.to_parquet(args.out, index=False)

    LOGGER.info("[8/8] Complete! Feature engineering finished successfully.")
    LOGGER.info(f"Extracted {features_df.shape[1]} columns across {features_df.shape[0]} athletes.")


if __name__ == "__main__":
    main()
