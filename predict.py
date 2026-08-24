"""
CLI Entrypoint for Batch Inference and Submission File Generation.

Usage:
    python predict.py --features outputs/features.parquet --out outputs/submission.csv
    python predict.py --test_dir "data/Test data" --out outputs/submission.csv
"""

import argparse
import os
import sys
import pandas as pd

# Ensure root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DEFAULT_OBS_END, FEATURES_PARQUET_PATH, MODELS_DIR, SUBMISSION_PATH
from src.data_loader import load_all_raw_data
from src.feature_engineering import combine_features
from src.prediction import Predictor
from src.utils import LOGGER


def main():
    parser = argparse.ArgumentParser(description="PLAYHACK Prediction & Submission Generator")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to precomputed features parquet file.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/Test data",
        help="Directory containing unlabelled test CSV files.",
    )
    parser.add_argument(
        "--obs_end",
        type=str,
        default=DEFAULT_OBS_END,
        help=f"Observation window end date for test data (default: {DEFAULT_OBS_END}).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=MODELS_DIR,
        help=f"Path to saved models directory (default: {MODELS_DIR}).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=SUBMISSION_PATH,
        help=f"Output path for submission CSV (default: {SUBMISSION_PATH}).",
    )
    args = parser.parse_args()

    LOGGER.info("=" * 70)
    LOGGER.info("STARTING PLAYHACK INFERENCE PIPELINE")
    LOGGER.info(f"Model directory: {args.model_dir}")
    LOGGER.info(f"Output path: {args.out}")
    LOGGER.info("=" * 70)

    # 1. Obtain feature dataframe
    if args.features and os.path.exists(args.features):
        LOGGER.info(f"Loading precomputed features from {args.features}...")
        features_df = pd.read_parquet(args.features)
    elif os.path.exists(args.test_dir):
        LOGGER.info(f"Extracting features on-the-fly from test directory: {args.test_dir}...")
        raw_test = load_all_raw_data(args.test_dir, default_obs_end=args.obs_end)
        features_df = combine_features(raw_test, obs_end=args.obs_end, include_labels=False)
    elif os.path.exists(FEATURES_PARQUET_PATH):
        LOGGER.info(f"Falling back to default features at {FEATURES_PARQUET_PATH}...")
        features_df = pd.read_parquet(FEATURES_PARQUET_PATH)
    else:
        raise FileNotFoundError(
            f"Could not find input features or test directory at '{args.test_dir}'."
        )

    # 2. Run predictor
    predictor = Predictor(models_dir=args.model_dir)
    submission_df = predictor.generate_submission(features_df, output_path=args.out)

    LOGGER.info("\nInference pipeline completed successfully!")
    print(submission_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
