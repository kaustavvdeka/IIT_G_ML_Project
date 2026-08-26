"""
Run the Complete End-to-End Pipeline in One Script.
"""

import os
import sys
import time
import zipfile
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all_raw_data
from src.feature_engineering import combine_features
from src.prediction import Predictor
from src.training import Trainer
from src.utils import LOGGER


def extract_test_data(data_dir: str) -> str:
    """Extracts test data zip if present and returns the path to the extracted folder."""
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
    test_dir = os.path.join(data_dir, "Test data")
    
    if zip_files:
        zip_path = zip_files[0]
        LOGGER.info(f"Found test data archive: {zip_path}. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        # The zip might extract to a nested folder, or right into `Test data`.
        # We will assume it creates a folder structure that includes the CSVs.
        # Let's search for the actual folder containing `athlete_metadata.csv` or similar inside the extracted paths
        for root, dirs, files in os.walk(data_dir):
            if "athlete_metadata.csv" in files and root != data_dir:
                return root
                
    return test_dir


def run_pipeline():
    t_start = time.time()
    LOGGER.info("=" * 70)
    LOGGER.info("RUNNING COMPLETE PLAYHACK PIPELINE")
    LOGGER.info("=" * 70)

    # 1. Feature Engineering
    data_dir = "data"
    obs_end = "2026-02-03"
    features_path = "outputs/features.parquet"

    LOGGER.info("[1/3] Ingesting raw CSVs and building features...")
    raw_data = load_all_raw_data(data_dir, default_obs_end=obs_end)
    features_df = combine_features(raw_data, obs_end=obs_end, include_labels=True)

    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_parquet(features_path, index=False)
    LOGGER.info(f"Saved feature table to {features_path} ({features_df.shape[0]} rows, {features_df.shape[1]} cols)")

    # 2. Model Training & Cross-Validation
    LOGGER.info("\n[2/3] Training and evaluating ML models across 5 folds...")
    trainer = Trainer(features_df, n_splits=5, random_state=42)
    benchmark_results = trainer.run_full_pipeline()

    # 3. Test Prediction & Submission
    LOGGER.info("\n[3/3] Generating submission.csv...")
    predictor = Predictor()
    
    test_dir = extract_test_data(data_dir)
    
    if os.path.exists(test_dir) and any(f.endswith('.csv') for f in os.listdir(test_dir)):
        LOGGER.info(f"Processing test data from {test_dir}")
        raw_test = load_all_raw_data(test_dir, default_obs_end=obs_end)
        test_features_df = combine_features(raw_test, obs_end=obs_end, include_labels=False)
        sub_df = predictor.generate_submission(test_features_df, output_path="outputs/submission.csv")
    else:
        LOGGER.warning("Test data not found or invalid. Generating submission on validation features instead.")
        sub_df = predictor.generate_submission(features_df, output_path="outputs/submission.csv")

    LOGGER.info(f"\nALL DONE in {time.time() - t_start:.2f} seconds!")


if __name__ == "__main__":
    run_pipeline()
