"""
CLI Entrypoint for Model Training, Cross-Validation, and Leaderboard Evaluation.

Usage:
    python train_and_evaluate.py --features outputs/features.parquet --folds 5
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Ensure root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import FEATURE_IMPORTANCE_PATH, FEATURES_PARQUET_PATH, PLOTS_DIR
from src.training import Trainer
from src.utils import LOGGER


def plot_top_features(importance_path: str, output_path: str = os.path.join(PLOTS_DIR, "feature_importance.png")) -> None:
    """Generates a static bar chart for the top 20 features."""
    if not os.path.exists(importance_path):
        return

    df = pd.read_csv(importance_path).head(20).iloc[::-1]  # Invert for horizontal bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(df["feature"], df["importance"], color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.xlabel("Importance Score", fontsize=12)
    plt.title("Top 20 Predictive Features (Injury Risk)", fontsize=14, pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    LOGGER.info(f"Saved feature importance plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PLAYHACK Leakage-Safe Model Training & Benchmarking")
    parser.add_argument(
        "--features",
        type=str,
        default=FEATURES_PARQUET_PATH,
        help=f"Path to features parquet file (default: {FEATURES_PARQUET_PATH}).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of GroupKFold cross-validation splits (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    LOGGER.info("=" * 70)
    LOGGER.info("STARTING PLAYHACK TRAINING & EVALUATION PIPELINE")
    LOGGER.info(f"Feature dataset: {args.features}")
    LOGGER.info(f"GroupKFold splits: {args.folds}")
    LOGGER.info(f"Random seed: {args.seed}")
    LOGGER.info("=" * 70)

    if not os.path.exists(args.features):
        raise FileNotFoundError(
            f"Feature file '{args.features}' not found.\n"
            f"Please run 'python feature_engineering.py' first."
        )

    LOGGER.info(f"Loading features from {args.features}...")
    features_df = pd.read_parquet(args.features)
    LOGGER.info(f"Loaded {features_df.shape[0]} athletes with {features_df.shape[1]} columns.")

    # Initialize Trainer and execute pipeline
    trainer = Trainer(features_df, n_splits=args.folds, random_state=args.seed)
    benchmark_results = trainer.run_full_pipeline()

    # Generate static plot
    plot_top_features(FEATURE_IMPORTANCE_PATH)

    LOGGER.info("\nTraining & evaluation completed successfully!")


if __name__ == "__main__":
    main()
