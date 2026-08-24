"""
Inference and Submission Generation Pipeline for PLAYHACK.

Loads trained fold models, generates leakage-safe predictions for test sets,
applies the optimal F1 threshold and physical bounds, and outputs submission.csv.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import BENCHMARK_RESULTS_PATH, MODELS_DIR, SUBMISSION_PATH
from src.utils import LOGGER, load_json


class Predictor:
    """
    Handles inference using trained fold models or ensembles.
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.benchmark_meta = self._load_benchmark_metadata()

    def _load_benchmark_metadata(self) -> Dict[str, Any]:
        if os.path.exists(BENCHMARK_RESULTS_PATH):
            return load_json(BENCHMARK_RESULTS_PATH)
        return {}

    def _load_fold_models(self, task: str, model_name: str) -> List[Tuple[Any, Any]]:
        path = os.path.join(self.models_dir, task, f"{model_name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict_classification(
        self,
        X_df: pd.DataFrame,
        model_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generates classification probabilities and thresholded binary predictions.
        """
        cls_meta = self.benchmark_meta.get("classification", {})

        if model_name is None:
            # Pick best model from benchmark
            if "Top3_Ensemble" in cls_meta and cls_meta["Top3_Ensemble"]["score"] >= max(
                [v["score"] for k, v in cls_meta.items() if k != "Top3_Ensemble"] + [0]
            ):
                model_name = "Top3_Ensemble"
            elif cls_meta:
                model_name = max(cls_meta.items(), key=lambda x: x[1]["score"])[0]
            else:
                model_name = "LightGBM"

        LOGGER.info(f"Using classification model: '{model_name}'")

        if model_name == "Top3_Ensemble":
            top3_names = cls_meta.get("Top3_Ensemble", {}).get("models_in_ensemble", ["LightGBM", "XGBoost", "RandomForest"])
            all_ens_probs = []
            for m_name in top3_names:
                m_probs = self._predict_single_classifier(X_df, m_name)
                all_ens_probs.append(m_probs)
            probs = np.mean(all_ens_probs, axis=0)
            threshold = cls_meta.get("Top3_Ensemble", {}).get("best_threshold", 0.5)
        else:
            probs = self._predict_single_classifier(X_df, model_name)
            threshold = cls_meta.get(model_name, {}).get("best_threshold", 0.5)

        binary_preds = (probs >= threshold).astype(int)
        return probs, binary_preds, threshold

    def _predict_single_classifier(self, X_df: pd.DataFrame, model_name: str) -> np.ndarray:
        fold_models = self._load_fold_models("classification", model_name)
        fold_probs = []

        exclude_cols = ["athlete_id", "obs_window_end", "injured_in_risk_window", "onset_day_offset", "recovery_duration"]
        feature_cols = [c for c in X_df.columns if c not in exclude_cols]
        X = X_df[feature_cols]

        for preprocessor, model in fold_models:
            X_proc = preprocessor.transform(X)
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_proc)[:, 1]
            else:
                d = model.decision_function(X_proc)
                p = 1.0 / (1.0 + np.exp(-d))
            fold_probs.append(p)

        return np.mean(fold_probs, axis=0)

    def predict_regression(
        self,
        X_df: pd.DataFrame,
        task: str,
        clip_bounds: Tuple[float, Optional[float]],
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generates regression predictions averaged over all 5 folds.
        """
        task_meta = self.benchmark_meta.get(task, {})
        if model_name is None:
            if task_meta:
                model_name = min(task_meta.items(), key=lambda x: x[1]["score"])[0]
            else:
                model_name = "LightGBM"

        LOGGER.info(f"Using {task} regression model: '{model_name}'")
        fold_models = self._load_fold_models(task, model_name)
        fold_preds = []

        exclude_cols = ["athlete_id", "obs_window_end", "injured_in_risk_window", "onset_day_offset", "recovery_duration"]
        feature_cols = [c for c in X_df.columns if c not in exclude_cols]
        X = X_df[feature_cols]

        min_val, max_val = clip_bounds

        for preprocessor, model in fold_models:
            X_proc = preprocessor.transform(X)
            preds = model.predict(X_proc)
            if max_val is not None:
                preds = np.clip(preds, min_val, max_val)
            else:
                preds = np.maximum(preds, min_val)
            fold_preds.append(preds)

        mean_preds = np.mean(fold_preds, axis=0)
        if max_val is not None:
            mean_preds = np.clip(mean_preds, min_val, max_val)
        else:
            mean_preds = np.maximum(mean_preds, min_val)

        return mean_preds

    def generate_submission(
        self,
        features_df: pd.DataFrame,
        output_path: str = SUBMISSION_PATH,
    ) -> pd.DataFrame:
        """
        Builds submission dataframe matching the PLAYHACK format.
        """
        LOGGER.info(f"Generating full submission predictions for {len(features_df)} athletes...")

        athlete_ids = features_df["athlete_id"].values

        # 1. Predict Injury
        probs, binary_preds, thresh = self.predict_classification(features_df)

        # 2. Predict Onset (1 - 30)
        onset_preds = self.predict_regression(features_df, "onset", clip_bounds=(1.0, 30.0))
        onset_preds_int = np.round(onset_preds).astype(int)

        # 3. Predict Recovery (>= 0)
        rec_preds = self.predict_regression(features_df, "recovery", clip_bounds=(0.0, None))
        rec_preds_int = np.round(rec_preds).astype(int)

        submission_df = pd.DataFrame({
            "athlete_id": athlete_ids,
            "injured_in_risk_window": binary_preds,
            "onset_day_offset": onset_preds_int,
            "recovery_duration": rec_preds_int,
        })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission_df.to_csv(output_path, index=False)
        LOGGER.info(f"Submission saved to {output_path} (Shape: {submission_df.shape})")

        LOGGER.info(f"Predicted Injury Rate: {(binary_preds == 1).mean() * 100:.2f}% ({binary_preds.sum()}/{len(binary_preds)})")
        return submission_df
