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

import io
import sys
from src.config import BENCHMARK_RESULTS_PATH, MODELS_DIR, SUBMISSION_PATH
from src.models import TabularPreprocessor
from src.utils import LOGGER, load_json

# Alias into __main__ for legacy unpickling compatibility across Streamlit and CLI
if "__main__" in sys.modules:
    setattr(sys.modules["__main__"], "TabularPreprocessor", TabularPreprocessor)


class _RobustUnpickler(pickle.Unpickler):
    """Custom unpickler that safely resolves TabularPreprocessor regardless of originating module namespace."""
    def find_class(self, module, name):
        if name == "TabularPreprocessor":
            from src.models import TabularPreprocessor
            return TabularPreprocessor
        return super().find_class(module, name)


def _safe_transform(preprocessor: Any, X: pd.DataFrame) -> np.ndarray:
    """Defensively transforms features, repairing any unpickled scikit-learn attribute differences."""
    if hasattr(preprocessor, "num_imputer"):
        imputer = preprocessor.num_imputer
        if hasattr(imputer, "statistics_"):
            if not hasattr(imputer, "_fill_dtype"):
                imputer._fill_dtype = imputer.statistics_.dtype
            if not hasattr(imputer, "_fit_dtype"):
                imputer._fit_dtype = imputer.statistics_.dtype
    return preprocessor.transform(X)


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

    def get_available_models(self, task: str) -> List[str]:
        """Returns the list of available model names stored on disk for a given task."""
        task_dir = os.path.join(self.models_dir, task)
        if not os.path.exists(task_dir):
            return []
        return [
            os.path.splitext(f)[0]
            for f in os.listdir(task_dir)
            if f.endswith(".pkl")
        ]

    def _load_fold_models(self, task: str, model_name: str) -> List[Tuple[Any, Any]]:
        path = os.path.join(self.models_dir, task, f"{model_name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        with open(path, "rb") as f:
            return _RobustUnpickler(f).load()

    def predict_classification(
        self,
        X_df: pd.DataFrame,
        model_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generates classification probabilities and thresholded binary predictions.
        """
        cls_meta = self.benchmark_meta.get("classification", {})
        avail_models = self.get_available_models("classification")

        if not avail_models:
            raise FileNotFoundError(f"No trained classification models found in {os.path.join(self.models_dir, 'classification')}")

        if model_name is None:
            # Check if Top3_Ensemble is applicable
            top3_ensemble_meta = cls_meta.get("Top3_Ensemble")
            if top3_ensemble_meta:
                ensemble_models = top3_ensemble_meta.get("models_in_ensemble", [])
                # Check if at least 2 models in the ensemble are available on disk
                avail_in_ensemble = [m for m in ensemble_models if m in avail_models]
                if len(avail_in_ensemble) >= 2 and top3_ensemble_meta.get("score", 0) >= max(
                    [v.get("score", 0) for k, v in cls_meta.items() if k != "Top3_Ensemble" and k in avail_models] + [0]
                ):
                    model_name = "Top3_Ensemble"

            if model_name is None:
                # Pick the highest scoring available single classifier
                scored_available = {k: v for k, v in cls_meta.items() if k in avail_models and k != "Top3_Ensemble"}
                if scored_available:
                    model_name = max(scored_available.items(), key=lambda x: x[1].get("score", 0.0))[0]
                else:
                    model_name = "LightGBM" if "LightGBM" in avail_models else avail_models[0]

        LOGGER.info(f"Using classification model: '{model_name}'")

        if model_name == "Top3_Ensemble":
            top3_names = cls_meta.get("Top3_Ensemble", {}).get("models_in_ensemble", ["LightGBM", "XGBoost", "RandomForest"])
            # Filter to models available on disk
            active_names = [m for m in top3_names if m in avail_models]
            if not active_names:
                active_names = avail_models[:min(3, len(avail_models))]

            all_ens_probs = []
            for m_name in active_names:
                m_probs = self._predict_single_classifier(X_df, m_name)
                all_ens_probs.append(m_probs)
            probs = np.mean(all_ens_probs, axis=0)
            threshold = cls_meta.get("Top3_Ensemble", {}).get("best_threshold", 0.5)
        else:
            if model_name not in avail_models:
                LOGGER.warning(f"Classification model '{model_name}' not found on disk. Falling back to '{avail_models[0]}'.")
                model_name = avail_models[0]
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
            X_proc = _safe_transform(preprocessor, X)
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
        avail_models = self.get_available_models(task)

        if not avail_models:
            raise FileNotFoundError(f"No trained {task} regression models found in {os.path.join(self.models_dir, task)}")

        if model_name is None:
            scored_available = {k: v for k, v in task_meta.items() if k in avail_models}
            if scored_available:
                model_name = min(scored_available.items(), key=lambda x: x[1].get("score", 999.0))[0]
            else:
                model_name = "LightGBM" if "LightGBM" in avail_models else avail_models[0]
        elif model_name not in avail_models:
            LOGGER.warning(f"{task} regression model '{model_name}' not found on disk. Falling back to '{avail_models[0]}'.")
            model_name = avail_models[0]

        LOGGER.info(f"Using {task} regression model: '{model_name}'")
        fold_models = self._load_fold_models(task, model_name)
        fold_preds = []

        exclude_cols = ["athlete_id", "obs_window_end", "injured_in_risk_window", "onset_day_offset", "recovery_duration"]
        feature_cols = [c for c in X_df.columns if c not in exclude_cols]
        X = X_df[feature_cols]

        min_val, max_val = clip_bounds

        for preprocessor, model in fold_models:
            X_proc = _safe_transform(preprocessor, X)
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
