"""
End-to-End Training and Benchmark Orchestration Module.

Trains classification (Task A), onset regression (Task B), and recovery regression (Task C)
under leakage-safe GroupKFold, tunes F1 thresholds, constructs a top-3 ensemble,
and produces the benchmark leaderboard and saved models.
"""

import copy
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import BENCHMARK_RESULTS_PATH, FEATURE_IMPORTANCE_PATH, MODELS_DIR, PREDICTIONS_PATH
from src.evaluation import best_f1_threshold, calculate_classification_metrics, calculate_regression_metrics, get_group_kfold_splits
from src.models import TabularPreprocessor, get_classification_models, get_regression_models
from src.utils import LOGGER, save_json, set_seed


class Trainer:
    """
    Orchestrates cross-validation, model training, evaluation, and ensembling.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.df = features_df.copy()
        self.n_splits = n_splits
        self.random_state = random_state
        set_seed(random_state)

        # Feature columns (exclude IDs, dates, targets)
        exclude_cols = [
            "athlete_id",
            "obs_window_end",
            "injured_in_risk_window",
            "onset_day_offset",
            "recovery_duration",
        ]
        self.feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        LOGGER.info(f"Trainer initialized with {len(self.feature_cols)} feature columns.")

    # ==========================================================================
    # TASK A: INJURY CLASSIFICATION
    # ==========================================================================
    def train_classification_benchmarks(self) -> Dict[str, Any]:
        """
        Runs GroupKFold cross-validation for all candidate classifiers on Task A.
        """
        LOGGER.info("=" * 70)
        LOGGER.info("STARTING TASK A: INJURY CLASSIFICATION BENCHMARK")
        LOGGER.info("=" * 70)

        target_col = "injured_in_risk_window"
        if target_col not in self.df.columns:
            raise KeyError(f"Target '{target_col}' not found in dataset.")

        y = self.df[target_col].values
        splits = get_group_kfold_splits(self.df, group_col="athlete_id", n_splits=self.n_splits)

        candidate_models = get_classification_models(self.random_state)
        results = {}
        oof_predictions: Dict[str, np.ndarray] = {}
        trained_models: Dict[str, Any] = {}

        for model_name, model_template in candidate_models.items():
            LOGGER.info(f"\n--- Evaluating Classifier: {model_name} ---")
            oof_probs = np.zeros(len(self.df))
            fold_models = []

            for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
                X_train = self.df.iloc[train_idx][self.feature_cols]
                y_train = y[train_idx]
                X_val = self.df.iloc[val_idx][self.feature_cols]
                y_val = y[val_idx]

                # Preprocess features
                scale = model_name in ["LogisticRegression"]
                preprocessor = TabularPreprocessor(scale_numeric=scale)
                X_train_proc = preprocessor.fit_transform(X_train)
                X_val_proc = preprocessor.transform(X_val)

                # Clone model template
                model = copy.deepcopy(model_template)
                model.fit(X_train_proc, y_train)

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_val_proc)[:, 1]
                else:
                    decision = model.decision_function(X_val_proc)
                    probs = 1.0 / (1.0 + np.exp(-decision))

                oof_probs[val_idx] = probs
                fold_models.append((preprocessor, model))

            # F1 threshold optimization on OOF predictions
            threshold_tuning = best_f1_threshold(y, oof_probs)
            best_thresh = threshold_tuning["best_threshold"]
            best_f1 = threshold_tuning["best_f1"]
            prec = threshold_tuning["precision"]
            rec = threshold_tuning["recall"]

            oof_preds = (oof_probs >= best_thresh).astype(int)
            overall_metrics = calculate_classification_metrics(y, oof_preds, oof_probs)

            LOGGER.info(
                f"Model: {model_name:<18} | Best Threshold: {best_thresh:.3f} | "
                f"OOF F1: {best_f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}"
            )

            results[model_name] = {
                "task": "classification",
                "target": target_col,
                "metric": "F1",
                "score": best_f1,
                "best_threshold": best_thresh,
                "precision": prec,
                "recall": rec,
                "accuracy": overall_metrics["accuracy"],
                "roc_auc": overall_metrics.get("roc_auc", 0.0),
                "threshold_curve": threshold_tuning,
            }
            oof_predictions[model_name] = oof_probs
            trained_models[model_name] = fold_models

        # ======================================================================
        # TOP-3 CLASSIFIER ENSEMBLE
        # ======================================================================
        LOGGER.info("\n" + "=" * 50)
        LOGGER.info("BUILDING TOP-3 CLASSIFICATION ENSEMBLE")
        LOGGER.info("=" * 50)

        sorted_models = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
        top3_names = [m[0] for m in sorted_models[:3]]
        LOGGER.info(f"Top 3 Classifiers selected for ensemble: {top3_names}")

        if len(top3_names) >= 2:
            ensemble_probs = np.mean([oof_predictions[m] for m in top3_names], axis=0)
            ens_tuning = best_f1_threshold(y, ensemble_probs)
            ens_thresh = ens_tuning["best_threshold"]
            ens_f1 = ens_tuning["best_f1"]
            ens_prec = ens_tuning["precision"]
            ens_rec = ens_tuning["recall"]

            best_single_f1 = sorted_models[0][1]["score"]
            best_single_name = sorted_models[0][0]

            LOGGER.info(
                f"Ensemble (Top-3) | Best Threshold: {ens_thresh:.3f} | "
                f"OOF F1: {ens_f1:.4f} | Precision: {ens_prec:.4f} | Recall: {ens_rec:.4f}"
            )

            if ens_f1 > best_single_f1:
                LOGGER.info(f"Ensemble improved over best single model ({best_single_name}: {best_single_f1:.4f}) by +{ens_f1 - best_single_f1:.4f} F1.")
            else:
                LOGGER.info(f"Ensemble ({ens_f1:.4f}) did not improve over the best single model ({best_single_name}: {best_single_f1:.4f}). Use the best individual classifier.")

            results["Top3_Ensemble"] = {
                "task": "classification",
                "target": target_col,
                "metric": "F1",
                "score": ens_f1,
                "best_threshold": ens_thresh,
                "precision": ens_prec,
                "recall": ens_rec,
                "models_in_ensemble": top3_names,
                "threshold_curve": ens_tuning,
            }
            oof_predictions["Top3_Ensemble"] = ensemble_probs

        return {
            "results": results,
            "oof_predictions": oof_predictions,
            "trained_models": trained_models,
        }

    # ==========================================================================
    # TASKS B & C: REGRESSION (ONSET & RECOVERY)
    # ==========================================================================
    def train_regression_benchmarks(
        self,
        task_name: str,
        target_col: str,
        clip_bounds: Tuple[float, Optional[float]],
    ) -> Dict[str, Any]:
        """
        Trains regressors strictly on athletes where injured_in_risk_window == 1.
        """
        LOGGER.info("=" * 70)
        LOGGER.info(f"STARTING {task_name.upper()} BENCHMARK (Target: {target_col})")
        LOGGER.info("=" * 70)

        # Filter strictly injured athletes
        injured_df = self.df[self.df["injured_in_risk_window"] == 1].copy().reset_index(drop=True)
        LOGGER.info(f"Training strictly on {len(injured_df)} injured athletes.")

        if target_col not in injured_df.columns or injured_df[target_col].dropna().empty:
            LOGGER.warning(f"Target '{target_col}' has no valid values. Skipping {task_name}.")
            return {"results": {}, "oof_predictions": {}, "trained_models": {}}

        # Drop any NaN target rows in conditional subset
        valid_mask = injured_df[target_col].notnull()
        injured_df = injured_df[valid_mask].reset_index(drop=True)
        y = injured_df[target_col].values

        splits = get_group_kfold_splits(injured_df, group_col="athlete_id", n_splits=self.n_splits)
        candidate_models = get_regression_models(self.random_state)

        results = {}
        oof_predictions: Dict[str, np.ndarray] = {}
        trained_models: Dict[str, Any] = {}

        min_val, max_val = clip_bounds

        for model_name, model_template in candidate_models.items():
            LOGGER.info(f"\n--- Evaluating Regressor: {model_name} ---")
            oof_preds = np.zeros(len(injured_df))
            fold_models = []

            for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
                X_train = injured_df.iloc[train_idx][self.feature_cols]
                y_train = y[train_idx]
                X_val = injured_df.iloc[val_idx][self.feature_cols]
                y_val = y[val_idx]

                scale = model_name in ["Ridge"]
                preprocessor = TabularPreprocessor(scale_numeric=scale)
                X_train_proc = preprocessor.fit_transform(X_train)
                X_val_proc = preprocessor.transform(X_val)

                model = copy.deepcopy(model_template)
                model.fit(X_train_proc, y_train)

                preds = model.predict(X_val_proc)
                # Apply bounds constraints
                if max_val is not None:
                    preds = np.clip(preds, min_val, max_val)
                else:
                    preds = np.maximum(preds, min_val)

                oof_preds[val_idx] = preds
                fold_models.append((preprocessor, model))

            metrics = calculate_regression_metrics(y, oof_preds)
            LOGGER.info(
                f"Model: {model_name:<18} | OOF MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f} | R2: {metrics['r2']:.3f}"
            )

            results[model_name] = {
                "task": task_name,
                "target": target_col,
                "metric": "MAE",
                "score": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
            oof_predictions[model_name] = oof_preds
            trained_models[model_name] = fold_models

        return {
            "results": results,
            "oof_predictions": oof_predictions,
            "trained_models": trained_models,
            "injured_df": injured_df,
        }

    # ==========================================================================
    # FEATURE IMPORTANCE EXTRACTION
    # ==========================================================================
    def extract_feature_importance(
        self,
        trained_models: Dict[str, Any],
        best_model_name: str,
    ) -> pd.DataFrame:
        """
        Extracts and averages feature importances across folds for tree models.
        """
        if best_model_name not in trained_models:
            return pd.DataFrame()

        fold_models = trained_models[best_model_name]
        all_importances = []
        feature_names = []

        for preprocessor, model in fold_models:
            if hasattr(model, "feature_importances_"):
                all_importances.append(model.feature_importances_)
                feature_names = preprocessor.feature_names_out_
            elif hasattr(model, "coef_"):
                all_importances.append(np.abs(model.coef_.flatten()))
                feature_names = preprocessor.feature_names_out_

        if not all_importances:
            return pd.DataFrame()

        mean_imp = np.mean(all_importances, axis=0)
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_imp,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return imp_df

    # ==========================================================================
    # FULL PIPELINE EXECUTION
    # ==========================================================================
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Executes Tasks A, B, and C, formats leaderboard, and saves models and predictions.
        """
        # Task A: Classification
        cls_out = self.train_classification_benchmarks()

        # Task B: Onset (Day 1-30)
        onset_out = self.train_regression_benchmarks(
            task_name="onset",
            target_col="onset_day_offset",
            clip_bounds=(1.0, 30.0),
        )

        # Task C: Recovery (>= 0)
        rec_out = self.train_regression_benchmarks(
            task_name="recovery",
            target_col="recovery_duration",
            clip_bounds=(0.0, None),
        )

        # Combine results into Leaderboard
        leaderboard = []
        for mod, r in cls_out["results"].items():
            leaderboard.append({
                "model": mod,
                "task": "classification",
                "target": "injured_in_risk_window",
                "metric": "F1",
                "score": r["score"],
                "best_threshold": r.get("best_threshold", 0.5),
                "precision": r.get("precision", 0.0),
                "recall": r.get("recall", 0.0),
            })

        for mod, r in onset_out["results"].items():
            leaderboard.append({
                "model": mod,
                "task": "onset",
                "target": "onset_day_offset",
                "metric": "MAE",
                "score": r["score"],
                "rmse": r.get("rmse", 0.0),
            })

        for mod, r in rec_out["results"].items():
            leaderboard.append({
                "model": mod,
                "task": "recovery",
                "target": "recovery_duration",
                "metric": "MAE",
                "score": r["score"],
                "rmse": r.get("rmse", 0.0),
            })

        leaderboard_df = pd.DataFrame(leaderboard)
        LOGGER.info("\n" + "=" * 70)
        LOGGER.info("FINAL PLAYHACK BENCHMARK LEADERBOARD")
        LOGGER.info("=" * 70)
        print(leaderboard_df.to_string(index=False))

        # Save Benchmark Results JSON
        benchmark_payload = {
            "classification": cls_out["results"],
            "onset": onset_out["results"],
            "recovery": rec_out["results"],
            "leaderboard": leaderboard,
        }
        save_json(benchmark_payload, BENCHMARK_RESULTS_PATH)
        LOGGER.info(f"Saved benchmark results to {BENCHMARK_RESULTS_PATH}")

        # Extract & Save Feature Importance for winning single classifier
        cls_singles = {k: v for k, v in cls_out["results"].items() if k != "Top3_Ensemble"}
        best_cls_name = max(cls_singles.items(), key=lambda x: x[1]["score"])[0]
        imp_df = self.extract_feature_importance(cls_out["trained_models"], best_cls_name)
        if not imp_df.empty:
            imp_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
            LOGGER.info(f"Saved feature importance for '{best_cls_name}' to {FEATURE_IMPORTANCE_PATH}")

        # Save OOF Predictions
        oof_df = pd.DataFrame({"athlete_id": self.df["athlete_id"]})
        if "injured_in_risk_window" in self.df.columns:
            oof_df["y_true_injured"] = self.df["injured_in_risk_window"]
        for mod, probs in cls_out["oof_predictions"].items():
            oof_df[f"prob_{mod}"] = probs
        oof_df.to_csv(PREDICTIONS_PATH, index=False)
        LOGGER.info(f"Saved OOF predictions to {PREDICTIONS_PATH}")

        # Save Model Artifacts
        self._save_models(cls_out["trained_models"], "classification")
        self._save_models(onset_out["trained_models"], "onset")
        self._save_models(rec_out["trained_models"], "recovery")

        return benchmark_payload

    def _save_models(self, models_dict: Dict[str, Any], task_subfolder: str) -> None:
        """Serializes trained model artifacts using pickle."""
        save_dir = os.path.join(MODELS_DIR, task_subfolder)
        os.makedirs(save_dir, exist_ok=True)
        for model_name, fold_models in models_dict.items():
            path = os.path.join(save_dir, f"{model_name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(fold_models, f)
        LOGGER.info(f"Saved {len(models_dict)} trained models to {save_dir}/")
