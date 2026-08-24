"""
Evaluation Metrics, Threshold Optimization, and GroupKFold Validation.

Enforces zero athlete leakage across folds and optimizes the classification threshold
for maximum F1 score on out-of-fold predictions.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.utils import LOGGER


def best_f1_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Sweeps decision thresholds from 0.05 to 0.95 to maximize F1 score.

    Args:
        y_true: Ground truth binary target (0 or 1).
        probabilities: Predicted probabilities for class 1.
        threshold_grid: Optional custom array of candidate thresholds.

    Returns:
        Dictionary containing optimal threshold, best F1, precision, recall, and curves.
    """
    if threshold_grid is None:
        threshold_grid = np.linspace(0.05, 0.95, 91)

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    best_thresh = 0.5
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0

    thresholds_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    for t in threshold_grid:
        preds = (probabilities >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)

        thresholds_list.append(float(t))
        f1_list.append(float(f1))
        precision_list.append(float(prec))
        recall_list.append(float(rec))

        if f1 > best_f1:
            best_f1 = float(f1)
            best_thresh = float(t)
            best_prec = float(prec)
            best_rec = float(rec)

    return {
        "best_threshold": round(best_thresh, 4),
        "best_f1": round(best_f1, 4),
        "precision": round(best_prec, 4),
        "recall": round(best_rec, 4),
        "thresholds": thresholds_list,
        "f1_scores": f1_list,
        "precisions": precision_list,
        "recalls": recall_list,
    }


def get_group_kfold_splits(
    df: pd.DataFrame,
    group_col: str = "athlete_id",
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates GroupKFold train/val index splits and validates that no athlete
    ever appears in both training and validation sets.
    """
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not found in DataFrame.")

    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].values
    splits = []

    LOGGER.info(f"Setting up GroupKFold with {n_splits} splits grouped by '{group_col}'...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups), 1):
        train_athletes = set(df.iloc[train_idx][group_col].unique())
        val_athletes = set(df.iloc[val_idx][group_col].unique())

        overlap = train_athletes.intersection(val_athletes)
        if overlap:
            error_msg = (
                f"FATAL LEAKAGE in Fold {fold}: Found {len(overlap)} overlapping athletes "
                f"between train and validation sets!\nOverlap sample: {list(overlap)[:5]}"
            )
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

        LOGGER.info(
            f"Fold {fold}: Train athletes={len(train_athletes)} ({len(train_idx)} rows) | "
            f"Val athletes={len(val_athletes)} ({len(val_idx)} rows) | Overlap={len(overlap)}"
        )
        splits.append((train_idx, val_idx))

    return splits


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculates F1, Precision, Recall, Accuracy, and ROC-AUC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = 0.0

    return metrics


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculates MAE, RMSE, and R2 for regression tasks.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }
