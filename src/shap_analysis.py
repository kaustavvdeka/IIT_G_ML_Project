"""
SHAP and Interpretability Analysis Module for PLAYHACK.

Provides feature attribution and hypothesis validation on key trend indicators:
- training_load_change_7d_vs_30d
- sleep_change_7d_vs_30d
- hr_change_7d_vs_30d
"""

import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PLOTS_DIR
from src.utils import LOGGER

HAS_SHAP = False
try:
    import shap
    HAS_SHAP = True
except Exception as e:
    LOGGER.warning(f"SHAP package not available ({e}) — falling back to model feature importances.")


def run_shap_analysis(
    model: Any,
    X_sample: pd.DataFrame,
    feature_names: List[str],
    output_plot_path: str = os.path.join(PLOTS_DIR, "shap_summary.png"),
) -> Optional[np.ndarray]:
    """
    Computes SHAP values for the winning model and produces a summary plot.
    """
    if not HAS_SHAP:
        LOGGER.info("Skipping SHAP computation (SHAP not installed/compatible).")
        return None

    try:
        LOGGER.info("Computing SHAP values for tree explanation...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Handle binary classification output (list of arrays or 2D array)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_to_plot,
            X_sample,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.title("PLAYHACK Feature Attribution (SHAP Summary)", fontsize=14, pad=15)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
        plt.savefig(output_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        LOGGER.info(f"Saved SHAP summary plot to {output_plot_path}")
        return shap_values_to_plot

    except Exception as e:
        LOGGER.warning(f"SHAP analysis encountered an error ({e}) — proceeding without SHAP plot.")
        return None

def get_single_athlete_shap(
    model: Any,
    X_single: pd.DataFrame,
    feature_names: List[str],
) -> Optional[Dict[str, float]]:
    """
    Computes SHAP values for a single athlete and returns a dictionary mapping
    features to their SHAP contribution.
    """
    if not HAS_SHAP:
        return None

    try:
        # Extract base estimator if wrapped in CalibratedClassifierCV
        if hasattr(model, "calibrated_estimators_"):
            model = model.calibrated_estimators_[0].estimator
            
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_single)
        
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]
            
        shap_dict = dict(zip(feature_names, shap_vals))
        return shap_dict
        
    except Exception as e:
        LOGGER.warning(f"Error computing single athlete SHAP: {e}")
        return None
