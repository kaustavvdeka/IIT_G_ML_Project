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
