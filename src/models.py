"""
Machine Learning Model Zoo for PLAYHACK Injury Prediction.

Supports:
- Tier 1: XGBoost, LightGBM, CatBoost (with graceful fallback if not installed)
- Tier 2: Random Forest, Extra Trees, HistGradientBoosting
- Tier 3: Logistic Regression, Ridge
Includes preprocessors (imputation, categorical encoding, scaling).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import LOGGER

# Safe optional imports
HAS_XGB = False
HAS_LGB = False
HAS_CAT = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    LOGGER.warning("XGBoost not installed — skipping XGBoost models.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    LOGGER.warning("LightGBM not installed — skipping LightGBM models.")

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    LOGGER.warning("CatBoost not installed — skipping CatBoost models.")


# ==============================================================================
# 1. PREPROCESSING WRAPPER
# ==============================================================================
class TabularPreprocessor:
    """
    Leakage-safe tabular preprocessor that handles missing values,
    numeric scaling, and one-hot encoding of categorical features.
    """

    def __init__(self, scale_numeric: bool = False):
        self.scale_numeric = scale_numeric
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler() if scale_numeric else None
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = [c for c in X.columns if c not in self.num_cols and c not in ["athlete_id", "obs_window_end"]]

        if self.num_cols:
            self.num_imputer.fit(X[self.num_cols])
            if self.scaler:
                imputed_num = self.num_imputer.transform(X[self.num_cols])
                self.scaler.fit(imputed_num)

        if self.cat_cols:
            cat_df = X[self.cat_cols].astype(str).fillna("missing")
            self.encoder.fit(cat_df)
            encoded_cat_names = self.encoder.get_feature_names_out(self.cat_cols).tolist()
        else:
            encoded_cat_names = []

        self.feature_names_out_ = self.num_cols + encoded_cat_names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        num_arr = np.empty((len(X), 0))
        if self.num_cols:
            if hasattr(self.num_imputer, "statistics_"):
                if not hasattr(self.num_imputer, "_fill_dtype"):
                    self.num_imputer._fill_dtype = self.num_imputer.statistics_.dtype
                if not hasattr(self.num_imputer, "_fit_dtype"):
                    self.num_imputer._fit_dtype = self.num_imputer.statistics_.dtype
            num_arr = self.num_imputer.transform(X[self.num_cols])
            if self.scaler:
                num_arr = self.scaler.transform(num_arr)

        cat_arr = np.empty((len(X), 0))
        if self.cat_cols:
            cat_df = X[self.cat_cols].astype(str).fillna("missing")
            cat_arr = self.encoder.transform(cat_df)

        if num_arr.shape[1] > 0 and cat_arr.shape[1] > 0:
            return np.hstack([num_arr, cat_arr])
        elif num_arr.shape[1] > 0:
            return num_arr
        else:
            return cat_arr

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


# ==============================================================================
# 2. MODEL FACTORY FOR CLASSIFICATION (TASK A)
# ==============================================================================
def get_classification_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of candidate classification models.
    Only includes models whose dependencies are available.
    """
    models = {}

    # Tier 1: Gradient Boosted Decision Trees
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
        )

    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
        )

    if HAS_CAT:
        models["CatBoost"] = cb.CatBoostClassifier(
            iterations=150,
            depth=4,
            learning_rate=0.05,
            random_seed=random_state,
            verbose=False,
        )

    # Tier 2: Bagging & Ensembles
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
    )

    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=random_state,
    )

    # Tier 3: Linear Baselines
    models["LogisticRegression"] = LogisticRegression(
        C=0.5,
        max_iter=500,
        random_state=random_state,
    )

    return models


# ==============================================================================
# 3. MODEL FACTORY FOR REGRESSION (TASKS B & C)
# ==============================================================================
def get_regression_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns candidate regressors for onset and recovery duration tasks.
    """
    models = {}

    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
        )

    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
            verbose=-1,
        )

    if HAS_CAT:
        models["CatBoost"] = cb.CatBoostRegressor(
            iterations=100,
            depth=3,
            learning_rate=0.05,
            random_seed=random_state,
            verbose=False,
        )

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1,
    )

    models["HistGradientBoosting"] = HistGradientBoostingRegressor(
        max_iter=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=random_state,
    )

    models["Ridge"] = Ridge(alpha=1.0, random_state=random_state)

    return models
