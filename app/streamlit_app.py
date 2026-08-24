"""
PLAYHACK Sports Injury Prediction Interactive Web Application.

Run via:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as pd_st
import streamlit as st

# Add workspace root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.components import (
    plot_athlete_time_series,
    plot_confusion_matrix_interactive,
    plot_feature_importance_chart,
    plot_model_comparison_bar,
    plot_threshold_curve,
)
from src.config import (
    BENCHMARK_RESULTS_PATH,
    DATA_DIR,
    FEATURE_IMPORTANCE_PATH,
    FEATURES_PARQUET_PATH,
    PREDICTIONS_PATH,
)
from src.data_loader import RawDataLoader
from src.prediction import Predictor
from src.utils import load_json

st.set_page_config(
    page_title="PLAYHACK — Injury Prediction Pipeline",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# DATA CACHING & LOADERS
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_features_dataset() -> pd.DataFrame:
    if os.path.exists(FEATURES_PARQUET_PATH):
        return pd.read_parquet(FEATURES_PARQUET_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_benchmark_data() -> dict:
    if os.path.exists(BENCHMARK_RESULTS_PATH):
        return load_json(BENCHMARK_RESULTS_PATH)
    return {}


@st.cache_data(show_spinner=False)
def load_feature_importance_data() -> pd.DataFrame:
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_oof_predictions_data() -> pd.DataFrame:
    if os.path.exists(PREDICTIONS_PATH):
        return pd.read_csv(PREDICTIONS_PATH)
    return pd.DataFrame()


# Load artifacts
features_df = load_features_dataset()
benchmark_meta = load_benchmark_data()
importance_df = load_feature_importance_data()
oof_df = load_oof_predictions_data()


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🏃‍♂️ PLAYHACK ML")
st.sidebar.markdown("**Sports Injury Prediction Pipeline**")
st.sidebar.markdown("---")

nav_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Dashboard",
        "📂 Dataset Explorer",
        "👤 Athlete View",
        "🔬 Feature Explorer",
        "🏆 Model Benchmark",
        "🎯 Classification Analysis",
        "🌲 Feature Importance & SHAP",
        "🔮 Prediction Tool",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "🛡️ **Leakage Guard Enabled**\n"
    "Features are strictly extracted within the 30-day observation window. "
    "Cross-validation uses 5-fold `GroupKFold` on athletes."
)


# ==============================================================================
# 1. DASHBOARD TAB
# ==============================================================================
if nav_page == "📊 Dashboard":
    st.title("🏃‍♂️ PLAYHACK Sports Injury Prediction System")
    st.markdown(
        "An end-to-end, leakage-safe machine learning system predicting **injury risk**, **onset day**, and **recovery duration**."
    )

    col1, col2, col3, col4 = st.columns(4)

    total_athletes = len(features_df) if not features_df.empty else 3000
    num_features = len(features_df.columns) if not features_df.empty else 0
    injured_count = int(features_df["injured_in_risk_window"].sum()) if "injured_in_risk_window" in features_df.columns else 0
    injury_rate = (injured_count / total_athletes * 100) if total_athletes > 0 else 0

    col1.metric("Total Athletes", f"{total_athletes:,}")
    col2.metric("Extracted Features", f"{num_features}")
    col3.metric("Injured Athletes", f"{injured_count:,}")
    col4.metric("Injury Rate", f"{injury_rate:.2f}%")

    st.markdown("---")

    col5, col6, col7 = st.columns(3)

    # Leaderboard KPI metrics
    cls_meta = benchmark_meta.get("classification", {})
    onset_meta = benchmark_meta.get("onset", {})
    rec_meta = benchmark_meta.get("recovery", {})

    best_cls_score = max([v["score"] for v in cls_meta.values()]) if cls_meta else 0.0
    best_onset_score = min([v["score"] for v in onset_meta.values()]) if onset_meta else 0.0
    best_rec_score = min([v["score"] for v in rec_meta.values()]) if rec_meta else 0.0

    col5.metric("Best Classification F1", f"{best_cls_score:.4f}", help="Task A Official Metric (OOF F1)")
    col6.metric("Best Onset MAE", f"{best_onset_score:.2f} days", help="Task B Official Metric (MAE on injured)")
    col7.metric("Best Recovery MAE", f"{best_rec_score:.2f} days", help="Task C Official Metric (MAE on injured)")

    st.markdown("### 📋 Executive Summary")
    st.markdown("""
    - **Temporal Integrity**: Structural boundaries guarantee zero knowledge from Day 31-60 is used for feature calculation.
    - **Grouping Integrity**: `GroupKFold` ensures athletes never appear in both training and evaluation folds.
    - **Conditional Modeling**: Onset day and recovery duration regressors are trained conditionally on verified injured athletes.
    - **Threshold Optimization**: Decision boundary tuned per model to maximize the F1-Score rather than arbitrary 0.5 default.
    """)


# ==============================================================================
# 2. DATASET EXPLORER TAB
# ==============================================================================
elif nav_page == "📂 Dataset Explorer":
    st.title("📂 Dataset Explorer")
    st.markdown("Explore the multi-source wearable and session records.")

    dataset_choice = st.selectbox(
        "Select Table to Inspect",
        [
            "Feature Dataset (features.parquet)",
            "Daily Activity",
            "Sleep Records",
            "Training Sessions",
            "Athlete Metadata",
            "Weight Logs",
            "Training Labels",
        ],
    )

    if dataset_choice == "Feature Dataset (features.parquet)":
        st.write(f"**Shape:** {features_df.shape[0]} rows × {features_df.shape[1]} columns")
        search_id = st.text_input("Filter by Athlete ID (optional):", "")
        if search_id:
            filtered_df = features_df[features_df["athlete_id"].astype(str) == str(search_id)]
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(features_df.head(100), use_container_width=True)

    else:
        # Load raw sample
        loader = RawDataLoader(DATA_DIR)
        table_map = {
            "Daily Activity": loader.load_daily_activity,
            "Sleep Records": loader.load_sleep_day,
            "Training Sessions": loader.load_training_sessions,
            "Athlete Metadata": loader.load_athlete_metadata,
            "Weight Logs": loader.load_weight_logs,
            "Training Labels": loader.load_train_labels,
        }
        df_raw = table_map[dataset_choice]()
        st.write(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
        st.dataframe(df_raw.head(100), use_container_width=True)


# ==============================================================================
# 3. ATHLETE VIEW TAB
# ==============================================================================
elif nav_page == "👤 Athlete View":
    st.title("👤 Individual Athlete Profile & Biometrics")

    if features_df.empty:
        st.warning("Feature dataset not available. Please run feature engineering first.")
    else:
        athlete_list = features_df["athlete_id"].tolist()
        selected_athlete = st.selectbox("Select Athlete ID:", athlete_list, index=0)

        athlete_row = features_df[features_df["athlete_id"] == selected_athlete].iloc[0]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Sport", str(athlete_row.get("sport", "N/A")))
        col2.metric("Position", str(athlete_row.get("position", "N/A")))
        col3.metric("Age", str(athlete_row.get("age", "N/A")))
        col4.metric("Height", f"{athlete_row.get('height_cm', 'N/A')} cm")
        col5.metric("Weight", f"{athlete_row.get('weight_kg_baseline', 'N/A')} kg")

        st.markdown("---")
        st.subheader("📈 7-day vs 30-day Trend Ratios")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Steps ACWR", f"{athlete_row.get('steps_change_7d_vs_30d', 1.0):.2f}x")
        m2.metric("Workload ACWR", f"{athlete_row.get('training_load_change_7d_vs_30d', 1.0):.2f}x")
        m3.metric("Sleep Ratio", f"{athlete_row.get('sleep_change_7d_vs_30d', 1.0):.2f}x")
        m4.metric("Heart Rate Ratio", f"{athlete_row.get('hr_change_7d_vs_30d', 1.0):.2f}x")

        # Load raw activity for time-series charts
        loader = RawDataLoader(DATA_DIR)
        raw_act = loader.load_daily_activity()
        ath_act = raw_act[raw_act["athlete_id"] == selected_athlete].sort_values("date")

        if not ath_act.empty:
            st.markdown("### 🏃‍♂️ Daily Steps & Calories Over Time")
            fig_steps = plot_athlete_time_series(ath_act, "TotalSteps", "Daily Step Count", "Steps")
            st.plotly_chart(fig_steps, use_container_width=True)


# ==============================================================================
# 4. FEATURE EXPLORER TAB
# ==============================================================================
elif nav_page == "🔬 Feature Explorer":
    st.title("🔬 Feature Distribution & Statistics")

    if features_df.empty:
        st.warning("Feature dataset not available.")
    else:
        feature_cols = [c for c in features_df.columns if c not in ["athlete_id", "obs_window_end"]]
        selected_feat = st.selectbox("Select Feature to Analyze:", feature_cols)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.histogram(
                features_df,
                x=selected_feat,
                color="injured_in_risk_window" if "injured_in_risk_window" in features_df.columns else None,
                barmode="overlay",
                title=f"Distribution of '{selected_feat}'",
                template="plotly_dark",
                opacity=0.75,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📊 Descriptive Statistics")
            series = features_df[selected_feat]
            stats_df = pd.DataFrame({
                "Metric": ["Data Type", "Count", "Missing %", "Mean", "Std Dev", "Min", "Median", "Max"],
                "Value": [
                    str(series.dtype),
                    str(len(series.dropna())),
                    f"{series.isnull().mean() * 100:.2f}%",
                    f"{series.mean():.3f}" if pd.api.types.is_numeric_dtype(series) else "N/A",
                    f"{series.std():.3f}" if pd.api.types.is_numeric_dtype(series) else "N/A",
                    f"{series.min():.3f}" if pd.api.types.is_numeric_dtype(series) else "N/A",
                    f"{series.median():.3f}" if pd.api.types.is_numeric_dtype(series) else "N/A",
                    f"{series.max():.3f}" if pd.api.types.is_numeric_dtype(series) else "N/A",
                ],
            })
            st.table(stats_df)


# ==============================================================================
# 5. MODEL BENCHMARK TAB
# ==============================================================================
elif nav_page == "🏆 Model Benchmark":
    st.title("🏆 Model Benchmark Leaderboard")

    leaderboard = benchmark_meta.get("leaderboard", [])
    if not leaderboard:
        st.info("No benchmark results found. Run `python train_and_evaluate.py` to train models.")
    else:
        lb_df = pd.DataFrame(leaderboard)
        st.dataframe(lb_df, use_container_width=True)

        st.markdown("### 📊 CV Performance Comparison")
        fig_bar = plot_model_comparison_bar(lb_df)
        st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# 6. CLASSIFICATION ANALYSIS TAB
# ==============================================================================
elif nav_page == "🎯 Classification Analysis":
    st.title("🎯 Classification Analysis & Threshold Tuning")

    cls_meta = benchmark_meta.get("classification", {})
    if not cls_meta:
        st.warning("Classification benchmark results not found.")
    else:
        model_names = list(cls_meta.keys())
        selected_model = st.selectbox("Select Classifier to Inspect:", model_names)
        m_info = cls_meta[selected_model]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("OOF F1-Score", f"{m_info.get('score', 0.0):.4f}")
        col2.metric("Optimal Threshold", f"{m_info.get('best_threshold', 0.5):.3f}")
        col3.metric("Precision", f"{m_info.get('precision', 0.0):.4f}")
        col4.metric("Recall", f"{m_info.get('recall', 0.0):.4f}")

        st.markdown("---")

        # Threshold curve
        curve_data = m_info.get("threshold_curve", {})
        if curve_data:
            fig_curve = plot_threshold_curve(curve_data)
            st.plotly_chart(fig_curve, use_container_width=True)

        # Confusion Matrix
        if not oof_df.empty and "y_true_injured" in oof_df.columns:
            prob_col = f"prob_{selected_model}"
            if prob_col in oof_df.columns:
                thresh = m_info.get("best_threshold", 0.5)
                y_pred = (oof_df[prob_col] >= thresh).astype(int)
                fig_cm = plot_confusion_matrix_interactive(oof_df["y_true_injured"].values, y_pred.values)
                st.plotly_chart(fig_cm, use_container_width=True)


# ==============================================================================
# 7. FEATURE IMPORTANCE & SHAP TAB
# ==============================================================================
elif nav_page == "🌲 Feature Importance & SHAP":
    st.title("🌲 Feature Importance & Model Explainability")

    if importance_df.empty:
        st.info("Feature importance data not found. Run `python train_and_evaluate.py`.")
    else:
        st.markdown("### Top Predictive Features Driving Injury Risk")
        fig_imp = plot_feature_importance_chart(importance_df, top_n=20)
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("### 🔬 Hypothesis Evaluation")
        st.markdown("""
        - **Workload Spike Hypothesis (`training_load_change_7d_vs_30d`)**: Checks whether acute training spikes relative to chronic baseline drive injuries.
        - **Sleep Deficit Hypothesis (`sleep_change_7d_vs_30d`)**: Evaluates whether acute reductions in sleep duration increase vulnerability.
        - **Heart Rate Strain (`hr_change_7d_vs_30d`)**: Analyzes resting/active heart rate elevations as fatigue indicators.
        """)


# ==============================================================================
# 8. PREDICTION TOOL TAB
# ==============================================================================
elif nav_page == "🔮 Prediction Tool":
    st.title("🔮 Live Athlete Risk Predictor")

    st.markdown(
        "> ⚠️ **Important Medical Disclaimer**: *This tool produces statistical machine-learning risk estimates "
        "from observational data. It does NOT provide a medical diagnosis or certainty.*"
    )

    if features_df.empty:
        st.warning("Features dataset not loaded.")
    else:
        athlete_list = features_df["athlete_id"].tolist()
        sel_id = st.selectbox("Select Athlete for Live Evaluation:", athlete_list)

        ath_data = features_df[features_df["athlete_id"] == sel_id]

        if st.button("🚀 Run Risk Evaluation"):
            predictor = Predictor()
            probs, binary_preds, thresh = predictor.predict_classification(ath_data)
            onset_pred = predictor.predict_regression(ath_data, "onset", clip_bounds=(1.0, 30.0))[0]
            rec_pred = predictor.predict_regression(ath_data, "recovery", clip_bounds=(0.0, None))[0]

            prob_val = float(probs[0])
            is_injured = int(binary_preds[0])

            st.markdown("---")
            st.subheader("📋 Evaluation Results")

            c1, c2, c3 = st.columns(3)
            risk_tier = "🔴 HIGH RISK" if prob_val >= thresh else "🟢 LOW RISK"
            c1.metric("Predicted Injury Risk", f"{prob_val * 100:.1f}%", delta=risk_tier)
            c2.metric("Expected Onset Day", f"Day {int(np.round(onset_pred))}" if is_injured else "N/A (Healthy)")
            c3.metric("Expected Recovery Duration", f"{int(np.round(rec_pred))} days" if is_injured else "N/A (Healthy)")
