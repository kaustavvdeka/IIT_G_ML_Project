"""
Visual Component and Chart Helper Functions for Streamlit App.

Builds clean, interactive Plotly and Altair charts.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_metric_card(title: str, value: Any, delta: Optional[str] = None) -> str:
    """Helper to format KPI HTML cards."""
    return f"""
    <div style="background-color: #1e2130; padding: 16px; border-radius: 8px; border: 1px solid #2e344e; margin-bottom: 12px;">
        <p style="color: #90a4ae; font-size: 13px; margin: 0; text-transform: uppercase;">{title}</p>
        <h2 style="color: #00e676; margin: 4px 0 0 0; font-size: 26px;">{value}</h2>
        {f'<p style="color: #81c784; font-size: 12px; margin: 0;">{delta}</p>' if delta else ''}
    </div>
    """


def plot_feature_importance_chart(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart for top predictive features."""
    top_df = df.head(top_n).sort_values(by="importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=top_df["importance"],
            y=top_df["feature"],
            orientation="h",
            marker=dict(
                color=top_df["importance"],
                colorscale="Viridis",
                line=dict(color="#000000", width=1),
            ),
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Predictive Features",
        xaxis_title="Feature Importance",
        yaxis_title="Feature Name",
        height=550,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
    )
    return fig


def plot_threshold_curve(tuning_dict: Dict[str, Any]) -> go.Figure:
    """Plots F1, Precision, and Recall across thresholds."""
    thresholds = tuning_dict.get("thresholds", [])
    f1s = tuning_dict.get("f1_scores", [])
    precs = tuning_dict.get("precisions", [])
    recs = tuning_dict.get("recalls", [])
    best_t = tuning_dict.get("best_threshold", 0.5)
    best_f1 = tuning_dict.get("best_f1", 0.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=f1s, mode="lines", name="F1 Score", line=dict(color="#00e676", width=3)))
    fig.add_trace(go.Scatter(x=thresholds, y=precs, mode="lines", name="Precision", line=dict(color="#29b6f6", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=thresholds, y=recs, mode="lines", name="Recall", line=dict(color="#ffa726", width=2, dash="dot")))

    # Best threshold vertical line
    fig.add_vline(x=best_t, line_width=2, line_dash="dash", line_color="#ff5252", annotation_text=f"Best T={best_t} (F1={best_f1})")

    fig.update_layout(
        title="F1 Threshold Tuning Curve (OOF Predictions)",
        xaxis_title="Classification Threshold",
        yaxis_title="Score",
        height=400,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_confusion_matrix_interactive(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Plots an interactive heatmap confusion matrix."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    labels = ["Healthy (0)", "Injured (1)"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18},
        )
    )
    fig.update_layout(
        title="Out-of-Fold Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=380,
        template="plotly_dark",
    )
    return fig


def plot_model_comparison_bar(leaderboard_df: pd.DataFrame) -> go.Figure:
    """Plots comparison of models by CV metric."""
    fig = px.bar(
        leaderboard_df,
        x="model",
        y="score",
        color="task",
        barmode="group",
        title="Model Benchmark Comparison",
        text_auto=".3f",
        template="plotly_dark",
    )
    fig.update_layout(height=420)
    return fig


def plot_athlete_time_series(df_series: pd.DataFrame, value_col: str, title: str, y_label: str) -> go.Figure:
    """Plots daily or hourly trends for an individual athlete."""
    fig = px.line(
        df_series,
        x="date",
        y=value_col,
        title=title,
        markers=True,
        template="plotly_dark",
    )
    fig.update_layout(height=300, yaxis_title=y_label, margin=dict(l=20, r=20, t=40, b=20))
    return fig
