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


def plot_risk_gauge(probability: float, threshold: float = 0.5) -> go.Figure:
    """Plots a risk gauge for an individual athlete."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={'text': "Injury Risk Probability"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#ff5252" if probability >= threshold else "#00e676"},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, threshold * 100], 'color': "rgba(0, 230, 118, 0.3)"},
                {'range': [threshold * 100, 100], 'color': "rgba(255, 82, 82, 0.3)"}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark")
    return fig


def plot_shap_waterfall_interactive(shap_dict: Dict[str, float]) -> go.Figure:
    """Plots an interactive SHAP waterfall chart for a single athlete."""
    if not shap_dict:
        return go.Figure()
        
    # Sort by absolute impact
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    # Reverse to plot largest at the top/end
    sorted_shap.reverse()
    
    features = [x[0] for x in sorted_shap]
    values = [x[1] for x in sorted_shap]
    
    colors = ["#ff5252" if v > 0 else "#00e676" for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside"
    ))
    
    fig.update_layout(
        title="Top 10 Risk Factors (SHAP Contributions)",
        xaxis_title="Impact on Risk Probability (Log-odds / Prob)",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plots an interactive correlation heatmap for numerical features."""
    corr = df.select_dtypes(include=[np.number]).corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        hoverinfo='text',
    ))
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=700,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark"
    )
    return fig
