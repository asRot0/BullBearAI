import streamlit as st
import pandas as pd
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir / "src"))

from visualization import plot_utils

st.title("Model Metrics")
# st.title(":material/bar_chart: Model Metrics")
st.caption("Evaluate and compare ML/DL model performance with interactive metrics and heatmaps")

json_path = "../saved_models/metrics_test.json"
metrics_df = pd.DataFrame(plot_utils.load_metrics(json_path)).T

with st.container():
    display_type = st.segmented_control("Metrics Overview", ["Dataframe", "JSON"])

if display_type == "Dataframe":
    st.dataframe(metrics_df.style.format("{:.2f}"), use_container_width=False)
elif display_type == "JSON":
    st.json(metrics_df.to_dict())


metrics = metrics_df.copy()
core_metrics = ["RMSE", "R2", "DA"]

best = {
    "RMSE": metrics["RMSE"].idxmin(),   # lower = better
    "R2": metrics["R2"].idxmax(),       # higher = better
    "DA": metrics["DA"].idxmax()        # higher = better
}

insights = {
    "Best RMSE": ("RMSE", best["RMSE"]),
    "Best R²": ("R2", best["R2"]),
    "Best DA (%)": ("DA", best["DA"])
}

averages = metrics[core_metrics].mean()
icons = {
    "RMSE": ":material/monitoring:",
    "R2": ":material/insights:",
    "DA": ":material/trending_up:"
}

cols = st.columns(3)

for idx, (label, (metric_name, model)) in enumerate(insights.items()):
    value = metrics.loc[model, metric_name]
    delta = value - averages[metric_name]

    with cols[idx].container(border=True):
        st.metric(
            label=f"{label} {icons[metric_name]}  \n**{model}**",
            value=round(value, 3),
            delta=round(delta, 3)
        )

st.caption(
    ":green[:material/arrow_upward:] Positive delta"
    " | :red[:material/arrow_downward:] Negative delta"
)

st.divider()
cols = st.columns([2, 1.5])

with cols[0].container(border=True):
    st.subheader(":green[:material/insights:] Performance Overview")
    st.markdown(
        "Quick glance at model evaluation metrics and comparisons. "
        "Select a model below to view detailed metrics."
    )
    fig_heatmap = plot_utils.plot_model_comparison_heatmap_app(json_path=json_path, figsize=(10,6))
    st.pyplot(fig_heatmap, width="content")

with cols[1].container(border=True):
    st.subheader(":blue[:material/trending_up:] Key Metric Comparison")
    metric = st.selectbox("Select Metric", options=list(metrics_df.columns), index=1)
    fig_bars = plot_utils.plot_model_comparison_bar_app(json_path=json_path, metric=metric)
    st.pyplot(fig_bars, width="content")
