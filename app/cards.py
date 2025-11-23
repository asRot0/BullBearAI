import streamlit as st
import sys
import pandas as pd
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir / "src"))

from visualization import plot_utils


def dataframe_card():
    st.page_link("data.py", label="Data", icon=":material/table:")
    st.dataframe(st.session_state.chart_data, height=220)


def charts_card():
    st.page_link("charts.py", label="Charts", icon=":material/insert_chart:")
    st.bar_chart(st.session_state.chart_data, height=230)


def media_card():
    st.page_link("media.py", label="Media", icon=":material/image:")
    st.video("https://s3-us-west-2.amazonaws.com/assets.streamlit.io/videos/hero-video.mp4", autoplay=True)


def chat_card():
    st.page_link("chat.py", label="Chat", icon=":material/chat:")
    st.chat_message("user").write("Hello, world!")
    st.chat_message("assistant").write("Hello, user!")
    st.chat_input("Type something")


def status_card():
    st.page_link("status.py", label="Status", icon=":material/error:")
    cols = st.columns(2)
    cols[0].error("Error")
    cols[0].warning("Warning")
    cols[1].info("Info")
    cols[1].success("Success")
    cols = st.columns(2)
    if cols[0].button("Snow!"):
        st.snow()
    if cols[1].button("Balloons!"):
        st.balloons()


def metrics_card():
    st.page_link("metrics.py", label="Metrics", icon=":material/bar_chart:")

    # Short concise description
    st.caption("A quick comparison of model performance across key metrics.")

    # The heatmap
    fig = plot_utils.plot_model_comparison_heatmap_app(
        json_path='../saved_models/metrics_test.json',
        figsize=(12, 6),
        cmap='RdBu'
    )
    st.pyplot(fig, width="content")

    # Tip under the plot
    st.caption(":material/info: Tip: Darker cells indicate stronger performance.")


def model_card():
    st.page_link("model.py", label="AI Model", icon=":material/memory:")
    metrics = st.session_state.get("model_metrics", None)
    df = st.session_state.get("df_results", None)

    if metrics is None or df is None:
        st.info("Run your model to see metrics preview here.")
        return

    # Prepare micro-chart data (last 20 points)
    last_point = 40
    # actual_trend = df["Actual"].values[-last_point:].tolist()
    pred_trend = df["Predicted"].values[-last_point:].tolist()
    error_trend = (df["Predicted"] - df["Actual"]).values[-last_point:].tolist()
    volatility_trend = pd.Series(error_trend).rolling(5).std().fillna(0).tolist()
    mape_trend = (np.abs((df["Actual"] - df["Predicted"]) / df["Actual"]) * 100).values[-last_point:].tolist()

    mape = metrics["mape"]
    r2 = metrics["r2"]
    vol = metrics["volatility"]

    row = st.container(horizontal=True)
    with row:
        mape_delta = round(mape_trend[-1] - mape_trend[-2], 3)

        st.metric(
            "MAPE (%)",
            f"{round(mape, 2)}%",
            delta=mape_delta,
            chart_data=mape_trend,
            chart_type="area",
            border=True,
            delta_color="normal"  # lower is better
        )

        r2_delta = round(r2 - 0.5, 3)  # relative to average baseline
        st.metric(
            "R² Score",
            round(r2, 3),
            delta=r2_delta,
            chart_data=pred_trend,
            chart_type="line",
            border=True,
            delta_color="normal"  # higher is better
        )

        vol_delta = round(volatility_trend[-1] - volatility_trend[-2], 4)

        st.metric(
            "Volatility",
            round(vol, 4),
            delta=vol_delta,
            chart_data=volatility_trend,
            chart_type="bar",
            border=True,
            delta_color="inverse"  # lower is better
        )
