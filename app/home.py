import streamlit as st
from cards import (
    dataframe_card,
    charts_card,
    chat_card,
    media_card,
    status_card,
    metrics_card,
)


st.title(":material/hub: BullBearAI")
st.caption("Advanced Stock Market Prediction & Analysis Platform")

# Two-column layout for intro + highlights
cols = st.columns([2, 1])

with cols[0].container(border=True):
    st.header("Welcome to BullBearAI!")
    st.markdown(
        """
BullBearAI is an integrated platform for **stock market analysis** and **price prediction** using advanced  
**Machine Learning** and **Deep Learning** models.

**Key Capabilities**
- Load and preprocess stock data seamlessly  
- Explore trends with interactive **EDA & visualization tools**  
- Build and compare **ARIMA, SVM, CNN, LSTM, and Hybrid models**  
- Evaluate performance using detailed **metrics & dashboards**

:material/navigation: Use the **sidebar** to explore different modules.
"""
    )
    st.divider()
    st.subheader("Quick Highlights")
    st.markdown(
    ":green[:material/auto_awesome_motion: Clean & preprocess data]  | "
    ":blue[:material/insights: Interactive visualizations]  | "  
    ":orange[:material/smart_toy: ML & DL models]  | "
    ":violet[:material/leaderboard: Performance tracking]"
    )

with cols[1].container(border=True):
    st.markdown("### :material/auto_graph: Key Features")
    st.badge("Data Upload", icon=":material/file_upload:", color="green")
    st.badge("EDA Plots", icon=":material/show_chart:", color="blue")
    st.badge("Model Training", icon=":material/computer:", color="orange")
    st.badge("Metrics", icon=":material/bar_chart:", color="violet")

    st.markdown(
        ":green[:material/auto_awesome:] Sleek & Modern UI  |  "
        ":orange[:material/emoji_objects:] Intuitive User Flow  |  "
        ":blue[:material/trending_up:] Insightful Performance"
    )

st.divider()
st.subheader("Quick Start Example")
st.code(
    """
import pandas as pd
from src.data_loader.load_data import load_stock_data

# Load data
df = load_stock_data('data/raw/AAPL.csv')

# Visualize closing price
st.line_chart(df['Close/Last'])
"""
)
st.caption("💡 You can directly load and visualize your stock data using BullBearAI's pipeline.")


cols = st.columns(2)
with cols[0].container(height=310):
    dataframe_card()
with cols[1].container(height=310):
    charts_card()
with cols[0].container(height=310):
    metrics_card()
with cols[1].container(height=310):
    chat_card()
with cols[0].container(height=310):
    media_card()
with cols[1].container(height=310):
    status_card()


st.divider()
st.markdown(
    """
    <div style="text-align:center; font-size:14px; color:#555;">
    Built with ❤️ by Asif Ahmed | 
    <a href="https://github.com/asRot0/BullBearAI/app/home.py" target="_blank">GitHub Repo</a> |
    Version 1.0
    </div>
    """,
    unsafe_allow_html=True
)