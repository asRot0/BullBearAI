import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir / "src"))

from visualization import eda_plot

# st.title(":material/insights: Exploratory Data Analysis")
st.title("Exploratory Data Analysis")
st.caption("Visualize key patterns, trends, and correlations for deeper market insights.")


# st.header("Chart elements")
chart_data = st.session_state.chart_data
map_data = st.session_state.map_data

tabs = st.tabs(["Chart Elements", "EDA Dashboard"])

with tabs[0]:
    st.subheader("Area chart")
    st.area_chart(chart_data)
    st.subheader("Bar chart")
    st.bar_chart(chart_data)
    st.subheader("Line chart")
    st.line_chart(chart_data)
    st.subheader("Scatter chart")
    st.scatter_chart(chart_data)
    st.subheader("Map")
    st.map(map_data, color=st.get_option("theme.chartCategoricalColors")[0])

with tabs[1]:
    if "raw_data" not in st.session_state or st.session_state.raw_data is None:
        st.warning("⚠ Please upload data first from the 'Data' page.")
    else:
        df = st.session_state.raw_data

        cols = st.columns([4, 1])
        show_all = cols[1].toggle("Show all plots", value=False)

        if not show_all:
            plot_mode = cols[0].radio(
                "Plot Mode 🐬",
                ["Single Plot", "Multi Plot"],
                horizontal=True
            )

            eda_options = [
                "Volume Distribution",
                "Price Distribution",
                "Time Series",
                "Correlation Analysis",
                "Outliers",
                "Outlier Detection",
                "Rolling Averages",
                "Seasonal Decomposition",
                "Moving Averages",
                "Relative Strength Index"
            ]

            cols = st.columns([3, 2])
            if plot_mode == "Single Plot":
                eda_option = cols[0].selectbox("🍂 Select an EDA plot", eda_options, index=0)
                run_plot = st.button("Run Plot", type="primary")

                if run_plot:
                    st.subheader(eda_option)
                    if eda_option == "Volume Distribution":
                        st.pyplot(eda_plot.volume_distribution(df))
                    elif eda_option == "Price Distribution":
                        st.pyplot(eda_plot.price_plots(df))
                    elif eda_option == "Time Series":
                        st.pyplot(eda_plot.line_plots(df))
                    elif eda_option == "Correlation Analysis":
                        st.pyplot(eda_plot.correlation_heatmap(df))
                    elif eda_option == "Outliers":
                        st.pyplot(eda_plot.outlier_plots(df))
                    elif eda_option == "Outlier Detection":
                        st.pyplot(eda_plot.outlier_detection(df))
                    elif eda_option == "Rolling Averages":
                        st.pyplot(eda_plot.rolling_avg_plot(df))
                    elif eda_option == "Seasonal Decomposition":
                        st.pyplot(eda_plot.seasonal_decomposition(df))
                    elif eda_option == "Moving Averages":
                        st.pyplot(eda_plot.sma_ema_plot(df))
                    elif eda_option == "Relative Strength Index":
                        st.pyplot(eda_plot.rsi_plot(df))

            else:
                multi_options = cols[0].multiselect(
                    "🦋 Select one or more EDA plots to display",
                    eda_options
                )
                run_multi_plot = st.button("Run Selected Plots", type="primary")

                if run_multi_plot and multi_options:
                    plot_cols = st.columns(2)
                    col_idx = 0

                    for opt in multi_options:
                        with plot_cols[col_idx].expander(f"{opt}", expanded=True):
                            fig = None
                            if opt == "Volume Distribution":
                                fig = eda_plot.volume_distribution(df)
                            elif opt == "Price Distribution":
                                fig = eda_plot.price_plots(df)
                            elif opt == "Time Series":
                                fig = eda_plot.line_plots(df)
                            elif opt == "Correlation Analysis":
                                fig = eda_plot.correlation_heatmap(df)
                            elif opt == "Outliers":
                                fig = eda_plot.outlier_plots(df)
                            elif opt == "Outlier Detection":
                                fig = eda_plot.outlier_detection(df)
                            elif opt == "Rolling Averages":
                                fig = eda_plot.rolling_avg_plot(df)
                            elif opt == "Seasonal Decomposition":
                                fig = eda_plot.seasonal_decomposition(df)
                            elif opt == "Moving Averages":
                                fig = eda_plot.sma_ema_plot(df)
                            elif opt == "Relative Strength Index":
                                fig = eda_plot.rsi_plot(df)

                            if fig:
                                st.pyplot(fig, clear_figure=True)
                        col_idx = 1 - col_idx


        else:
            st.success("🧪 Displaying all EDA plots")
            with st.spinner("Generating all plots..."):
                with st.expander("📦 Volume Distribution", expanded=True):
                    st.pyplot(eda_plot.volume_distribution(df))

                with st.expander("💲 Price Distribution"):
                    st.pyplot(eda_plot.price_plots(df))

                with st.expander("⏱️ Time Series Trends"):
                    st.pyplot(eda_plot.line_plots(df))

                with st.expander("🧩 Correlation Analysis"):
                    st.pyplot(eda_plot.correlation_heatmap(df))

                with st.expander("🪨 Outliers (Boxplots)"):
                    st.pyplot(eda_plot.outlier_plots(df))

                with st.expander("🚨 Outlier Detection (Z-score)"):
                    st.pyplot(eda_plot.outlier_detection(df))

                with st.expander("🌀 Rolling Averages (7-day & 30-day)"):
                    st.pyplot(eda_plot.rolling_avg_plot(df))

                with st.expander("📆 Seasonal Decomposition"):
                    st.pyplot(eda_plot.seasonal_decomposition(df))

                with st.expander("📏 Moving Averages"):
                    st.pyplot(eda_plot.sma_ema_plot(df))

                with st.expander("⚡ Relative Strength Index (RSI)"):
                    st.pyplot(eda_plot.rsi_plot(df))
            st.info("✅ All plots rendered successfully.")