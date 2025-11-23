import streamlit as st
import pandas as pd
import numpy as np
import os

# st.title(":material/grid_on: Data Overview")
st.title("Data Overview")
st.caption("Interactively browse your dataset with descriptive insights and quick summaries.")


if "raw_data" not in st.session_state:
    st.session_state.raw_data = None
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None
chart_data = st.session_state.chart_data

tabs = st.tabs(["Display type", "Explore", "Visualize", "Feature Eng", "ML Prep", "Clean & Save", "Upload"])

with tabs[0]:
    with st.container():
        display_type = st.segmented_control("View Mode", ["Dataframe", "Data editor", "Table", "JSON"], default="Dataframe")

    cols = st.columns(3)
    event = None
    if display_type == "Dataframe":
        st.info("Select rows to compute metrics for a subset of the data.")
        event = st.dataframe(chart_data, use_container_width=True, on_select="rerun", selection_mode="multi-row")
        if 'df' in st.session_state:
            df = st.session_state.df
            st.write(df.head())
    elif display_type == "Data editor":
        st.data_editor(chart_data, num_rows="dynamic", use_container_width=True)
    elif display_type == "Table":
        st.table(chart_data)
    elif display_type == "JSON":
        st.json(chart_data.to_dict(orient="records"), expanded=True)

    metric_values = {}
    if event is not None and event.selection.rows:
        metric_values["a_value"] = chart_data["a"].iloc[event.selection.rows].mean()
        metric_values["a_delta"] = metric_values["a_value"] - chart_data["a"].mean()
        metric_values["b_value"] = chart_data["b"].iloc[event.selection.rows].mean()
        metric_values["b_delta"] = metric_values["b_value"] - chart_data["b"].mean()
        metric_values["c_value"] = chart_data["c"].iloc[event.selection.rows].mean()
        metric_values["c_delta"] = metric_values["c_value"] - chart_data["c"].mean()
    else:
        metric_values["a_value"] = chart_data["a"].mean()
        metric_values["a_delta"] = chart_data["a"].std()
        metric_values["b_value"] = chart_data["b"].mean()
        metric_values["b_delta"] = -chart_data["b"].std()
        metric_values["c_value"] = chart_data["c"].mean()
        metric_values["c_delta"] = 0

    cols[0].metric(
        "Metric A",
        round(metric_values["a_value"],4),
        round(metric_values["a_delta"], 4),
        border=True
    )
    cols[1].metric(
        "Metric B",
        round(metric_values["b_value"],4),
        round(metric_values["b_delta"], 4),
        border=True
    )
    cols[2].metric(
        "Metric C",
        round(metric_values["c_value"],4),
        round(metric_values["c_delta"], 4),
        border=True
    )

with tabs[1]:
    st.subheader("Data Exploration")
    df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.raw_data
    if df is not None:
        st.write("Dataset Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write("Summary Statistics:")
        st.dataframe(df.describe())
        if st.checkbox("Show first 50 rows"):
            st.dataframe(df.head(50))
    else:
        st.info("Upload or clean data first.")

with tabs[2]:
    st.subheader("Visualizations")
    df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.raw_data
    if df is not None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        col = st.selectbox("Select column to visualize", options=numeric_cols)
        chart_type = st.radio("Chart Type", ["Line", "Bar", "Histogram"])

        if chart_type == "Line":
            st.line_chart(df[col])
        elif chart_type == "Bar":
            st.bar_chart(df[col])
        elif chart_type == "Histogram":
            st.write(df[col].hist())
            st.bar_chart(df[col].value_counts().sort_index())

    else:
        st.info("Upload or clean data first.")

with tabs[3]:
    st.subheader("Feature Engineering")
    df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.raw_data
    if df is not None:
        if st.checkbox("Add Moving Average"):
            window = st.slider("Window size", 2, 30, 5)
            for col in df.select_dtypes(include=np.number).columns:
                df[f"{col}_MA{window}"] = df[col].rolling(window).mean()
            st.session_state.cleaned_data = df
            st.success("Moving averages added!")
            st.dataframe(df.head())
    else:
        st.info("Upload or clean data first.")

with tabs[4]:
    st.subheader("ML / AI Preparation")
    df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.raw_data
    if df is not None:
        if st.checkbox("Show correlation matrix"):
            st.dataframe(df.corr())
        if st.checkbox("Select features for ML"):
            features = st.multiselect("Select feature columns", df.columns.tolist())
            target = st.selectbox("Select target column", df.columns.tolist())
            if features and target:
                st.write("Features:", features)
                st.write("Target:", target)
    else:
        st.info("Upload or clean data first.")

with tabs[5]:
    st.subheader("Data Cleaning")
    if st.session_state.raw_data is not None:
        df = st.session_state.raw_data.copy()

        # Cleaning options
        if st.checkbox("Drop duplicates"):
            df = df.drop_duplicates()
        if st.checkbox("Fill missing values with 0"):
            df = df.fillna(0)
        if st.checkbox("Drop rows with missing values"):
            df = df.dropna()

        st.session_state.cleaned_data = df
        st.dataframe(df.head())

        # Save cleaned file
        file_name = st.text_input("File name to save cleaned CSV", value="cleaned_data.csv")
        if st.button("Save Cleaned CSV"):
            df.to_csv(file_name, index=False)
            st.success(f"Cleaned data saved as {file_name}")
    else:
        st.info("Upload raw data first in the Upload tab.")

with tabs[6]:
    # st.subheader("Upload Raw Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.raw_data = df
        st.success('Data loaded!')
        st.dataframe(df.head())
