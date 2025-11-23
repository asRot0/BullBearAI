import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")

if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False

if "df_results" not in st.session_state:
    st.session_state["df_results"] = None

if "model_metrics" not in st.session_state:
    st.session_state["model_metrics"] = None

if "show_reset_dialog" not in st.session_state:
    st.session_state["show_reset_dialog"] = False


@st.cache_resource
def load_artifacts():
    model = load_model("../saved_models/hybrid_cnn_lstm_model.keras", compile=False)
    scaler = joblib.load("../saved_models/hybrid_cnn_lstm_scaler.pkl")
    return model, scaler


@st.dialog("🔄 Confirm Reset")
def reset_confirmation():
    if st.session_state["show_reset_dialog"]:
        st.write("""Are you sure you want to reset the dashboard?
        All predictions, charts, and tables will be cleared.""")

        colA, colB = st.columns(2)

        if colA.button("❌ Cancel", use_container_width=True):
            st.session_state["show_reset_dialog"] = False
            st.rerun()

        if colB.button("✅ Yes, Reset", type="primary", use_container_width=True):
            st.session_state["prediction_done"] = False
            st.session_state["df_results"] = None
            st.session_state["show_reset_dialog"] = False
            st.rerun()


st.title("Model Evaluation")
st.caption("Load pre-trained models, generate predictions, and view instant performance insights.")
# st.caption("Interactive metrics, advanced analytics, and dynamic row-based selection.")
# st.divider()

# st.subheader(":material/upload_file: Upload Dataset")
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload your processed dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    test_split = st.slider("Select Test Split (%)", 10, 80, 20)

    run_prediction = st.button(
        ":material/rocket_launch: Run Prediction",
        type="primary",
        use_container_width=True
    )

    if run_prediction and uploaded_file is not None:
        with st.spinner("Processing data and generating predictions..."):
            time.sleep(2)
            try:
                data = pd.read_csv(uploaded_file, index_col=0)
                model, scaler = load_artifacts()

                target_column = "Target_Close_Next_Day"
                features = data.drop(columns=[target_column])
                combined = pd.concat([features, data[[target_column]]], axis=1)

                scaled = scaler.transform(combined)
                scaled_df = pd.DataFrame(scaled, columns=list(features.columns) + [target_column])

                def create_sequences(df, target, window=3):
                    X, y = [], []
                    for i in range(window, len(df)):
                        X.append(df.iloc[i-window:i].values)
                        y.append(df.iloc[i-1][target])
                    return np.array(X), np.array(y)

                X, y = create_sequences(scaled_df, target_column)
                split = int((1 - test_split / 100) * len(X))
                X_test, y_test = X[split:], y[split:]
                df_results = data.iloc[-len(y_test):].copy()

                y_pred = model.predict(X_test).reshape(-1, 1)
                y_test = y_test.reshape(-1, 1)

                # inverse scaling
                y_pred_d = np.zeros((len(y_pred), scaled_df.shape[1]))
                y_test_d = np.zeros((len(y_test), scaled_df.shape[1]))
                y_pred_d[:, -1] = y_pred[:, 0]
                y_test_d[:, -1] = y_test[:, 0]

                df_results["Predicted"] = scaler.inverse_transform(y_pred_d)[:, -1]
                df_results["Actual"]   = scaler.inverse_transform(y_test_d)[:, -1]

                # SAVE TO SESSION STATE
                st.session_state["prediction_done"] = True
                st.session_state["df_results"] = df_results

                st.toast("🎉 Prediction completed successfully!")
                # st.rerun()
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.session_state["prediction_done"] = False


if st.session_state["prediction_done"]:
    df_results = st.session_state["df_results"]

    tab_overview, tab_table, tab_chart, tab_stats = st.tabs(
        ["📌 Overview", "📋 Table", "💹 Chart", "🗃️ Details"]
    )

    with tab_overview:
        st.subheader("📌 Metrics Summary")
        # st.badge("Metrics Summary", icon="📌", color="green")

        event = st.dataframe(
            df_results,
            use_container_width=True,
            selection_mode="multi-row",
            on_select="rerun",
            key="metric_table"
        )

        # selected or full data
        if event and event.selection and event.selection.rows:
            df_sel = df_results.iloc[event.selection.rows]
        else:
            df_sel = df_results

        actual = df_sel["Actual"].values
        pred = df_sel["Predicted"].values

        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)
        da = (
            np.mean(
                np.sign(actual[1:] - actual[:-1]) == np.sign(pred[1:] - pred[:-1])
            ) * 100
            if len(actual) > 1
            else 0
        )

        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        volatility = np.std(pred - actual)
        sharpe = (np.mean(pred - actual)) / (volatility + 1e-9)

        col1, col2, col3, col4 = st.columns(4)
        col5, _, col6, _ = st.columns(4)

        col1.metric(":material/trending_down: RMSE", round(rmse, 3))
        col2.metric(":material/trending_up: R² Score", round(r2, 3), border=True)
        col3.metric(":material/trending_up: Directional Accuracy", f"{round(da,2)}%")
        col4.metric(":material/trending_down: MAPE", f"{round(mape,2)}%", border=True)
        col5.metric(":material/show_chart: Volatility", round(volatility, 4), border=True)
        col6.metric(":material/show_chart: Sharpe Ratio", round(sharpe, 4), border=True)

        st.session_state["model_metrics"] = {
            "r2": round(r2, 4),
            "mape": round(mape, 4),
            "volatility": round(volatility, 4)
        }

        if st.button("🔄 Reset Dashboard", type="secondary", use_container_width=False):
            st.session_state["show_reset_dialog"] = True
            reset_confirmation()

    with tab_table:
        st.subheader("📋 Predictions Table")
        st.table(df_results)

    with tab_chart:
        st.subheader("💹 Actual vs Predicted Chart")
        show_selected = st.toggle("Show only selected rows", False)

        if show_selected and event and event.selection.rows:
            plot_df = df_sel
        else:
            plot_df = df_results

        c1 = "#4AB2FF"
        # c1 = "#7FE1FF"
        c2 = "#FF6EC7"
        # c2 = "#FFB277"

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(plot_df["Actual"].values, label="Actual", linewidth=2, color=c1)
        ax.plot(plot_df["Predicted"].values, label="Predicted", linestyle="--", color=c2)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    with tab_stats:
        st.subheader("🗃️ Detailed Statistics")
        st.write(df_results.describe())

        st.info("""
        ### 📌 Key Insight  
        Metrics dynamically adjust based on your row selections.  
        Use this to explore model behavior across specific periods.
        """)
