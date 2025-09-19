import os
import pandas as pd

PRED_FILE = "saved_models/predictions.csv"

def save_predictions(model_name, y_true, y_pred, clear=False):
    """
    Save predictions for a model into a shared CSV file.

    Parameters:
        model_name (str): name of the model (column will be created if new).
        y_true (array-like): ground truth values (only written first time).
        y_pred (array-like): predicted values for this model.
        clear (bool): if True, clear previous CSV and start fresh.
    """
    # Convert to DataFrame
    df_new = pd.DataFrame({
        "y_true": list(y_true),
        model_name: list(y_pred)
    })

    if clear or not os.path.exists(PRED_FILE):
        # Start fresh
        df_new.to_csv(PRED_FILE, index=False)
        print(f"[INFO] Created new prediction file with model={model_name}")
    else:
        df = pd.read_csv(PRED_FILE)

        # Ensure same length
        min_len = min(len(df), len(df_new))
        df = df.iloc[:min_len]
        df_new = df_new.iloc[:min_len]

        # If y_true missing, add
        if "y_true" not in df.columns:
            df["y_true"] = df_new["y_true"]

        # Add/replace model column
        df[model_name] = df_new[model_name]
        df.to_csv(PRED_FILE, index=False)
        print(f"[INFO] Updated predictions with model={model_name}")


def clear_predictions():
    """Delete/reset the predictions file."""
    if os.path.exists(PRED_FILE):
        os.remove(PRED_FILE)
        print("[INFO] Cleared prediction file.")
    else:
        print("[INFO] No prediction file found to clear.")
