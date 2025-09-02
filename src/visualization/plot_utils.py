import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, title="Predicted vs Actual"):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

def plot_residuals(y_true, y_pred, title="Residuals Plot"):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(residuals)), residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Residuals")
    plt.show()

def plot_training_history(history, title="Training History"):
    if not history:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_model_comparison(results, metric="RMSE", title="Model Comparison"):
    """
    results: dict {model_name: {"MAE":..., "RMSE":..., "R2":...}}
    """
    models = list(results.keys())
    scores = [results[m][metric] for m in models]

    plt.figure(figsize=(8, 5))
    plt.bar(models, scores)
    plt.title(f"{title} ({metric})")
    plt.ylabel(metric)
    plt.xticks(rotation=30)
    plt.show()
