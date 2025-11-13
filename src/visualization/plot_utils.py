import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

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

def plot_model_comparison_heatmap(json_path: str, figsize=(12, 6), cmap="RdBu_r"):
    # Load metrics JSON
    with open(json_path, "r") as f:
        metrics = json.load(f)

    # Convert JSON → DataFrame (wide format)
    df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})

    # Convert wide → long format
    df_melt = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    # Pivot for heatmap (Metric = rows, Model = columns)
    pivot_df = df_melt.pivot(index="Metric", columns="Model", values="Score")

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        cbar_kws={'label': 'Performance Score'}
    )
    plt.title("Model Performance Comparison (All Metrics)", fontsize=16, weight="bold")
    plt.ylabel("Metric")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

def plot_model_comparison_bars(json_path: str, drop_models=None, figsize=None, cmap="RdBu_r"):
    """
    Plot normalized horizontal bar charts per model across metrics
    (signed, so bars extend left/right relative to peers).
    """
    # Load metrics JSON
    with open(json_path, "r") as f:
        metrics = json.load(f)

    df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    # print(df.head())
    # print(df['Model'])
    if drop_models:
        df.drop(drop_models, inplace=True)
    df_melt = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    # Metrics where lower is better
    lower_is_better = {"MAE", "RMSE", "Max Drawdown"}

    # Flip sign so higher always = better
    df_melt["AdjScore"] = df_melt.apply(
        lambda r: -r["Score"] if r["Metric"] in lower_is_better else r["Score"], axis=1
    )

    # Normalize per metric to [-1, 1]
    pivot = df_melt.pivot(index="Metric", columns="Model", values="AdjScore")
    pivot_norm = pivot.apply(
        lambda row: ((row - row.min()) / (row.max() - row.min()) * 2 - 1)
        if row.max() > row.min() else 0,
        axis=1
    )

    metrics_list = pivot_norm.index.tolist()
    models = pivot_norm.columns.tolist()

    # Color setup
    cmap = plt.get_cmap(cmap)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Figure size
    n_models = len(models)
    fig_w = max(10, n_models * 2)
    fig_h = max(5, len(metrics_list) * 0.5)
    if figsize is None:
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(ncols=n_models, figsize=figsize, sharey='col')

    if n_models == 1:
        axes = [axes]

    y = np.arange(len(metrics_list))

    for ax, model in zip(axes, models):
        vals = pivot_norm[model].values
        colors = cmap(norm(vals))
        ax.barh(y=y, width=vals, color=colors, edgecolor="none")
        ax.set_xlim(-1.1, 1.1)
        ax.set_title(model, fontsize=9, weight="bold")
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.4)

        if ax is axes[0]:
            ax.set_yticks(y)
            ax.set_yticklabels(metrics_list, fontsize=9)
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([""] * len(metrics_list))

        ax.set_xticks([])

    # Colorbar on the left
    cbar_ax = fig.add_axes([0.05, 0.12, 0.02, 0.75])
    fig.colorbar(sm, cax=cbar_ax, orientation="vertical", label="Relative Performance")

    fig.suptitle("Model Comparison Across Metrics", fontsize=14, weight="bold")
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.4, top=0.9, bottom=0.08)
    plt.show()

def plot_model_comparison_heatmap_app(json_path: str, figsize=(12, 6), cmap="RdBu_r"):
    # Load metrics JSON
    with open(json_path, "r") as f:
        metrics = json.load(f)

    # Convert JSON → DataFrame (wide format)
    df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})

    # Convert wide → long format
    df_melt = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    # Pivot for heatmap (Metric = rows, Model = columns)
    pivot_df = df_melt.pivot(index="Metric", columns="Model", values="Score")

    # Plot heatmap
    # plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        cbar_kws={'label': 'Performance Score'},
        ax=ax
    )
    # ax.set_title("Model Performance Comparison (All Metrics)", fontsize=16, weight="bold")
    ax.set_ylabel("Metric")
    ax.set_xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    return fig