import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
from datetime import datetime

# Plot settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

def price_plots(df):
    price_cols = ['Close/Last', 'Open', 'High', 'Low']
    palette = sns.color_palette("Set2", len(price_cols))  # Colorful and distinct

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(price_cols):
        sns.kdeplot(df[col], fill=True, color=palette[i], ax=axes[i], linewidth=2)
        axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Density')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # Remove top and right spines for a cleaner look
    sns.despine()

    plt.tight_layout()
    return plt

def line_plots(df):
    # Line plots
    cols = ['Close/Last', 'Open', 'Volume']
    colors = sns.color_palette("husl", len(cols))  # Vibrant distinct colors

    # Create subplots in one row
    fig, axes = plt.subplots(1, len(cols), figsize=(18, 5), sharex=True)

    for i, col in enumerate(cols):
        axes[i].plot(df['Date'], df[col], color=colors[i], linewidth=2)
        axes[i].set_title(f'{col} Over Time', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel(col)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, linestyle='--', alpha=0.8)

    # Adjust layout
    plt.tight_layout()
    return plt

def correlation_heatmap(df):
    # Pearson correlation
    corr_matrix = df[['Close/Last', 'Open', 'High', 'Low', 'Volume']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='rocket', fmt=".2f")
    # plt.title("Correlation Heatmap")
    return plt

def outlier_plots(df):
    cols = ['Close/Last', 'Open', 'High', 'Low', 'Volume']
    palette = sns.color_palette("Set3", len(cols))

    # Create subplots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(x=df[col], ax=axes[i], color=palette[i], linewidth=1.5)
        axes[i].set_title(f'Boxplot for {col}', fontsize=12, fontweight='bold')
        axes[i].grid(True, linestyle='--', alpha=0.8)

    # Hide any unused subplot (in case number of plots < subplots)
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])  # Clean up extra subplot

    sns.despine()
    plt.tight_layout()
    return plt

def outlier_detection(df):
    z_scores = np.abs(zscore(df[['Close/Last', 'Open', 'High', 'Low', 'Volume']]))
    outliers = (z_scores > 3).any(axis=1)
    df_outliers = df[outliers]

    # Use light yellow-orange theme
    base_color = '#FFE699'  # soft pastel yellow
    highlight_color = '#FFA07A'  # light salmon-orange

    # Plot Volume Boxplot with highlighted outliers
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df['Volume'], color=base_color, linewidth=2)

    # Overlay dots for detected outliers
    sns.stripplot(x=df_outliers['Volume'], color=highlight_color, size=8, jitter=0.15, label="Anomalies", zorder=10)

    plt.title("Volume Distribution with Highlighted Outliers", fontsize=14, fontweight='bold')
    plt.xlabel("Volume")
    plt.legend()
    sns.despine()
    plt.grid(True, linestyle="--", alpha=0.8)
    plt.tight_layout()
    return plt

def volume_distribution(df):

    # Volume Distribution
    num_bins = 50

    # Compute histogram using density=True
    counts, bins = np.histogram(df['Volume'], bins=num_bins, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Random pastel colors
    colors = sns.color_palette("pastel", num_bins)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw bars with density heights
    for i in range(num_bins):
        ax.bar(
            bin_centers[i],
            counts[i],
            width=(bins[1] - bins[0]),
            color=colors[i % len(colors)],
            edgecolor='black',
            alpha=0.9
        )

    # KDE line
    sns.kdeplot(df['Volume'], ax=ax, color='red', linewidth=1.5, label='KDE')

    # Customize plot
    # ax.set_title('Volume Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Volume')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.8)
    sns.despine()

    plt.tight_layout()
    return fig
