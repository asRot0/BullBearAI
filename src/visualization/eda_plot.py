import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
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

def rolling_avg_plot(df):
    # Rolling averages (7-day & 30-day)
    df['Close_7d'] = df['Close/Last'].rolling(window=7).mean()
    df['Close_30d'] = df['Close/Last'].rolling(window=30).mean()

    # Define more vibrant custom colors
    close_color = "#FF6F61"  # Coral
    ma7_color = "#6A5ACD"  # Slate Blue
    ma30_color = "#20B2AA"  # Light Sea Green
    gap_color = "#FFD700"  # Gold

    # Create the plot with custom background color
    fig, ax = plt.subplots(figsize=(12, 6))
    # fig.patch.set_facecolor('#f0f0f5')  # Light gray background
    # ax.set_facecolor('#fafafa')         # Slightly off-white axes area

    # Plot lines
    ax.plot(df['Date'], df['Close/Last'], label='Close', color=close_color, linewidth=2)
    ax.plot(df['Date'], df['Close_7d'], label='7-Day MA', color=ma7_color, linestyle='--', linewidth=2)
    ax.plot(df['Date'], df['Close_30d'], label='30-Day MA', color=ma30_color, linestyle=':', linewidth=2)

    # Fill between MAs with attractive color
    ax.fill_between(df['Date'], df['Close_7d'], df['Close_30d'],
                    where=(df['Close_7d'] > df['Close_30d']),
                    interpolate=True, color='lightgreen', alpha=0.3, label='Bullish Gap')

    ax.fill_between(df['Date'], df['Close_7d'], df['Close_30d'],
                    where=(df['Close_7d'] <= df['Close_30d']),
                    interpolate=True, color='lightcoral', alpha=0.3, label='Bearish Gap')

    # Customizations
    ax.set_title('Close Price with 7-Day & 30-Day Moving Averages',
                 fontsize=14, fontweight='bold', color='#333333')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    sns.despine()

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

def seasonal_decomposition(df):
    # Perform decomposition
    decomposition = seasonal_decompose(df['Close/Last'], model='additive', period=30)

    # Extract components
    observed = decomposition.observed
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Define colors
    colors = {
        'observed': '#1f77b4',  # Blue
        'trend': '#2ca02c',  # Green
        'seasonal': '#ff7f0e',  # Orange
        'residual': '#d62728'  # Red
    }

    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#f2f2f2')

    # Plot each component with colors and labels
    axes[0].plot(df['Date'], observed, color=colors['observed'], label='Observed', linewidth=2)
    axes[0].set_title('Observed', fontsize=12, weight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].plot(df['Date'], trend, color=colors['trend'], label='Trend', linewidth=2)
    axes[1].set_title('Trend', fontsize=12, weight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    axes[2].plot(df['Date'], seasonal, color=colors['seasonal'], label='Seasonal', linewidth=2)
    axes[2].set_title('Seasonal', fontsize=12, weight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    axes[3].scatter(df['Date'], residual, color=colors['residual'], label='Residual', s=10)
    axes[3].set_title('Residual', fontsize=12, weight='bold')
    axes[3].grid(True, linestyle='--', alpha=0.7)

    # Highlight residuals greater than 2 standard deviations from the mean
    threshold = 2 * residual.std()
    anomalies = residual[abs(residual) > threshold]
    axes[3].scatter(df['Date'][anomalies.index], anomalies, color='yellow', label='Anomalies', s=30, zorder=5)

    # Apply color gradient to seasonal component (gradient based on date)
    norm = plt.Normalize(seasonal.min(), seasonal.max())  # Normalize the seasonal component
    cmap = cm.coolwarm  # Choose a color map (can be 'plasma', 'coolwarm', etc.)

    # Plot gradient line + shaded area under seasonal component
    for i in range(1, len(seasonal)):
        x = [df['Date'][i - 1], df['Date'][i]]
        y = [seasonal[i - 1], seasonal[i]]
        color = cmap(norm(seasonal[i]))

        # Line segment
        axes[2].plot(x, y, color=color, lw=2)
        # Shaded area (fill between line and y=0)
        axes[2].fill_between(x, y, [0, 0], color=color, alpha=0.3)

    # Customize x-axis
    for ax in axes:
        ax.label_outer()
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Seasonal Decomposition of Close Price', fontsize=15, weight='bold')
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    return plt