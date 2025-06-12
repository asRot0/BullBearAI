
## Project Progress Overview: BullBearAI
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![License](https://img.shields.io/github/license/your-username/BullBearAI)
![Status](https://img.shields.io/badge/Progress-Phase%202%20✅-green)

This project is designed to predict stock market trends using traditional ML, deep learning, and a hybrid LSTM-CNN architecture. Below is the step-by-step progress with brief descriptions.

### Project Structure

```
BullBearAI/
│
├── data/                        # Raw and processed stock data
│   ├── raw/                     # Untouched downloaded data
│   ├── interim/                 # Intermediate transformation outputs
│   └── processed/               # Cleaned and final datasets
│
├── notebooks/                  # Jupyter notebooks for EDA, modeling, evaluation
│   ├── 01_eda.ipynb                        # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb        # Feature engineering techniques
│   ├── 03_ml_baselines.ipynb               # Traditional ML models: SVM, RF, LR, Gradient Boosting
│   ├── 04_time_series_models.ipynb         # Time series statistical models: ARIMA, SARIMA, GARCH
│   ├── 05_cnn_model.ipynb                  # CNN-based deep learning model
│   ├── 06_lstm_model.ipynb                 # LSTM (RNN) based sequence model
│   ├── 07_hybrid_cnn_lstm_model.ipynb      # Hybrid CNN-LSTM deep model
│   └── 08_model_comparison.ipynb           # Evaluation & performance comparison
│
├── src/                        # All source code
│   ├── config/                 # Configuration files and parameters
│   │   └── config.yaml
│   ├── data_loader/            # Data loading and preprocessing scripts
│   │   └── load_data.py
│   ├── features/               # Feature engineering functions
│   │   └── technical_indicators.py
│   ├── models/                 # ML & DL model definitions
│   │   ├── arima_model.py
│   │   ├── svm_model.py
│   │   ├── cnn_model.py
│   │   ├── lstm_model.py
│   │   └── hybrid_model.py
│   ├── training/               # Training and validation loops
│   │   └── train_model.py
│   ├── evaluation/             # Metrics and model comparisons
│   │   └── evaluate.py
│   └── visualization/          # Custom plotting functions
│       └── plot_utils.py
│
├── saved_models/               # Checkpoints and final models (.h5 or .pth)
│
├── reports/                    # Analysis reports, result plots, performance graphs
│   ├── figures/
│   └── model_comparison.md
│
├── cli/                        # Command-line tools for automation
│   └── run_train.py
│
├── tests/                      # Unit tests for various components
│   ├── test_models.py
│   ├── test_utils.py
│   └── test_data_loader.py
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview, setup, and usage
├── LICENSE                     # License info
└── .gitignore                  # Files to ignore in version control
```


### Data Loading & Initial Inspection
- Loaded the raw stock market data (Netflix stock) from the `data/raw/` directory.
- Verified file integrity, parsed dates correctly, and ensured data types were appropriate.
- Saved a clean version in `data/processed/netflix_cleaned.csv`.

### Data Cleaning
- Removed duplicates and handled any missing/null values.
- Renamed columns for consistency and usability (`Close/Last` instead of `Close*`).
- Converted all date fields to `datetime` format.
- Ensured data is sorted chronologically.
- Exported cleaned dataset to `data/processed/`.

### Exploratory Data Analysis (EDA)
- Visualized time-series trends of `Close`, `Volume`, and `Open`.
- Used Seaborn and Matplotlib for:
  - Moving averages
  - Seasonal decomposition
  - Daily/Monthly return distributions
- Checked for trends, volatility, and patterns.
- Identified data gaps, outliers, or anomalies.
- All EDA work is saved in `notebooks/01_eda.ipynb`.

### Feature Engineering
Performed a comprehensive set of transformations to prepare predictive features:

#### Date-Based Features
- Extracted: `Year`, `Month`, `Day`, `DayOfWeek`, and `IsWeekend`.

#### Lag Features
- Created lagged versions of `Close/Last` and `Volume` (lags: 1, 2, 3 days).

#### Rolling Statistics
- Computed rolling means, medians, stds, max, min for 7, 14, and 30-day windows.

#### Volatility Measures
- Daily percentage change, return, and rolling return metrics.

#### Technical Indicators
- Simple & Exponential Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands

#### Target Variable
- `Target_Close_Next_Day`: Next day’s close price
- `Target_UpDown`: Binary classification target (1 = price goes up, 0 = down)

Engineered dataset saved to: `data/interim/engineered_features.csv`.

---

### Machine Learning Baseline Models (Regression)

This notebook builds baseline **regression models** to predict:

- **`Target_Close_Next_Day`** — the actual next-day closing price of the stock.

**Implemented Models**:
- Linear Regression  
- Support Vector Regression (SVR)  
- Random Forest Regressor  
- Gradient Boosting Regressor  

**Highlights**:
- Models trained on engineered features including lag features, rolling window stats, and technical indicators (e.g., RSI, MACD, Bollinger Bands).
- Evaluation metrics include:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score

- Model Performance Metrics

    | Model  | MAE   | RMSE  | R² Score |
    |--------|-------|-------|----------|
    | LR  | *19.25* | *22.43* | *-0.30* |
    | SVR  | *27.82* | *34.50* | *-2.08* |
    | RF  | *9.04* | *11.88* | *0.63* |
    | GB  | *8.72* | *11.40* | *0.66* |

- **Visualizations**:
  - Actual vs Predicted Prices (line plot)
  - Residual Plot (errors)
  - MAE & RMSE comparison bar charts

---

### Time Series Modeling (ARIMA, SARIMA, GARCH)

This section compares three powerful time series models:

- **ARIMA**: Captures trend using autoregressive and moving average components.
- **SARIMA**: Extends ARIMA by modeling seasonality.
- **GARCH**: Models time-varying volatility (useful for financial series).

#### Model Performance Metrics

| Model  | MAE   | RMSE  |
|--------|-------|-------|
| ARIMA  | *6.134887* | *15.929801* |
| SARIMA | *19.205966* | *21.711764* |

- **MAE (Mean Absolute Error)**: Measures average absolute errors.
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more.

#### Key Takeaways

- **ARIMA** works well for capturing trend but may struggle with seasonality.
- **SARIMA** provides improved results when seasonality is present.
- **GARCH** is useful to understand and forecast volatility (especially useful in financial data like stock prices).

---

### CNN-Based Model

Use deep learning (CNN) to model patterns in stock price sequences and predict future values with better local feature extraction than traditional models.

| Step | Description |
|------|-------------|
| Scaling | Applies `MinMaxScaler` to normalize prices between 0 and 1. |
| Sequence Generation | Converts time series into sequences using sliding windows. |
| CNN Architecture | 1D Convolution + MaxPooling + Dense layers. |
| Training | Compiled with `adam` optimizer and `mse` loss. |
| Evaluation | MAE, RMSE, and future price predictions plotted. |

#### Performance Summary

| Metric | Value |
|--------|-------|
| MAE    | *9.66* |
| RMSE   | *11.93* |

---

### LSTM-Based Model

Leverage LSTM (a variant of RNN) for time series forecasting of stock prices using historical closing data. LSTMs are well-suited for sequential data due to their ability to preserve long-term memory and overcome the vanishing gradient problem in vanilla RNNs.
- `Close/Last`: Normalized closing price.
- `Target_Close_Next_Day`: Target value to predict (next day’s closing price).

- **LSTM Architecture**:
  - Contains memory cells with gates (input, forget, and output).
  - Capable of learning both short-term and long-term temporal patterns.
- **Sliding Window**: We use 60-day historical windows to predict the next day's price.
- **EarlyStopping**: To avoid overfitting (patience = 10)

#### Evaluation Metrics (on Inverse Scaled Real Prices)

| Metric | Value |
|--------|-------|
| MAE    | *8.3355* |
| RMSE   | *10.2783* |

---

### Hybrid CNN-LSTM Model

Combines 1D Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) layers to model both local temporal patterns (via CNN) and long-term dependencies (via LSTM) in stock price data.

This hybrid approach captures short-term market fluctuations (via convolution) and sequential trends (via recurrence) more effectively than using either architecture alone.
- `Sliding Window`: 60-day lookback window for sequence construction.
- `Regularization`: Dropout and EarlyStopping (patience = 10) to mitigate overfitting.

#### Evaluation Metrics (on Inverse Scaled Real Prices)

| Metric | Value |
|--------|-------|
| MAE    | *5.53* |
| RMSE   | *6.94* |

---
