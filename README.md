
## Project Progress Overview: BullBearAI

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
│   ├── 04_time_series_models.ipynb         # Time series statistical models: ARIMA, SARIMA, GARCH, etc.
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

---

### Data Cleaning
- Removed duplicates and handled any missing/null values.
- Renamed columns for consistency and usability (`Close/Last` instead of `Close*`).
- Converted all date fields to `datetime` format.
- Ensured data is sorted chronologically.
- Exported cleaned dataset to `data/processed/`.

---

### Exploratory Data Analysis (EDA)
- Visualized time-series trends of `Close`, `Volume`, and `Open`.
- Used Seaborn and Matplotlib for:
  - Moving averages
  - Seasonal decomposition
  - Daily/Monthly return distributions
- Checked for trends, volatility, and patterns.
- Identified data gaps, outliers, or anomalies.
- All EDA work is saved in `notebooks/01_eda.ipynb`.

---

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

📁 Engineered dataset saved to: `data/interim/engineered_features.csv`.

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
- **Visualizations**:
  - Actual vs Predicted Prices (line plot)
  - Residual Plot (errors)
  - MAE & RMSE comparison bar charts

---

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![License](https://img.shields.io/github/license/your-username/BullBearAI)
![Status](https://img.shields.io/badge/Progress-Phase%201%20✅-green)
