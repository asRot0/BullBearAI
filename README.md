
## 🚀 Project Progress Overview: BullBearAI

This project is designed to predict stock market trends using traditional ML, deep learning, and a hybrid LSTM-CNN architecture. Below is the step-by-step progress with brief descriptions.

```
BullBearAI/
│
├── data/                        # Raw and processed stock data
│   ├── raw/                     # Untouched downloaded data
│   ├── interim/                 # Intermediate transformation outputs
│   └── processed/               # Cleaned and final datasets
│
├── notebooks/                  # Jupyter notebooks for EDA, prototyping
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_hybrid_model.ipynb
│   └── 05_evaluation.ipynb
│
├── src/                        # All source code
│   ├── config/                 # Configuration files and parameters
│   │   └── config.yaml
│   ├── data_loader/            # Data loading and preprocessing scripts
│   │   └── load_data.py
│   ├── features/               # Feature engineering functions
│   │   └── technical_indicators.py
│   ├── models/                 # ML & DL models
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


### 📥 1. Data Loading & Initial Inspection
- Loaded the raw stock market data (Netflix stock) from the `data/raw/` directory.
- Verified file integrity, parsed dates correctly, and ensured data types were appropriate.
- Saved a clean version in `data/processed/netflix_cleaned.csv`.

---

### 🧹 2. Data Cleaning
- Removed duplicates and handled any missing/null values.
- Renamed columns for consistency and usability (`Close/Last` instead of `Close*`).
- Converted all date fields to `datetime` format.
- Ensured data is sorted chronologically.
- Exported cleaned dataset to `data/processed/`.

---

### 📊 3. Exploratory Data Analysis (EDA)
- Visualized time-series trends of `Close`, `Volume`, and `Open`.
- Used Seaborn and Matplotlib for:
  - Moving averages
  - Seasonal decomposition
  - Daily/Monthly return distributions
- Checked for trends, volatility, and patterns.
- Identified data gaps, outliers, or anomalies.
- All EDA work is saved in `notebooks/01_eda.ipynb`.

---

### 🧠 4. Feature Engineering
Performed a comprehensive set of transformations to prepare predictive features:

#### 📅 Date-Based Features
- Extracted: `Year`, `Month`, `Day`, `DayOfWeek`, and `IsWeekend`.
#### 🔁 Lag Features
- Created lagged versions of `Close/Last` and `Volume` (lags: 1, 2, 3 days).
#### 🔄 Rolling Statistics
- Computed rolling means, medians, stds, max, min for 7, 14, and 30-day windows.
#### 📈 Volatility Measures
- Daily percentage change, return, and rolling return metrics.
#### 📊 Technical Indicators
- Simple & Exponential Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
#### 🎯 Target Variable
- `Target_Close_Next_Day`: Next day’s close price
- `Target_UpDown`: Binary classification target (1 = price goes up, 0 = down)

📁 Engineered dataset saved to: `data/interim/engineered_features.csv`.

---
