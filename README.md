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