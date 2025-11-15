
# 📊 BullBearAI Final Model Evaluation

## 🏁 Model Performance Summary

| Model             | MAE   | RMSE  |
| ----------------- | ----- | ----- |
| Linear Regression | 19.25 | 22.43 |
| SVM               | 27.82 | 34.50 |
| Random Forest     | 9.04  | 11.88 |
| Gradient Boosting | 8.72  | 11.40 |
| ARIMA             | 6.13  | 15.92 |
| SARIMA            | 19.21 | 21.71 |
| CNN               | 9.14  | 11.24 |
| LSTM              | 8.67  | 10.50 |
| Hybrid CNN-LSTM   | 4.13  | 5.15  |


## 🥇 Best Model: **Hybrid CNN-LSTM**
- MAE: 5.5300
- RMSE: 6.9400

## 📈 Prediction Comparison
![Model Predictions](figures/predictions_comparison.png)

## 📊 Metric Comparison
![Bar Chart](figures/model_performance.png)

---
Generated from `08_model_comparison.ipynb`


| Model             | MAE   | RMSE  | R²    |
| ----------------- | ----- | ----- | ----- |
| ARIMA             | 6.13  | 15.92 | 0.67  |
| SARIMA            | 19.21 | 21.71 | -3.32 |
| Linear Regression | 19.25 | 22.43 | -0.30 |
| SVM               | 27.82 | 34.50 | -2.08 |
| Random Forest     | 9.04  | 11.88 | 0.63  |
| Gradient Boosting | 8.72  | 11.40 | 0.66  |
| CNN               | 9.14  | 11.24 | 0.67  |
| LSTM              | 8.67  | 10.50 | 0.37  |
| **Hybrid CNN-LSTM** | **4.13** | **5.15** | **0.93** |


| Model             | Directional Accuracy (%) |
| ----------------- | ------------------------ |
| ARIMA             | 47.73                    |
| SARIMA            | 41.38                    |
| Linear Regression | 77.27                    |
| SVM               | 52.27                    |
| Random Forest     | 40.91                    |
| Gradient Boosting | 63.64                    |
| CNN               | 45.24                    |
| LSTM              | 46.88                    |
| **Hybrid CNN-LSTM**| **76.74**                |



| Model             | Cumulative Return (%) | Sharpe Ratio | Max Drawdown (%) |
| ----------------- | --------------------- | ------------ | ---------------- |
| ARIMA             | inf                   | NaN          | NaN              |
| SARIMA            | 1.32                  | 2.13         | -0.60            |
| Linear Regression | -12.69                | -1.45        | -24.57           |
| SVM               | 3.48                  | 0.79         | -19.13           |
| Random Forest     | -3.42                 | 0.03         | -24.12           |
| Gradient Boosting | -6.72                 | -0.39        | -22.27           |
| CNN               | -10.40                | -2.29        | -23.28           |
| LSTM              | -10.24                | -2.53        | -18.02           |
| **Hybrid CNN-LSTM**| **-8.44**             | **-1.23**    | **-23.10**       |
