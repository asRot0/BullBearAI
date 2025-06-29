"""
ARIMA Modeling Script
---------------------
Defines functions for training and forecasting stock prices using the ARIMA model.
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def train_arima_model(series, order=(5, 1, 0)):
    """
    Train ARIMA model on a time series.

    Args:
        series (pd.Series): Univariate time series (e.g., closing price).
        order (tuple): ARIMA (p, d, q) order.

    Returns:
        model_fit: Fitted ARIMA model object.
    """
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit


def forecast_arima(model_fit, steps):
    """
    Forecast future values using trained ARIMA model.

    Args:
        model_fit: Fitted ARIMA model.
        steps (int): Number of future steps to forecast.

    Returns:
        forecast (np.ndarray): Forecasted values.
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast


def evaluate_arima(y_true, y_pred):
    """
    Evaluate ARIMA forecast.

    Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Forecasted values.

    Returns:
        dict: MAE and RMSE scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4)
    }
