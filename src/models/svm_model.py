"""
Support Vector Machine (SVM) Regression Model
---------------------------------------------
Defines training and evaluation utilities for applying SVM to stock price prediction.
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_svm_model(X_train, y_train, kernel="rbf", C=100, gamma=0.1, epsilon=0.1):
    """
    Train a Support Vector Regression (SVR) model.

    Args:
        X_train (np.ndarray): Feature matrix for training.
        y_train (np.ndarray): Target values for training.
        kernel (str): Kernel type - 'linear', 'poly', 'rbf', or 'sigmoid'.
        C (float): Regularization parameter.
        gamma (float): Kernel coefficient.
        epsilon (float): Epsilon-tube within which no penalty is associated.

    Returns:
        model (SVR): Trained SVR model.
    """
    model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_train, y_train.ravel())
    return model


def predict_svm(model, X_test):
    """
    Generate predictions using a trained SVR model.

    Args:
        model (SVR): Trained SVR model.
        X_test (np.ndarray): Feature matrix for testing.

    Returns:
        np.ndarray: Predicted target values.
    """
    return model.predict(X_test)


def evaluate_svm(y_true, y_pred):
    """
    Evaluate SVM model performance.

    Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.

    Returns:
        dict: MAE and RMSE scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4)
    }
