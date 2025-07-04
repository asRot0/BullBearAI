import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

class SVMModel:
    def __init__(self, kernel='rbf', C=100, gamma=0.1, epsilon=0.1):
        """
        Initialize the Support Vector Regression (SVR) model.

        Parameters:
        - kernel (str): Kernel type - 'linear', 'poly', 'rbf', or 'sigmoid'.
        - C (float): Regularization parameter.
        - gamma (float): Kernel coefficient.
        - epsilon (float): Epsilon-tube within which no penalty is associated.
        """
        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)

    def train(self, X_train, y_train):
        """
        Train the Support Vector Regression (SVR) model.

        Parameters:
        - X_train: np.ndarray or pd.DataFrame
        - y_train: np.ndarray or pd.Series
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict using the trained SVR model.

        Parameters:
        - X_test: np.ndarray or pd.DataFrame

        Returns:
        - np.ndarray of predictions
        """
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model using MAE and RMSE.

        Parameters:
        - y_true: Ground truth target values
        - y_pred: Predicted values

        Returns:
        - dict with MAE and RMSE
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {"MAE": mae, "RMSE": rmse}

    def save_model(self, path):
        """
        Save the trained SVR model to a file.

        Parameters:
        - path: File path to save the model
        """
        joblib.dump(self.model, path)

    def load_model(self, path):
        """
        Load a trained SVR model from a file.

        Parameters:
        - path: File path to load the model from
        """
        self.model = joblib.load(path)
