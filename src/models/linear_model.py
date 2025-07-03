import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        """
        Train the Linear Regression model.

        Parameters:
        - X_train: np.ndarray or pd.DataFrame
        - y_train: np.ndarray or pd.Series
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict using the trained Linear Regression model.

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
        Save the trained model to a file.

        Parameters:
        - path: File path to save the model
        """
        joblib.dump(self.model, path)

    def load_model(self, path):
        """
        Load a trained model from a file.

        Parameters:
        - path: File path to load the model from
        """
        self.model = joblib.load(path)
