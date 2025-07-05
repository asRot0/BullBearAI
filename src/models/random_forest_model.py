import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize the Random Forest Regressor.

        Parameters:
        - n_estimators: Number of trees in the forest
        - max_depth: Maximum depth of the tree
        - random_state: Seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X_train, y_train):
        """
        Train the Random Forest model.

        Parameters:
        - X_train: np.ndarray or pd.DataFrame
        - y_train: np.ndarray or pd.Series
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict using the trained Random Forest model.

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
