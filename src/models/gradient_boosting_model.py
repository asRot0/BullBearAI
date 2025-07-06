import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

class GradientBoostingModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        """
        Initialize the Gradient Boosting Regressor model.

        Parameters:
        - n_estimators: The number of boosting stages
        - learning_rate: Learning rate shrinks the contribution of each tree
        - max_depth: Maximum depth of individual regression estimators
        - random_state: For reproducibility
        """
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X_train, y_train):
        """
        Train the Gradient Boosting Regressor model.

        Parameters:
        - X_train: np.ndarray or pd.DataFrame
        - y_train: np.ndarray or pd.Series
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict using the trained Gradient Boosting model.

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
