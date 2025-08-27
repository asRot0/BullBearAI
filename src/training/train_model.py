import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import ML models
from src.models.linear_model import LinearRegressionModel
from src.models.svm_model import SVMModel
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel

# Import DL models
from src.models.cnn_model import CNNModel
from src.models.lstm_model import LSTMModel
from src.models.hybrid_model import HybridCNNLSTMModel

class ModelTrainer:
    def __init__(self, model_name, feature_cols, target_col, test_size=0.2, val_size=0.1, random_state=42):
        """
        Central training pipeline for ML/DL models.

        Parameters:
        - model_name: str, one of ['linear', 'svm', 'random_forest', 'gradient_boosting', 'cnn', 'lstm', 'hybrid']
        - feature_cols: list of features to use
        - target_col: target column name
        - test_size: fraction of test data
        - val_size: fraction of validation data (for DL models)
        """
        self.model_name = model_name
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.scaler = MinMaxScaler()
        self.model = None

    def prepare_data(self, df, sequence_length=60):
        """
        Prepares data for training.

        - For ML models: simple scaled features
        - For DL models: create sequences (X shape: [samples, timesteps, features])
        """
        features = df[self.feature_cols].values
        target = df[self.target_col].values

        # Scale features
        features_scaled = self.scaler.fit_transform(features.reshape(-1, len(self.feature_cols)))

        if self.model_name in ["cnn", "lstm", "hybrid"]:
            X, y = [], []
            for i in range(sequence_length, len(features_scaled)):
                X.append(features_scaled[i-sequence_length:i])
                y.append(target[i])
            X, y = np.array(X), np.array(y)

            # Split train/val/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            val_size_adj = int(len(X_train) * self.val_size)
            X_val, y_val = X_train[-val_size_adj:], y_train[-val_size_adj:]
            X_train, y_train = X_train[:-val_size_adj], y_train[:-val_size_adj]
            return X_train, X_val, X_test, y_train, y_val, y_test

        else:
            # For ML models
            X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=self.test_size, random_state=self.random_state)
            return X_train, X_test, y_train, y_test

    def initialize_model(self, input_shape=None):
        """
        Initialize the correct model based on model_name.
        """
        if self.model_name == "linear":
            self.model = LinearRegressionModel()
        elif self.model_name == "svm":
            self.model = SVMModel()
        elif self.model_name == "random_forest":
            self.model = RandomForestModel()
        elif self.model_name == "gradient_boosting":
            self.model = GradientBoostingModel()
        elif self.model_name == "cnn":
            self.model = CNNModel(input_shape)
        elif self.model_name == "lstm":
            self.model = LSTMModel(input_shape)
        elif self.model_name == "hybrid":
            self.model = HybridCNNLSTMModel(input_shape)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def train_and_evaluate(self, df):
        """
        Complete pipeline: prepare data, train, predict, evaluate.
        """
        if self.model_name in ["cnn", "lstm", "hybrid"]:
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df)
            self.initialize_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            history = self.model.train(X_train, y_train, X_val, y_val)
            y_pred = self.model.predict(X_test)
        else:
            X_train, X_test, y_train, y_test = self.prepare_data(df)
            self.initialize_model()
            self.model.train(X_train, y_train)
            y_pred = self.model.predict(X_test)

        results = self.model.evaluate(y_test, y_pred)
        return results
