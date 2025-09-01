import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

class HybridCNNLSTMModel:
    def __init__(self, input_shape,
                 conv_filters=64, kernel_size=3, pool_size=2,
                 lstm_units=64, dense_units=32, dropout_rate=0.2):
        """
        Hybrid CNN + LSTM model for stock price prediction.

        Parameters:
        - input_shape: tuple, (timesteps, features)
        - conv_filters: number of convolution filters
        - kernel_size: size of CNN kernel
        - pool_size: size of max pooling window
        - lstm_units: number of LSTM units
        - dense_units: number of neurons in dense layer
        - dropout_rate: dropout rate to prevent overfitting
        """
        self.input_shape = input_shape
        self.model = self._build_model(conv_filters, kernel_size, pool_size,
                                       lstm_units, dense_units, dropout_rate)

    def _build_model(self, conv_filters, kernel_size, pool_size, lstm_units, dense_units, dropout_rate):
        model = Sequential()
        # CNN Block
        model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', input_shape=self.input_shape))
        # model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Dropout(dropout_rate))

        # LSTM Block
        model.add(LSTM(units=lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))

        # Dense Block
        # model.add(Dense(units=dense_units, activation='relu'))
        # model.add(Dropout(dropout_rate))
        model.add(Dense(1))  # Regression output (stock price)

        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, patience=40):
        """
        Train the Hybrid CNN-LSTM model.
        """
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stop],
            verbose=1
        )
        return history

    def predict(self, X):
        """
        Predict using the trained Hybrid model.
        """
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred, scaler, scaler_shape):
        """
        Evaluate model with MAE and RMSE.
        """
        self.scaler = scaler

        y_test_reshaped = y_true.reshape(-1, 1)
        y_pred_reshaped = y_pred.reshape(-1, 1)

        y_pred_dummy = np.zeros((len(y_pred), scaler_shape))
        y_test_dummy = np.zeros((len(y_true), scaler_shape))
        y_pred_dummy[:, -1] = y_pred_reshaped[:, 0]
        y_test_dummy[:, -1] = y_test_reshaped[:, 0]

        # Inverse transform only the target
        y_pred_inv = scaler.inverse_transform(y_pred_dummy)[:, -1]
        y_test_inv = scaler.inverse_transform(y_test_dummy)[:, -1]

        # Evaluation Metrics
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

        # mae = mean_absolute_error(y_true, y_pred)
        # rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {"MAE": mae, "RMSE": rmse}

    def save_model(self, path):
        """
        Save trained model to disk.
        """
        self.model.save(path)

    def load_model(self, path):
        """
        Load trained model from disk.
        """
        self.model = load_model(path)
