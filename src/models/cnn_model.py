import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

class CNNModel:
    def __init__(self, input_shape, filters=64, kernel_size=3, dense_units=64, dropout_rate=0.3):
        """
        Initialize the CNN model.

        Parameters:
        - input_shape: Shape of the input data (timesteps, features)
        - filters: Number of filters for Conv1D
        - kernel_size: Kernel size for Conv1D
        - dense_units: Number of neurons in dense layer
        - dropout_rate: Dropout rate after dense layer
        """
        self.input_shape = input_shape
        self.model = self._build_model(filters, kernel_size, dense_units, dropout_rate)

    def _build_model(self, filters, kernet_size, dense_units, dropout_rate):
        model = Sequential()
        model.add(Conv1D(filters=filters, kernet_size=kernet_size, activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, patience=10):
        """
        Train the CNN model with early stopping.

        Returns:
        - training history object
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