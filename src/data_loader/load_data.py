import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_clean_stock_data(file_path: str, save_cleaned_path: str = None) -> pd.DataFrame:
    """
    Steps:
    - Convert 'Date' to datetime
    - Remove '$' signs from price columns
    - Remove commas from 'Volume'
    - Convert all columns to numeric where appropriate
    - Handle missing values

    Parameters:
    - file_path: str - path to raw CSV file
    - save_cleaned_path: str, optional - path to save the cleaned CSV. If None, not saved

    Returns:
    - Cleaned DataFrame
    """
    # Load data
    stock_data = pd.read_csv(file_path)
    print(stock_data.shape)
    print(stock_data.head())

    # Convert the 'Date' column to datetime
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Columns to clean (dollar sign removal)
    dollar_cols = ['Close/Last', 'Open', 'High', 'Low']
    for column in dollar_cols:
        if column in stock_data.columns:
            stock_data[column] = stock_data[column].replace({'\$': '', ',': ''}, regex=True).astype(float)

    # Clean 'Volume' column (remove commas and convert to float)
    if 'Volume' in stock_data.columns:
        stock_data['Volume'] = stock_data['Volume'].replace({',': ''}, regex=True).astype(float)

    # Handle missing values (drop rows with NaNs)
    stock_data.dropna(inplace=True)

    # Save to processed folder if path is given
    if save_cleaned_path:
        stock_data.to_csv(save_cleaned_path, index=False)

    return stock_data


def load_stock_data(input_path, sequence_length=60, feature_col='Close/Last', target_col='Target_Close_Next_Day'):
    """
    Load stock data and generate sequences for time series forecasting.

    Parameters:
    - input_path (str): Path to the processed CSV file.
    - sequence_length (int): Number of past days to use for each input sample.
    - feature_col (str or list): Single column or list of feature columns to use as input.
    - target_col (str): Column to be predicted.

    Returns:
    - X (np.array): Input sequences of shape (samples, sequence_length, n_features)
    - y (np.array): Target values of shape (samples,)
    - scaler (MinMaxScaler): Fitted scaler for inverse transforming
    """
    df = pd.read_csv(input_path)

    # Ensure feature_col is a list
    if isinstance(feature_col, str):
        feature_col = [feature_col]

    # Select and scale features
    feature_data = df[feature_col].values
    scaler = MinMaxScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)

    target_data = df[target_col].values

    X, y = [], []
    for i in range(sequence_length, len(feature_data_scaled)):
        X.append(feature_data_scaled[i-sequence_length:i])
        y.append(target_data[i])

    return np.array(X), np.array(y), scaler
