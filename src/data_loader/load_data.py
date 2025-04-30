import pandas as pd


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