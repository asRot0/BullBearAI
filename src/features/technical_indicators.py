"""
Technical Indicators Module for BullBearAI
------------------------------------------
This module provides reusable functions to compute popular technical indicators
used in financial time series analysis.
"""

import pandas as pd
import numpy as np


def calculate_sma(df, window=14, price_col='Close/Last'):
    """Simple Moving Average (SMA)"""
    return df[price_col].rolling(window=window).mean()


def calculate_ema(df, span=14, price_col='Close/Last'):
    """Exponential Moving Average (EMA)"""
    return df[price_col].ewm(span=span, adjust=False).mean()


def calculate_rsi(df, window=14, price_col='Close/Last'):
    """Relative Strength Index (RSI)"""
    delta = df[price_col].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=df.index)


def calculate_macd(df, price_col='Close/Last', fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence)"""
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_bollinger_bands(df, window=20, num_std=2, price_col='Close/Last'):
    """Bollinger Bands"""
    sma = df[price_col].rolling(window=window).mean()
    std = df[price_col].rolling(window=window).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return upper_band, lower_band


def calculate_atr(df, window=14):
    """Average True Range (ATR)"""
    high = df['High']
    low = df['Low']
    close = df['Close/Last']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr


def add_all_indicators(df):
    """
    Apply all supported technical indicators to the dataframe.

    Returns:
        DataFrame with new columns: SMA, EMA, RSI, MACD, MACD_Signal,
        Bollinger_Upper, Bollinger_Lower, ATR
    """
    df = df.copy()
    df['SMA_14'] = calculate_sma(df)
    df['EMA_14'] = calculate_ema(df)
    df['RSI_14'] = calculate_rsi(df)
    df['MACD'], df['MACD_Signal'] = calculate_macd(df)
    df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df)
    df['ATR_14'] = calculate_atr(df)
    return df

