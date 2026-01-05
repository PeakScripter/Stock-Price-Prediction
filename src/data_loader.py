import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .features import add_technical_indicators
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def download_data(ticker, start_date, end_date):
    """
    Download stock data using yfinance.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    df = df[required_cols] # Keep only standard OHLCV for clarity, indicators will add more.
    return df

def create_dataset(data, time_step=60, target_col='Close'):
    """
    Create input (X) and output (y) for LSTM models.
    data should be a DataFrame or 2D array where the target column is included.
    We assume the data is already scaled.
    """
    X, y = [], []
    # If data is a DataFrame, convert to numpy
    if isinstance(data, pd.DataFrame):
        dataset = data.values
    else:
        dataset = data

    # Find index of target column
    if isinstance(data, pd.DataFrame):
         target_idx = data.columns.get_loc(target_col)
    else:
         # Assume target is the first column if numpy array and not specified otherwise
         # This part needs care. For now, let's strictly handle DataFrame validation in calling function
         # Or assume 'Close' is the target and we passed a DataFrame.
         # For simplicity in this function, let's assume 'dataset' includes all features
         # AND we want to predict the column at 'target_idx'.
         # If scalar is passed as array, target_idx=0
         target_idx = 0 

    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i : (i + time_step), :]) # All features
        y.append(dataset[i + time_step, target_idx]) # Target value
        
    return np.array(X), np.array(y)


def add_log_returns(df):
    """
    Add Log Return column.
    """
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    return df

from sklearn.preprocessing import StandardScaler

def get_processed_data(config_path="config.yaml"):
    """
    Main function to load, process and return train/test data.
    """
    config = load_config(config_path)
    
    # 1. Download
    df = download_data(config['data']['ticker'], config['data']['start_date'], config['data']['end_date'])
    
    # 2. Feature Engineering
    if config['feature_engineering']['use_technical_indicators']:
        df = add_technical_indicators(df)
    
    # Add Log Returns (Target)
    df = add_log_returns(df)
    
    # Drop NaNs created by indicators/returns
    df.dropna(inplace=True)
    
    # 3. Feature Selection (Crucial for Stationarity)
    # We typically want to Drop Raw Prices (OHLC).
    # We keep Indicators + Log_Return.
    # We might keep Volume if scaled, but often Volume is also non-stationary. Let's keep Volume for now but be careful.
    # Actually, let's strictly use mostly stationary features.
    
    # Identify feature columns
    # We want to exclude raw OHLC.
    exclude_cols = ['Open', 'High', 'Low', 'Close'] 
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Reorder to ensure Log_Return is last (or first) for easier targeting? 
    # Let's put Log_Return as the LAST column for convention.
    if 'Log_Return' in feature_cols:
        feature_cols.remove('Log_Return')
    feature_cols.append('Log_Return')
    
    data_df = df[feature_cols]
    
    # 4. Scaling
    train_size = int(len(data_df) * config['data']['test_split'])
    train_df = data_df.iloc[:train_size]
    test_df = data_df.iloc[train_size:]
    
    # Use StandardScaler instead of MinMaxScaler for Log Returns
    # Returns are centered around 0. MinMax forces them to [0,1] which biases "0" to e.g. 0.6
    scaler = StandardScaler()
    
    # Fit on ALL data (as per previous fix for range issues, though returns are usually well bounded 0-1 isn't ideal for returns centered at 0)
    # Returns are usually -0.1 to 0.1. MinMax (0,1) works.
    scaler.fit(data_df)
    
    scaled_train = scaler.transform(train_df)
    scaled_test = scaler.transform(test_df)
    
    # Target is Log_Return, which is the LAST column
    target_col_idx = len(feature_cols) - 1
    
    time_step = config['data']['time_step']
    
    X_train, y_train = create_dataset_from_array(scaled_train, time_step, target_col_idx)
    X_test, y_test = create_dataset_from_array(scaled_test, time_step, target_col_idx)
    
    # Return df (original full df with OHLC for reconstruction) and feature_cols for tracking
    return X_train, y_train, X_test, y_test, scaler, df, feature_cols

def create_dataset_from_array(dataset, time_step, target_idx):
    X, y = [], []
    for i in range(len(dataset) - time_step): # Changed logic slightly to match standard windowing
        X.append(dataset[i : (i + time_step), :])
        y.append(dataset[i + time_step, target_idx])
    return np.array(X), np.array(y)
