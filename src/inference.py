import os
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from .features import add_technical_indicators

def save_model(model, scaler, path="models/model.h5"):
    """
    Save the model and scaler.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    
    # Save scaler with a modified extension
    scaler_path = path.replace(".h5", "").replace(".keras", "") + "_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {path}")
    print(f"Scaler saved to {scaler_path}")

def load_saved_model(path="models/model.h5"):
    """
    Load a model and its scaler.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    
    model = tf.keras.models.load_model(path)
    
    scaler_path = path.replace(".h5", "").replace(".keras", "") + "_scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    else:
        scaler = None
        print("No associated scaler found. Ensure manual scaling matches training.")
        
    return model, scaler

def forecast_future_robust(model, last_sequence, n_steps, scaler, feature_cols, last_raw_df, target_idx=-1, noise_level=0.0):
    """
    Robust forecasting using Log Returns with optional Monte Carlo noise.
    """
    current_sequence = last_sequence.copy()
    pred_log_returns = []
    
    # Make a working copy of the dataframe tail to append future rows
    history_df = last_raw_df.copy()
    
    if target_idx == -1:
        target_idx = len(feature_cols) - 1
        
    for _ in range(n_steps):
        # 1. Predict Next Log Return
        input_seq = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        pred_scaled_ret = model.predict(input_seq, verbose=0)[0, 0]
        
        # Inverse Transform
        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, target_idx] = pred_scaled_ret
        pred_ret = scaler.inverse_transform(dummy)[0, target_idx]
        
        # Add Monte Carlo Noise (if enabled)
        # Noise is added to the "True Return" space.
        # Ideally, noise_level is the Std Dev of the residual errors (RMSE).
        if noise_level > 0:
            noise = np.random.normal(0, noise_level)
            pred_ret += noise
        
        pred_log_returns.append(pred_ret)
        
        # 2. Reconstruct New Price
        last_close = history_df['Close'].iloc[-1]
        new_price = last_close * np.exp(pred_ret)
        
        # 3. Create New Raw Row
        # Assume O=H=L=C = new_price (Simplified)
        # Volume = last volume (naive)
        new_row = {
            'Open': new_price,
            'High': new_price,
            'Low': new_price,
            'Close': new_price,
            'Volume': history_df['Volume'].iloc[-1]
        }
        
        # Append to history
        # Create a DF for the single row
        new_row_df = pd.DataFrame([new_row])
        # We assume specific columns exist.
        
        # We append to history_df. 
        # Note: concat is slow in loop but n_steps is small (30-90).
        history_df = pd.concat([history_df, new_row_df], ignore_index=True)
        
        # 4. Recalculate Indicators on EXTENDED history
        # Only need the tail relevant for window (e.g. last 100)
        calc_df = history_df.tail(100).copy()
        
        # Add indicators (this adds cols RSI, MACD etc)
        # We also need 'Log_Return' added because it's a feature!
        
        calc_df = add_technical_indicators(calc_df)
        calc_df['Log_Return'] = np.log(calc_df['Close'] / calc_df['Close'].shift(1))
        
        # Now we have the new feature row (the last row of calc_df)
        # We need to extract ONLY the `feature_cols` relevant for the model
        # Fill NA if any (first row of return is NaN, but we are at tail so it's fine)
        calc_df = calc_df.fillna(0)
        
        new_features_raw = calc_df.iloc[-1][feature_cols].values.reshape(1, -1)
        
        # 5. Scale the new feature row
        new_features_scaled = scaler.transform(new_features_raw)
        
        # 6. Update Sequence
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = new_features_scaled[0]
        
    return pred_log_returns
