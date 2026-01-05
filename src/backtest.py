import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE handling division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

def backtest_strategy(price_data, predicted_prices, threshold=0.0):
    """
    Simulate a simple trading strategy:
    Buy if predicted price > current price + threshold
    Sell if predicted price < current price - threshold
    
    Args:
        price_data (np.array): Actual prices.
        predicted_prices (np.array): Predicted prices values for the NEXT day/step.
        threshold (float): Threshold to trigger trade.
        
    Returns:
        dict: Backtest results.
    """
    # Align data: predicted_prices[i] is prediction for time i+1?
    # Usually in time series X[i] -> y[i] where y[i] is price at t+1.
    # So if we have predictions y_pred, y_pred[i] corresponds to price at t+1 (relative to input window end).
    # price_data should be the actual prices corresponding to y_pred (y_true).
    
    # We assume trade logic:
    # At time t, we have current price P_t.
    # We predict P_{t+1}.
    # If P_{t+1} > P_t, Buy.
    
    # Since y_true and y_pred are aligned:
    # y_true[i] is price at t+1.
    # We need price at t.
    # This might require passing the original series shifted.
    # However, let's simplify. We compute returns based on the direction of prediction vs "previous" known price.
    
    # If we only have y_true (future prices) and y_pred, we need the "current" price which is y_true[i-1] ?
    # Let's assume price_data passed here is y_true.
    
    # Correct approach with just arrays:
    # We iterate. For step i, we look at prediction[i].
    # Prediction[i] basically says what the price will be.
    # We compare Prediction[i] with Price[i-1] (current known price).
    
    initial_balance = 10000.0
    balance = initial_balance
    position = 0 # 0: flat, 1: long
    holdings = 0.0
    
    # Returns relative to buy and hold?
    
    # Let's construct a dataframe for easier calc if indices matched, but we use arrays.
    
    signals = []
    
    # Since we need previous price, we start from i=1
    for i in range(1, len(price_data)):
        current_price = price_data[i-1]
        predicted_future_price = predicted_prices[i] # This aligns with price_data[i]
        actual_future_price = price_data[i]
        
        # Strategy
        if predicted_future_price > current_price * (1 + threshold):
            # Buy signal
            if position == 0:
                holdings = balance / current_price
                balance = 0
                position = 1
                signals.append('Buy')
            else:
                signals.append('Hold')
                
        elif predicted_future_price < current_price * (1 - threshold):
            # Sell signal
            if position == 1:
                balance = holdings * current_price
                holdings = 0
                position = 0
                signals.append('Sell')
            else:
                signals.append('Wait')
        else:
            signals.append('Hold/Wait')
            
    # Final value
    final_price = price_data[-1]
    final_value = balance + (holdings * final_price)
    
    total_return = (final_value - initial_balance) / initial_balance * 100
    
    # Calculate Sharpe Ratio (simplified, assuming daily steps)
    # We need daily portfolio values to compute sharpe
    
    return {
        "Initial Balance": initial_balance,
        "Final Value": final_value,
        "Total Return (%)": total_return
    }

def walk_forward_validation(model_fn, X, y, test_size=0.2, window_size=5):
    """
    Perform walk-forward validation (Rolling Window).
    This is computationally expensive as it retrains/updates the model repeatedly.
    For this project, we might simulate it or just do a simple implementation if requested.
    
    Args:
        model_fn: Function to create/compile model.
        X, y: Full dataset.
        
    Returns:
        list: Predictions.
    """
    # Placeholder for advanced implementation.
    # In deep learning, full retraining every step is too slow.
    pass
