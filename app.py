import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
from src.data_loader import get_processed_data, download_data, add_technical_indicators, create_dataset_from_array
from src.model import create_model
from src.backtest import calculate_metrics, backtest_strategy
from src.visualization import plot_interactive_predictions, plot_candlestick
from src.inference import save_model, load_saved_model, forecast_future_robust
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("Stock Price Prediction Dashboard")

# Sidebar Configuration
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "^NSEI")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2008-01-01")) # Default to longer history
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

model_source = st.sidebar.radio("Model Source", ["Train New Model", "Load Pre-trained Model"])

if model_source == "Train New Model":
    model_type = st.sidebar.selectbox("Model Type", ["lstm_cnn", "bidirectional_lstm", "attention_lstm"])
    epochs = st.sidebar.slider("Epochs", 1, 100, 50)
    batch_size = st.sidebar.slider("Batch Size", 16, 128, 64)
else:
    # List available models
    if not os.path.exists("models"):
        os.makedirs("models")
    model_files = [f for f in os.listdir("models") if f.endswith(".h5") or f.endswith(".keras")]
    if not model_files:
        st.sidebar.warning("No pre-trained models found in 'models/' directory.")
        selected_model_file = None
    else:
        selected_model_file = st.sidebar.selectbox("Select Model", model_files)

# Global variables for model and data
if 'model' not in st.session_state:
    st.session_state.model = None

# Load Data
@st.cache_data
def load_and_process_data(ticker, start, end):
    df = download_data(ticker, start, end)
    df = add_technical_indicators(df)
    return df

try:
    with st.spinner("Loading Data..."):
        df_raw = load_and_process_data(ticker, start_date, end_date)
        
    st.subheader(f"Raw Data: {ticker}")
    st.dataframe(df_raw.tail())
    
    # Visualization
    st.subheader("Candlestick Chart")
    st.plotly_chart(plot_candlestick(df_raw), use_container_width=True)
    
    # Logic for Training or Loading
    if model_source == "Train New Model":
        if st.button("Train Model"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Prepare Data (Log Return Logic)
            status_text.text("Preprocessing data (Log Returns)...")
            
            # Make a copy to avoid mutating cached df
            df = df_raw.copy()
            
            # 1. Add Log Returns
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            df.dropna(inplace=True)
            
            # 2. Select Features (Stationary)
            exclude_cols = ['Open', 'High', 'Low', 'Close']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
             # Ensure Log_Return is last
            if 'Log_Return' in feature_cols:
                feature_cols.remove('Log_Return')
            feature_cols.append('Log_Return')
            data_df = df[feature_cols]
            
            # 3. Scaling
            test_split = 0.8
            train_size = int(len(data_df) * test_split)
            train_df = data_df.iloc[:train_size]
            test_df = data_df.iloc[train_size:]
            
            scaler = StandardScaler()
            scaler.fit(data_df)
            
            scaled_train = scaler.transform(train_df)
            scaled_test = scaler.transform(test_df)
            
            target_idx = len(feature_cols) - 1 # Log Return is last
            time_step = 60
            
            X_train, y_train = create_dataset_from_array(scaled_train, time_step, target_idx)
            X_test, y_test = create_dataset_from_array(scaled_test, time_step, target_idx)
            
            status_text.text(f"Building {model_type} model...")
            
            config = {
                'type': model_type,
                'filters_cnn': 64,
                'kernel_size': 2,
                'pool_size': 2,
                'units_lstm': 50,
                'units_dense': 25,
                'dropout': 0.2,
                'learning_rate': 0.001
            }
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = create_model(input_shape, config)
            
            status_text.text("Training model...")
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Save Session State
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.feature_cols = feature_cols
            st.session_state.target_idx = target_idx
            st.session_state.time_step = time_step
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.last_raw_df = df_raw.tail(200) # Save tail for forecasting
            st.session_state.test_dates = test_df.index[time_step:] 
            
            progress_bar.progress(100)
            status_text.text("Training Complete!")
            
            save_name = st.text_input("Save Model As (e.g., model.h5)", "model.h5")
            if st.button("Save Model"):
                save_model(model, scaler, f"models/{save_name}")
                st.success(f"Model and Scaler saved to models/{save_name}")

    elif model_source == "Load Pre-trained Model" and selected_model_file:
         if st.button("Load Model"):
             model, loaded_scaler = load_saved_model(f"models/{selected_model_file}")
             st.session_state.model = model
             
             # Prep features again for context
             df = df_raw.copy()
             df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
             df.dropna(inplace=True)
             exclude_cols = ['Open', 'High', 'Low', 'Close']
             feature_cols = [c for c in df.columns if c not in exclude_cols]
             if 'Log_Return' in feature_cols: feature_cols.remove('Log_Return')
             feature_cols.append('Log_Return')
             data_df = df[feature_cols]
             
             if loaded_scaler:
                 scaler = loaded_scaler
                 st.info("Loaded scaler.")
             else:
                 st.warning("No saved scaler found! Fitting new scaler.")
                 scaler = StandardScaler()
                 scaler.fit(data_df)
             
             target_idx = len(feature_cols) - 1
             time_step = 60
             
             scaled_full = scaler.transform(data_df)
             X_full, y_full = create_dataset_from_array(scaled_full, time_step, target_idx)
             
             st.session_state.scaler = scaler
             st.session_state.feature_cols = feature_cols
             st.session_state.target_idx = target_idx
             st.session_state.time_step = time_step
             st.session_state.X_test = X_full[-252:]
             st.session_state.y_test = y_full[-252:]
             st.session_state.last_raw_df = df_raw.tail(200)
             
             st.success("Model loaded successfully!")

    # Evaluation
    if st.session_state.model is not None:
        model = st.session_state.model
        scaler = st.session_state.scaler
        feature_cols = st.session_state.feature_cols
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        target_idx = st.session_state.target_idx
        
        # Predict Returns
        test_predict_scaled = model.predict(X_test)
        
        # Helper to inverse transform specific column
        n_features = scaler.mean_.shape[0] if hasattr(scaler, 'mean_') else scaler.min_.shape[0]
        def inv_trans(pred, scaler, idx):
            dummy = np.zeros((len(pred), n_features))
            dummy[:, idx] = pred.flatten()
            return scaler.inverse_transform(dummy)[:, idx]
            
        y_test_ret = inv_trans(y_test, scaler, target_idx)
        test_pred_ret = inv_trans(test_predict_scaled, scaler, target_idx)
        
        # Metrics
        metrics = calculate_metrics(y_test_ret, test_pred_ret)
        st.subheader("Model Performance (Log Returns)")
        col1, col2 = st.columns(2)
        col1.metric("MAE (Return)", f"{metrics['MAE']:.6f}")
        col2.metric("RMSE (Return)", f"{metrics['RMSE']:.6f}")
        
        # Future Forecast
        st.subheader("Future Forecast (Robust)")
        forecast_days = st.slider("Days to Forecast", 1, 90, 30)
        enable_noise = st.checkbox("Enable Monte Carlo Simulation (Noise injection based on Test RMSE)")
        
        if st.button("Generate Forecast"):
            
            last_raw = st.session_state.last_raw_df
            
            # Reconstruct the scaled sequence for the very last step
            temp_df = last_raw.copy()
            temp_df['Log_Return'] = np.log(temp_df['Close'] / temp_df['Close'].shift(1))
            temp_df.dropna(inplace=True)
            temp_data = temp_df[feature_cols]
            
            scaled_tail = scaler.transform(temp_data)
            last_sequence = scaled_tail[-st.session_state.time_step:] 
            
            # Determine Noise Level
            noise_level = 0.0
            if enable_noise:
                noise_level = metrics['RMSE']
            
            # Forecast RETURNS
            pred_returns = forecast_future_robust(model, last_sequence, forecast_days, scaler, feature_cols, last_raw, target_idx, noise_level=noise_level)
            
            # Reconstruct PRICE path
            last_absolute_price = last_raw['Close'].iloc[-1]
            future_prices = []
            curr = last_absolute_price
            for r in pred_returns:
                curr = curr * np.exp(r)
                future_prices.append(curr)
                
            future_dates = pd.date_range(start=last_raw.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
            
            fig = go.Figure()
            # Show last 100 days of history
            fig.add_trace(go.Scatter(x=last_raw.index[-100:], y=last_raw['Close'].tail(100), name="History", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name="Forecast", line=dict(color='green', dash='dash')))
            fig.update_layout(title="Price Forecast (Reconstructed from Log Returns)", template="plotly_dark")
            st.plotly_chart(fig)
            
            st.write("Forecasted Prices:")
            st.dataframe(pd.DataFrame({"Date": future_dates, "Price": future_prices}))

except Exception as e:
    st.error(f"An error occurred: {e}")
