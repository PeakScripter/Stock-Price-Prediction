# Stock Price Prediction with Advanced Deep Learning

This project predicts stock prices using advanced Deep Learning models (LSTM-CNN, Bidirectional LSTM, Attention LSTM) and includes a comprehensive Streamlit dashboard for interactive analysis and backtesting.

**Key Update:** The model now predicts **Log Returns** instead of absolute prices to ensure stationarity and better generalization. It also features a **Robust Forecasting** module using Monte Carlo simulations to quantify uncertainty.

## Features

- **Advanced Models:**
  - **LSTM-CNN:** Hybrid architecture utilizing Convolutional Neural Networks for feature extraction and LSTM for sequence modeling.
  - **Bidirectional LSTM:** Captures patterns from both past and future states.
  - **Attention LSTM:** Focuses on relevant time steps for better prediction accuracy.
- **Log Return Prediction:** Predicts logarithmic returns to handle non-stationary stock data effectively.
- **Robust Forecasting:** Generates future price paths with optional Monte Carlo noise injection based on model RMSE to simulate market volatility.
- **Technical Indicators:** Automatically calculates RSI, MACD, Bollinger Bands, SMA, EMA, and Volume metrics.
- **Interactive Dashboard:** Built with Streamlit and Plotly for real-time visualization.
- **Backtesting:** Simulates a simple trading strategy and calculates key metrics like Total Return.

## Project Structure

```
.
├── app.py                  # Streamlit Dashboard entry point
├── config.yaml             # Configuration file
├── models/                 # specific directory for saved models
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data fetching and preprocessing
│   ├── features.py         # Technical indicator generation
│   ├── model.py            # Model architectures (Keras/TensorFlow)
│   ├── backtest.py         # Trading strategy and metrics
│   ├── visualization.py    # Plotly charting functions
│   └── inference.py        # Forecasting logic and model persistence
└── README.md
```

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: You may need to create a `requirements.txt` with:*
```
yfinance
pandas
numpy
scikit-learn
tensorflow
ta
streamlit
plotly
pyyaml
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

1. **Configuration**:
    - Enter a Stock Ticker (e.g., `^NSEI`, `AAPL`, `GOOG`).
    - Select a Date Range.
    - Choose **"Train New Model"** or **"Load Pre-trained Model"**.
2. **Training**:
    - Select Model Type (LSTM-CNN, Bi-LSTM, Attention LSTM).
    - Adjust Epochs and Batch Size.
    - Click **Train Model**. The system will automatically compute Log Returns and normalize data.
3. **Forecasting**:
    - After training/loading, scroll to "Future Forecast".
    - Adjust **Days to Forecast**.
    - Toggle **Enable Monte Carlo Simulation** to add noise based on test error.
    - Click **Generate Forecast** to view the projected price path with uncertainty (if enabled).

## Configuration

You can adjust default parameters in `config.yaml` or directly in the Streamlit sidebar.
