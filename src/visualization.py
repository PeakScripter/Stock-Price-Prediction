import plotly.graph_objects as go
import pandas as pd

def plot_interactive_predictions(df, test_predict, train_size, time_step):
    """
    Plot interactive stock price predictions using Plotly.
    """
    fig = go.Figure()

    # Actual Data
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines', name='Actual Price',
        line=dict(color='blue')
    ))

    # We need to align test predictions with dates
    # test_predict has length len(test) - time_step
    # start index for test predictions = train_size + time_step
    
    test_start_idx = train_size + time_step
    if test_start_idx < len(df):
        test_dates = df.index[test_start_idx : test_start_idx + len(test_predict)]
        
        # Flatten predictions if they are (N, 1)
        test_predict_flat = test_predict.flatten()
        
        fig.add_trace(go.Scatter(
            x=test_dates, y=test_predict_flat,
            mode='lines', name='Test Prediction',
            line=dict(color='red')
        ))

    fig.update_layout(
        title="Stock Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig

def plot_candlestick(df):
    """
    Plot interactive candlestick chart.
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
         name='Market Data'
    )])

    fig.update_layout(
         title='Interactive Candlestick Chart',
         xaxis_title='Date',
         yaxis_title='Price',
         template='plotly_dark'
    )
    return fig
