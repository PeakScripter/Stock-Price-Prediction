# Stock Price Prediction using LSTM-CNN and Monte Carlo Simulation

This project demonstrates a comprehensive approach to stock price prediction using a hybrid *LSTM-CNN* model and a *Monte Carlo Simulation* to forecast future stock prices. The code integrates historical data collection, data preprocessing, model training, and future price projection with uncertainty quantification. The methodology applied in this project is highly relevant to both financial forecasting and quantitative analysis.

---

## Project Structure

This project is structured around a few key components, each of which plays a crucial role in the end-to-end stock price prediction process:

1. *Data Collection*: Downloading stock data using Yahoo Finance (yfinance) and cleaning it for use in the model.
2. *Data Preprocessing*: Scaling the data and preparing it for training the LSTM-CNN hybrid model.
3. *Model Architecture*: Building and training an LSTM-CNN model that leverages temporal dependencies and spatial feature extraction.
4. *Monte Carlo Simulation*: Running multiple simulations to forecast future prices and assess the range of potential outcomes.
5. *Prediction and Visualization*: Plotting the results of the model predictions and the Monte Carlo projections.

---

## Data Collection

### Function: download_stock_data(ticker, start, end)

This function downloads stock data from Yahoo Finance based on the provided ticker symbol and date range. It checks for the required columns like Open, High, Low, Close, and Volume. The function ensures the data is clean, converting the necessary columns to numeric values and dropping any rows with missing data.

python
stock_data = download_stock_data('^NSEI', '2015-01-01', '2023-01-01')


In the example above, we download the stock data for the *NSE Nifty 50* index between January 1, 2015, and January 1, 2023.

---

## Data Preprocessing

### Function: preprocess_data(stock_data, column='Close')

This function extracts the relevant column (default: Close) from the dataset and scales the values between 0 and 1 using the MinMaxScaler. This is necessary for the neural network, which requires the data to be in a normalized format for efficient training.

The scaled data is returned along with the scaler to reverse the scaling process later when making predictions.

python
scaled_data, scaler = preprocess_data(stock_data)


---

## Creating Input Dataset for LSTM

### Function: create_dataset(data, time_step=60)

This function prepares the time series data in the required format for the LSTM model. The function creates windows of data with a given time_step (default 60), which defines the number of previous days used to predict the next day’s stock price.

The data is split into X (input) and y (output) arrays.

python
X, y = create_dataset(scaled_data, time_step)


---

## LSTM-CNN Model Architecture

### Function: build_lstm_cnn_model(time_step)

The hybrid *LSTM-CNN* model combines the strengths of both Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The model architecture includes:

- *1D Convolutional Layer*: For feature extraction over time.
- *MaxPooling Layer*: For down-sampling and reducing the dimensionality.
- *LSTM Layer*: To capture long-term temporal dependencies.
- *Dense Layers*: For prediction.

The model is compiled using the Adam optimizer and mean_squared_error as the loss function.

python
model = build_lstm_cnn_model(time_step)


---

## Training the LSTM-CNN Model

The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing. Early stopping is used during training to prevent overfitting by monitoring the validation loss.

python
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])


---

## Monte Carlo Simulation for Stock Price Forecasting

### Function: monte_carlo_simulation(stock_data, num_simulations=10000, num_days=252)

The Monte Carlo simulation generates thousands of possible future stock prices based on historical volatility and daily returns. For each simulation, it projects the future stock prices over a period (default 252 days or one year). The simulation outputs the mean projection and uncertainty bounds (5th and 95th percentiles).

python
projection_mean, projection_lower, projection_upper = monte_carlo_simulation(stock_data)


This provides a probabilistic forecast of future stock prices, complementing the deterministic forecast from the LSTM-CNN model.

---

## Prediction and Visualization

### Function: plot_predictions

This function plots both the predictions from the LSTM-CNN model and the Monte Carlo simulations. The plot includes:

- *LSTM-CNN Predictions*: For the test set.
- *Monte Carlo Mean*: The average projected stock price over the next 252 days.
- *Monte Carlo Uncertainty*: The range of possible future stock prices (5th to 95th percentile).

python
plot_predictions(stock_data, test_predict, future_index, projection_mean, projection_lower, projection_upper)


---

## Future Enhancements

- *Hyperparameter Tuning*: Optimize the parameters like learning rate, number of units, and layers for better performance.
- *Additional Features*: Incorporate external data such as macroeconomic indicators or sentiment analysis from news articles.
- *Model Comparison*: Compare the LSTM-CNN model with other models like ARIMA, XGBoost, or fully connected deep neural networks.

---

## Output Graph

![image](https://github.com/user-attachments/assets/67d4d0d3-7946-46f3-9ad7-a338f13f2c0e)


---

## Conclusion

This project showcases the integration of deep learning models and probabilistic simulations for stock price prediction. While the LSTM-CNN model captures temporal dependencies in the stock price, the Monte Carlo simulation adds a layer of uncertainty, providing a more comprehensive view of future price movements. This methodology can be applied across various financial instruments to enhance trading strategies, risk management, and investment decisions.
