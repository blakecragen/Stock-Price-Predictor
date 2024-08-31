import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import robin_stocks.robinhood as rh
from sortedcollections import SortedDict
import json
import threading

lock = threading.Lock()

# Step 1: Fetch data from Robinhood or alternative source
def fetch_stock_data(ticker, interval='day', span='5year'):
    rh.login(username='username', password='password')
    historicals = rh.stocks.get_stock_historicals(ticker, interval=interval, span=span)
    data = pd.DataFrame(historicals)
    data['begins_at'] = pd.to_datetime(data['begins_at'])
    data.sort_values('begins_at', inplace=True)
    numeric_columns = ['open_price', 'close_price', 'high_price', 'low_price', 'volume']
    for column in numeric_columns:
        data[column] = data[column].astype(float)
    rh.logout()
    return data

# Step 2: Preprocess the Data
def preprocess_data(data):
    data = data[['begins_at', 'close_price']]
    data.set_index('begins_at', inplace=True)
    # Normalize the 'close_price' data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])  # Last 60 days of prices
        y.append(scaled_data[i, 0])  # Predict the next day
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM input

    return X, y, scaler

# Step 3: Build Neural Network Model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Train the Model
def train_model(model, X_train, y_train, epochs=250, batch_size=50):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Step 5: Make Predictions
def predict_stock_price(model, X_test, scaler):
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    return predicted_stock_price

def predict_tomorrow_price(symbol='APPL', future_prices={}):
    try:
        with lock:
            data = fetch_stock_data(symbol, span='3month')  # Fetch stock data

        # print(data)
        X, y, scaler = preprocess_data(data)  # Preprocess the data

        model = create_model((X.shape[1], 1))  # Build the model
        model = train_model(model, X, y)  # Train the model

        predictions = predict_stock_price(model, X, scaler)  # Make predictions
        
        # Access with lock since it is a shared storage
        with lock:
            future_prices[symbol] = sum(predictions)/len(predictions) # Store prediction
        
        print(f"Completed Prediction for {symbol}")
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
    return

def main():
    stocks = json.load(open('stocks.json'))
    future_prices = SortedDict()
    threads = []

    for symbol in stocks['stocks']:
        # Create a thread for each stock
        thread = threading.Thread(target=predict_tomorrow_price, args=(symbol,future_prices,))
        threads.append(thread)
        thread.start()  # Start the thread
        

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    for stock in future_prices:
        print(f"Estimated price of {stock} is: ${float(future_prices[stock]):.2f}")