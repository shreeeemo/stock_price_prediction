from data_fetch import fetch_multiple_stocks
from preprocess import preprocess_data
from model_training import build_and_train_model
from predict_and_plot import start_interactive_plot 

# Define the top 10 S&P 500 companies by ticker
top_10_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B', 'JNJ', 'V']

# Fetch data for all companies
start_date = '2015-01-01'
end_date = '2023-12-31'
all_stock_data = fetch_multiple_stocks(top_10_tickers, start_date, end_date)

# Loop through each stock
results = {}
for ticker, data in all_stock_data.items():
    print(f"Processing {ticker}...")
    X, y, scaler = preprocess_data(data)
    
    # Train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train the model
    model_path = f"models/{ticker}_stock_price_model.h5"
    model = build_and_train_model(X_train, y_train, input_shape=(X_train.shape[1], 1), model_path=model_path)
    
    # Predict and store results
    results[ticker] = {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler
    }
