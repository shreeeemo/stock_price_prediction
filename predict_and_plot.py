import matplotlib.pyplot as plt
from ipywidgets import interact, fixed

def interactive_plot(ticker, results):
    result = results[ticker]
    model = result['model']
    X_test = result['X_test']
    y_test = result['y_test']
    scaler = result['scaler']

    # Predict and plot
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.title(f'Stock Price Prediction for {ticker}')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Function to start the interactive widget
def start_interactive_plot(results, top_10_tickers):
    interact(interactive_plot, ticker=top_10_tickers, results=fixed(results))
