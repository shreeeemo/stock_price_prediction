import yfinance as yf

def fetch_multiple_stocks(tickers, start_date, end_date):
    stock_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        stock_data[ticker] = data
        print(f"Fetched data for {ticker}")
    return stock_data
