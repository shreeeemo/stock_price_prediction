# Stock Price Prediction Viewer

This project uses **LSTM neural networks** to predict stock prices for the top 10 S&P 500 companies. It features a **web-based interactive dashboard** built with **Plotly Dash** to visualize the predictions dynamically.

## Features
- **Historical Stock Data**: Fetches stock data from Yahoo Finance using `yfinance`.
- **Neural Network Model**: LSTM-based deep learning model for stock price prediction.
- **Interactive Dashboard**: Visualize actual vs. predicted stock prices for each company with a dropdown menu.
- **Extensible Design**: Easily adapt to include additional stocks or new prediction models.

## Project Structure
.
├── main.py                   # Main script for data fetching, preprocessing, model training
├── data_fetch.py             # Fetches stock data using yfinance
├── preprocess.py             # Scales and sequences the stock data for the model
├── model_training.py         # Defines and trains the LSTM model
├── predict_and_plot.py       # Contains functions for interactive plotting
├── dashboard.py              # Dash app for the interactive dashboard
├── models/                   # Directory to save trained models
├── requirements.txt          # Dependencies required to run the project
├── README.md                 # Project documentation

# Project documentation

## Installation
To set up the project on your local machine, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shreeeemo/stock_price_predictor.git
   cd https://github.com/shreeeemo/stock_price_predictor.git

2. **Set Up a Virtual Environment**:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**:
pip install -r requirements.txt

4. **Fetch Data and Train Models**: Run the main script to fetch data and train the models:
python main.py

5. **Launch the Interactive Dashboard**: Run the dashboard to visualize predictions:
python dashboard.py

6. **Access the Dashboard**: Open your browser and navigate to:
http://127.0.0.1:8050

## Dependencies
The project requires the following Python libraries (see requirements.txt for version details):

numpy
pandas
tensorflow
sklearn
yfinance
dash
plotly

## How It Works
1. Data Fetching: Historical stock data is downloaded using the yfinance library.

2. Data Preprocessing:
Stock prices are scaled using MinMaxScaler.
Sequence data is created for the LSTM model.

3. Model Training:
A custom LSTM model is trained for each stock.
Models are saved in the models/ directory.

4. Interactive Visualization:
The dashboard.py script creates a Plotly Dash app.
Users can select a stock from a dropdown menu to view its predictions.

## Data Preprocessing:
Stock prices are scaled using MinMaxScaler.
Sequence data is created for the LSTM model.
Model Training:

A custom LSTM model is trained for each stock.
Models are saved in the models/ directory.
Interactive Visualization:

The dashboard.py script creates a Plotly Dash app.
Users can select a stock from a dropdown menu to view its predictions.

## Future Improvements
Add more stock tickers dynamically.
Include additional features like High, Low, and Volume for more robust predictions.
Implement model performance evaluation metrics (e.g., R-squared).
Add user authentication for customized dashboards.

## Contributing
Feel free to contribute to this project by opening an issue or submitting a pull request. Contributions are always welcome!
