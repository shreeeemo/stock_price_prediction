import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# Use the `results` dictionary from your main script
from main import results, top_10_tickers

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Stock Price Prediction Viewer"),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in results.keys()],
        value=list(results.keys())[0]
    ),
    dcc.Graph(id='stock-graph')
])

# Callback to update the graph
@app.callback(
    Output('stock-graph', 'figure'),
    [Input('dropdown', 'value')]
)
def update_graph(ticker):
    result = results[ticker]
    model = result['model']
    X_test = result['X_test']
    y_test = result['y_test']
    scaler = result['scaler']

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test_actual.flatten(), mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(y=predictions.flatten(), mode='lines', name='Predicted Prices'))
    fig.update_layout(
        title=f'Stock Price Prediction for {ticker}',
        xaxis_title='Days',
        yaxis_title='Price',
        legend_title='Legend'
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
