import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_and_train_model(X_train, y_train, input_shape, model_path="models/stock_price_model.h5", epochs=10, batch_size=32):
    # Ensure the directory for the model file exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Define the model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    # Save the model to a file
    # Ensure the file path uses the .keras extension to save in the native Keras format
    if not model_path.endswith(".keras"):
        model_path = model_path.replace(".h5", ".keras")

    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model
