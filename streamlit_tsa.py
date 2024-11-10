import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tqdm import tqdm
import base64

# Set up the background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                color: white; /* Set font color to white */
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Apply the background
set_background('bg_tsa.jpg')

# Function to load and preprocess the data
def preprocess_data(file):
    df = pd.read_csv(file)
    df_ema = apply_ema(df)
    lagged_df = create_lagged_features(df_ema, n_lags=5)

    X = lagged_df.dropna().drop(columns=['t1', 't2'])  
    y = lagged_df[['t1', 't2']]  

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    return X, y

# Function to create EMA
def apply_ema(data, span=10):
    data['t1_ema'] = data['t1'].ewm(span=span, adjust=False).mean()
    data['t2_ema'] = data['t2'].ewm(span=span, adjust=False).mean()
    return data

# Function to create lagged features
def create_lagged_features(data, n_lags=5):
    lagged_data = data.copy()
    for lag in range(1, n_lags + 1):
        lagged_data = pd.concat([lagged_data, data[['t1', 't2']].shift(lag).add_suffix(f'_lag{lag}')], axis=1)
    lagged_data = lagged_data.dropna()  
    return lagged_data

# Function to forecast for different window sizes
def forecast(model, X, windows=[60], batch_size=32):
    forecasts = {}
    for N in windows:
        forecasted_values = []

        for i in tqdm(range(0, len(X), batch_size), desc=f"Predicting for window size {N}"):
            batch_input = X[i:i+batch_size, :, :]
            batch_predictions = model.predict(batch_input)
            forecasted_values.append(batch_predictions)

        forecasted_values = np.vstack(forecasted_values)
        forecasts[N] = forecasted_values

    return forecasts

# Function to plot forecasts
def plot_forecasts(forecasts, true_values):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arange(len(true_values)), y=true_values[:, 0], mode='lines', name='Actual t1'))
    fig.add_trace(go.Scatter(x=np.arange(len(true_values)), y=true_values[:, 1], mode='lines', name='Actual t2'))

    for window, forecasted in forecasts.items():
        forecasted_x = np.arange(len(true_values), len(true_values) + len(forecasted))

        fig.add_trace(go.Scatter(x=forecasted_x, y=forecasted[:, 0], mode='lines', name=f'Forecasted t1 (N={window})'))
        fig.add_trace(go.Scatter(x=forecasted_x, y=forecasted[:, 1], mode='lines', name=f'Forecasted t2 (N={window})'))

    fig.update_layout(title='True and Forecasted Values for t1 and t2',
                      xaxis_title='Time Steps',
                      yaxis_title='Values',
                      template='plotly_dark')

    st.plotly_chart(fig)

# Streamlit app
st.title("Time Series Forecasting with RNN, GRU & LSTM")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
model_choice = st.selectbox("Select Model", ("Simple RNN", "GRU", "LSTM"))
window_size = st.slider("Select Window Size", min_value=10, max_value=200, value=60, step=10)

# Load model based on choice
model_dict = {
    "Simple RNN": "RNN_model.h5",
    "GRU": "GRU_model.h5"
    "LSTM" : "LSTM_model.h5"
}
model_path = model_dict[model_choice]
model = load_model(model_path)

if uploaded_file is not None:
    X, y = preprocess_data(uploaded_file)
    forecasts = forecast(model, X, windows=[window_size])
    plot_forecasts(forecasts, y)
