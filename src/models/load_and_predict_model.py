import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from src.models.model import LSTMModel
from src.utils.data_loader import fetch_data

def predict_close_price(ticker="AAPL", days=7, interval="5m", window_size=78):
    scaler = joblib.load('models/scaler.pkl')
    model = LSTMModel()
    model.load_state_dict(torch.load("models/LSTM_model.ptl"))
    model.eval()

    df = fetch_data(ticker, days=days, interval=interval)
    df = df.sort_index()

    df = df.drop(columns=['High', 'Low', 'Open', 'Volume'], errors='ignore')

    if df.shape[0] < window_size:
        raise ValueError(f"Not enough data: only {df.shape[0]} rows found.")

    last_window = df[-window_size:].copy()
    features = last_window.drop(columns=["Date"], errors='ignore')

    scaled = scaler.transform(features)
    recent_scaled_values = scaled.reshape(1, features.shape[0], features.shape[1])

    with torch.no_grad():
        input_tensor = torch.tensor(recent_scaled_values, dtype=torch.float32)
        output = model(input_tensor)
        predicted_scaled = output.item()

    inverse_input = np.zeros((1, features.shape[1]))
    inverse_input[0, features.columns.get_loc("Close")] = predicted_scaled
    predicted_close = scaler.inverse_transform(inverse_input)[0, features.columns.get_loc("Close")]

    current_close = features["Close"].values[-1]
    return current_close, predicted_close


def get_trade_signal(current, predicted, threshold=0.2):
    if current is None or predicted is None:
        return "⚠️ No Data", 0.0

    change = (predicted - current) / current * 100
    if change > threshold:
        return "BUY ✅", change
    elif change < -threshold:
        return "SELL ❌", change
    else:
        return "HOLD ⏸️", change
