import os
import sys
import torch
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import root_mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.train_model import train_lstm_for_days

def prepare_yesterday_data(df, scaler, window=78):
    df = df.copy()
    df['Date'] = df.index.date
    today = datetime.now().date()
    unique_dates = sorted(df['Date'].unique())

    if today in unique_dates:
        yesterday = unique_dates[-2]
        dby = unique_dates[-3]
    else:
        weekday = today.weekday()
        if weekday == 0:
            yesterday = today - timedelta(3)
            dby = today - timedelta(4)
        elif weekday == 5:
            yesterday = today - timedelta(1)
            dby = today - timedelta(2)
        elif weekday == 6:
            yesterday = today - timedelta(2)
            dby = today - timedelta(3)
        else:
            yesterday = today - timedelta(1)
            dby = today - timedelta(2)

    dby_df = df[df["Date"] == dby]
    yest_df = df[df["Date"] == yesterday]
    combined_df = pd.concat([dby_df, yest_df])

    if len(combined_df) < window + 1:
        return None, None, None

    feature_cols = [col for col in df.columns if col != "Date"]
    scaled = scaler.transform(combined_df[feature_cols])
    close_prices = combined_df["Close"].values

    x = []
    y = []
    for i in range(len(combined_df) - window - 1):
        x.append(scaled[i:window+i])
        y.append(close_prices[window+i+1])

    x_yes = torch.tensor(np.array(x), dtype=torch.float32)
    y_true = np.array(y)
    return x_yes, y_true


def evaluate_model(x_true, x_pred):
    return root_mean_squared_error(x_true, x_pred)


def select_best_lstm_model(ticker="AAPL", candidates=range(7, 31)):
    best_days = 0
    best_rmse = float("inf")
    best_model = None
    best_scaler = None
    os.makedirs("models", exist_ok=True)

    for days in candidates:
        print(f"Evaluating {days}-day model...")
        model, scaler, df = train_lstm_for_days(ticker, days)
        x_yes, y_true = prepare_yesterday_data(df, scaler)

        if x_yes is None:
            continue

        with torch.no_grad():
            x_pred = model(x_yes).squeeze()

        rmse = evaluate_model(y_true, x_pred.detach().numpy())

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_scaler = scaler
            best_days = days

    if best_model:
        torch.save(best_model.state_dict(), "models/LSTM_model.ptl")
        joblib.dump(best_scaler, "models/scaler.pkl")
        print(f"Best model with {best_days} days saved, RMSE: {best_rmse:.4f}")
    else:
        print("No suitable model found.")


if __name__ == "__main__":
    select_best_lstm_model("AAPL")
