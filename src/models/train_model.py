import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

from src.utils.data_loader import fetch_data
from src.models.model import LSTMModel

def train_lstm_for_days(ticker: str, days: int, window: int = 78):
    df = fetch_data(ticker, days=days)
    df.drop(['High', 'Low', 'Open', 'Volume'], axis=1, inplace=True)
    df.dropna(inplace=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    target_index = df.columns.get_loc("Close")
    x, y = [], []
    for i in range(len(scaled) - window - 1):
        x.append(scaled[i:i + window])
        y.append(scaled[i + window][target_index])

    x = torch.tensor(np.array(x), dtype=torch.float32)
    y = torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for _ in range(50):
        for xb, yb in dataloader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

    return model, scaler, df
