import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_data(ticker: str, days: int = 30, interval: str = "5m") -> pd.DataFrame:
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    df = yf.download(
        tickers=ticker,
        start=start_time.strftime("%Y-%m-%d"),
        end=end_time.strftime("%Y-%m-%d"),
        interval=interval,
        progress=True
    )

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}. Check if market is open or ticker is valid.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.dropna()
    df.reset_index(inplace=True)

    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)

    return df
