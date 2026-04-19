# 📈 Smart Paper Trading App

> A real-time AI-powered stock market simulator that combines live market data, LSTM-based price prediction, and a paper trading engine — built with Streamlit, PyTorch, and Yahoo Finance.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red?logo=streamlit)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-orange?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](licence)

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Project Structure](#-project-structure)
4. [Architecture & Data Flow](#-architecture--data-flow)
5. [Pipeline: Training Phase](#-pipeline-training-phase)
6. [Pipeline: Live Trading Phase](#-pipeline-live-trading-phase)
7. [File Execution — Step by Step](#-file-execution--step-by-step)
8. [How to Run](#️-how-to-run)
9. [Requirements](#️-requirements)
10. [Notes & Disclaimers](#-notes--disclaimers)
11. [Author](#-author)

---

## 🔍 Overview

Smart Paper Trading App lets you simulate stock trading with **zero financial risk**. It pulls real OHLCV data from Yahoo Finance, renders a live candlestick chart, and uses a pre-trained **LSTM neural network** to predict the next closing price and suggest BUY / SELL / HOLD actions. All trades are simulated against a virtual ₹100,000 (or $100,000) starting balance.

---

## 🚀 Features

| Feature | Details |
|---|---|
| 📊 Live candlestick chart | Auto-refreshes every 60 seconds using 1-minute OHLCV data |
| 🤖 LSTM price prediction | PyTorch model predicts next close price from 78 × 5-minute candles |
| 💡 Trade signal | BUY / SELL / HOLD recommendation with % change estimate |
| ✅ Paper buy & sell | Trades allowed every 5 minutes; enforced by real clock |
| 💼 Real-time portfolio | Cash, holdings, and total portfolio value update on every refresh |
| 🧾 Trade history | Full log of every order with timestamp, price, and quantity |
| 🏆 Best-model selection | Trains 24 candidate models (7–30 day windows), keeps lowest-RMSE model |

---

## 📦 Project Structure

```
Smart_Paper_Trade_AI/
│
├── app.py                          # Streamlit entry point — UI, chart, trading panel
│
├── src/
│   ├── models/
│   │   ├── model.py                # LSTM architecture definition (PyTorch nn.Module)
│   │   ├── train_model.py          # Trains one LSTM model for a given day-window
│   │   ├── selector_model.py       # Evaluates 7–30-day models, saves the best one
│   │   ├── load_and_predict_model.py  # Loads saved model, runs inference, emits signals
│   │   └── paper_trade.py          # PaperTrader class — buy, sell, portfolio state
│   │
│   └── utils/
│       └── data_loader.py          # Downloads and cleans OHLCV data from yfinance
│
├── models/
│   ├── LSTM_model.ptl              # Saved PyTorch model state dict (best model)
│   └── scaler.pkl                  # Fitted StandardScaler (joblib serialised)
│
├── requirements.txt
└── README.md
```

---

## 🏗️ Architecture & Data Flow

The system is split into two independent phases: a **one-time training pipeline** and a **continuous live trading pipeline**.

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         TRAINING PIPELINE                               ║
║  (Run once via selector_model.py to produce LSTM_model.ptl & scaler.pkl)║
║                                                                          ║
║  Yahoo Finance  ──►  data_loader.py  ──►  train_model.py               ║
║  (5m OHLCV,           fetch_data()          StandardScaler               ║
║   7–30 days)          clean & index         Sliding window (78 steps)    ║
║                                             LSTM training (50 epochs)    ║
║                              ▼                                           ║
║                       selector_model.py                                  ║
║                       evaluate each candidate on yesterday's data        ║
║                       pick lowest RMSE  ──►  models/LSTM_model.ptl      ║
║                                         ──►  models/scaler.pkl          ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                      LIVE TRADING PIPELINE                              ║
║           (Runs continuously inside Streamlit — app.py)                 ║
║                                                                          ║
║  Yahoo Finance  ──►  app.py (fetch_data)  ──►  Plotly Candlestick      ║
║  (1m OHLCV,          current_price                 chart rendered       ║
║   today)                   │                                             ║
║                             │                                            ║
║                             ▼                                            ║
║                    PaperTrader (paper_trade.py)                         ║
║                    ├── buy()  — debit cash, add holdings                ║
║                    ├── sell() — credit cash, remove holdings            ║
║                    └── status() — return live portfolio snapshot        ║
║                                                                          ║
║  Yahoo Finance  ──►  data_loader.py  ──►  load_and_predict_model.py    ║
║  (5m OHLCV,          fetch_data()          load LSTM_model.ptl          ║
║   last 7 days)       last 78 candles       load scaler.pkl              ║
║                       scale features       model inference              ║
║                       reshape → tensor     inverse-transform            ║
║                                     │                                   ║
║                                     ▼                                   ║
║                             get_trade_signal()                          ║
║                             BUY / SELL / HOLD + % change               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 🔧 Pipeline: Training Phase

> **When to run:** Before first use, or periodically to retrain on fresh data.  
> **Entry point:** `python src/models/selector_model.py`

### Step-by-step

```
selector_model.py  →  for days in range(7, 31):
    │
    ├── train_model.py :: train_lstm_for_days(ticker, days)
    │       │
    │       ├── data_loader.py :: fetch_data(ticker, days, interval="5m")
    │       │       └── yf.download() → raw MultiIndex DataFrame
    │       │           dropna, reset_index, set Datetime as index
    │       │           → clean pd.DataFrame (columns: Open, High, Low, Close, Volume)
    │       │
    │       ├── Drop High, Low, Open, Volume  → only Close (+ Adj Close if present)
    │       ├── StandardScaler.fit_transform()
    │       ├── Sliding window (size=78): X[i] = scaled[i:i+78], y[i] = Close[i+78]
    │       ├── TensorDataset + DataLoader (batch=32, shuffle=True)
    │       ├── LSTMModel (input=1, hidden=64, layers=2, output=1)
    │       ├── Adam(lr=0.001) + MSELoss
    │       └── 50 training epochs → return model, scaler, df
    │
    ├── selector_model.py :: prepare_yesterday_data(df, scaler)
    │       └── slices df to day-before-yesterday + yesterday
    │           scales features, builds X/y windows
    │           returns x_yes tensor, y_true array
    │
    ├── model(x_yes) → predictions
    ├── RMSE(y_true, predictions)
    └── track best_rmse → best_model, best_scaler

After all candidates evaluated:
    torch.save(best_model.state_dict(), "models/LSTM_model.ptl")
    joblib.dump(best_scaler, "models/scaler.pkl")
```

---

## 📡 Pipeline: Live Trading Phase

> **Entry point:** `streamlit run app.py`

### Step-by-step

```
Browser opens localhost:8501
    │
    ▼
app.py starts
    ├── st_autorefresh(interval=60 000 ms) — page reloads every 60 s
    ├── Session state init: PaperTrader(starting_cash=100 000)
    │
    ├── fetch_data("AAPL", interval="1m", period="1d")    ← inline helper
    │       └── yf.download → MultiIndex fix → timezone IST → return df
    │
    ├── current_price = df["Close"].iloc[-1]
    │
    ├── trader.status(current_price)                       ← paper_trade.py
    │       └── returns {Cash, Holdings, Current Price, Portfolio Value}
    │           rendered as fixed overlay card (HTML/CSS)
    │
    ├── plot_candlestick(df) → Plotly Figure → st.plotly_chart()
    │
    ├── Trading Panel
    │       ├── st.number_input("Enter Quantity")
    │       ├── Check datetime.now().minute % 5 == 0  (trade window gate)
    │       ├── [Buy] → trader.buy(current_price, qty, timestamp)
    │       │           cost = price × qty
    │       │           if cash ≥ cost: cash -= cost; holdings += qty; log trade
    │       └── [Sell] → trader.sell(current_price, qty, timestamp)
    │                    if holdings ≥ qty: cash += revenue; holdings -= qty; log trade
    │
    ├── trader.get_trade_dataframe() → st.dataframe (trade history)
    │
    └── LSTM Prediction Section
            ├── load_and_predict_model.py :: predict_close_price("AAPL")
            │       ├── joblib.load("models/scaler.pkl")
            │       ├── LSTMModel() + model.load_state_dict(LSTM_model.ptl)
            │       ├── fetch_data("AAPL", days=7, interval="5m")  ← data_loader.py
            │       ├── take last 78 rows (Close only after dropping OHLV)
            │       ├── scaler.transform() → reshape (1, 78, features)
            │       ├── torch.no_grad(): model(tensor) → predicted_scaled
            │       └── scaler.inverse_transform() → predicted_close (USD)
            │
            └── get_trade_signal(current, predicted, threshold=0.2)
                    change = (predicted - current) / current × 100
                    change > 0.2%  → BUY ✅
                    change < -0.2% → SELL ❌
                    else           → HOLD ⏸️
```

---

## 📂 File Execution — Step by Step

### 1. `src/utils/data_loader.py`
**Role:** Data ingestion utility  
**Called by:** `train_model.py`, `load_and_predict_model.py`

- Accepts `ticker`, `days`, and `interval` parameters.
- Downloads OHLCV data from Yahoo Finance via `yf.download()`.
- Flattens any `MultiIndex` columns produced by yfinance.
- Drops `NaN` rows, resets index, sets `Datetime` as the DataFrame index.
- Returns a clean `pd.DataFrame` ready for scaling and model input.

---

### 2. `src/models/model.py`
**Role:** Neural network architecture  
**Called by:** `train_model.py`, `load_and_predict_model.py`

- Defines `LSTMModel(nn.Module)` with 2 LSTM layers (hidden size 64) and a single linear output layer.
- `forward(x)` passes input through the LSTM stack and reads the last time-step's hidden state through the fully-connected layer to produce a scalar prediction.

---

### 3. `src/models/train_model.py`
**Role:** Single-model training  
**Called by:** `selector_model.py`

- `train_lstm_for_days(ticker, days, window=78)`:
  1. Calls `fetch_data()` to obtain `days` worth of 5-minute bars.
  2. Retains only the `Close` column (drops OHLV and Volume).
  3. Fits a `StandardScaler` and transforms the data.
  4. Builds overlapping sequence windows of length 78 → `(X, y)` pairs.
  5. Trains `LSTMModel` for 50 epochs with Adam optimizer and MSE loss.
  6. Returns `(model, scaler, df)`.

---

### 4. `src/models/selector_model.py`
**Role:** Model selection and persistence  
**Entry point:** Run directly (`python src/models/selector_model.py`) to (re)train

- `select_best_lstm_model(ticker, candidates=range(7,31))`:
  1. Iterates over 24 candidate training windows (7 to 30 days).
  2. For each, calls `train_lstm_for_days()` and then `prepare_yesterday_data()`.
  3. `prepare_yesterday_data()` isolates the two most recent complete trading days, scales them, and builds inference windows.
  4. Runs the freshly trained model over those windows and computes RMSE against actual closing prices.
  5. Tracks the candidate with the lowest RMSE.
  6. Saves the winner: `models/LSTM_model.ptl` (state dict) and `models/scaler.pkl` (fitted scaler).

---

### 5. `src/models/load_and_predict_model.py`
**Role:** Inference and signal generation  
**Called by:** `app.py`

- `predict_close_price(ticker, days=7, interval="5m", window_size=78)`:
  1. Loads `scaler.pkl` (joblib) and `LSTM_model.ptl` (torch).
  2. Fetches the last 7 days of 5-minute bars via `data_loader.fetch_data()`.
  3. Slices the most recent 78 rows (Close column only).
  4. Applies the saved scaler, reshapes to `(1, 78, features)`, runs a forward pass.
  5. Inverse-transforms the output scalar to get a price in USD.
  6. Returns `(current_close, predicted_close)`.

- `get_trade_signal(current, predicted, threshold=0.2)`:
  - Computes `% change = (predicted − current) / current × 100`.
  - Returns `"BUY ✅"` if change > +0.2%, `"SELL ❌"` if change < −0.2%, else `"HOLD ⏸️"`.

---

### 6. `src/models/paper_trade.py`
**Role:** Virtual trading engine  
**Called by:** `app.py`

- `PaperTrader(starting_cash=100 000)` maintains state: `cash`, `holdings`, `trades[]`.
- `buy(price, quantity, timestamp)`: validates sufficient cash, debits cost, increments holdings, appends to trade log.
- `sell(price, quantity, timestamp)`: validates sufficient holdings, credits revenue, decrements holdings, appends to trade log.
- `status(current_price)`: returns a snapshot dict (Cash, Holdings, Current Price, Portfolio Value).
- `get_trade_dataframe()`: returns the full trade history as a `pd.DataFrame`.

---

### 7. `app.py`
**Role:** Application entry point and UI  
**Run with:** `streamlit run app.py`

Execution order on every page load / 60-second auto-refresh:

| Step | Action |
|---|---|
| 1 | `st_autorefresh(60 000 ms)` — schedule next refresh |
| 2 | Init `PaperTrader` in `st.session_state` (once per session) |
| 3 | `fetch_data("AAPL", "1m", "1d")` — pull latest 1-minute bars, convert to IST |
| 4 | Compute `current_price` from last row |
| 5 | Call `trader.status(current_price)` — render portfolio overlay card |
| 6 | `plot_candlestick(df)` — render Plotly chart |
| 7 | Evaluate trade-window gate (`minute % 5 == 0`) — show Buy / Sell buttons |
| 8 | On button press: `trader.buy()` or `trader.sell()` |
| 9 | `trader.get_trade_dataframe()` — render trade history table |
| 10 | `predict_close_price("AAPL")` — run LSTM inference |
| 11 | `get_trade_signal(current, predicted)` — display BUY / SELL / HOLD metric |

---

## ▶️ How to Run

### Prerequisites
- Python 3.9 or higher
- Internet access (live market data)

### Install dependencies
```bash
pip install -r requirements.txt
```

### (Optional) Retrain the LSTM model
Run this once before launch, or periodically to update the model on fresh data:
```bash
python src/models/selector_model.py
```
This will evaluate 24 training windows and save the best model to `models/`.

### Launch the app
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚙️ Requirements

```
streamlit
yfinance
pandas
numpy
scikit-learn
torch
joblib
ta
plotly
streamlit_autorefresh
```

> Full pinned versions are in [`requirements.txt`](requirements.txt).

---

## 📌 Notes & Disclaimers

- 📊 This is a **paper trading** simulator — no real money is ever involved.
- 🕐 Live data works best during **NASDAQ / NYSE market hours** (US Eastern Time).
- ⚠️ Ensure your system clock is accurate for correct trade-window gating.
- 🤖 The LSTM model is a **short-term technical predictor** only. It does not account for news, earnings, macroeconomic events, or fundamental analysis.
- 🧪 Built for **educational and demonstration purposes** only. Do not use as financial advice.

---

## 👤 Author

**Soham Samanta**  
B.Tech in Computer Science  
Kalinga Institute of Industrial Technology (2023–2027)
