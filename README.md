# 📈 Smart Paper Trading App

A real-time stock market simulator using Streamlit, Plotly, and Yahoo Finance 1-minute data. Users can place paper trades (buy/sell), visualize live candlestick charts, and monitor their virtual portfolio value — all with ₹100,000 starting cash.

---

## 🚀 Features
- 🔁 Live 1-minute candlestick chart (auto-refresh)
- ✅ Buy/Sell every 5 minutes
- 💼 Portfolio value updates in real-time
- 🧾 Trade history with time, price, and quantity
- 📉 Uses real market data from Yahoo Finance

---

## 📦 Project Structure
```
Smart_Paper_Trade_AI/
├── app.py                    
├── src/
│   ├── models/
│   │   ├── train_model.py
│   │   ├── model.py
│   │   ├── load_and_predict_model.py
│   │   ├── selector_model.py
│   │   └── paper_trade.py    
│   └── utils/
│       └── data_loader.py  
├── models/  
│   ├── LsTM_model.ptl
│   └── scaler.pkl              
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚙️ Requirements
- Python 3.9+
- streamlit
- yfinance
- plotly
- pandas
- numpy

---

## 📌 Notes
- 📊 This is a **paper trading** simulator, no real money is involved.
- 🕐 Works best during **NASDAQ (US) market hours**.
- ⚠️ Make sure your system time is synced with live market for real-time accuracy.
- 🧪 For educational and demonstration purposes only.

---

## 👤 Author
**Soham Samanta**  
B.Tech in Computer Science  
Kalinga Institute of Industrial Technology (2023–2027)
