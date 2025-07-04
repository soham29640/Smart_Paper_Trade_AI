import pandas as pd

class PaperTrader:
    def __init__(self, starting_cash=100000):
        self.cash = starting_cash
        self.holdings = 0
        self.trades = []

    def buy(self, price, quantity, timestamp):
        cost = price * quantity
        if self.cash >= cost:
            self.cash -= cost
            self.holdings += quantity
            self.trades.append((timestamp, "BUY", price, quantity, self.cash, self.holdings))
            return f"✅ Bought {quantity} @ {price:.2f}"
        return "❌ Not enough cash"

    def sell(self, price, quantity, timestamp):
        if self.holdings >= quantity:
            revenue = price * quantity
            self.cash += revenue
            self.holdings -= quantity
            self.trades.append((timestamp, "SELL", price, quantity, self.cash, self.holdings))
            return f"✅ Sold {quantity} @ {price:.2f}"
        return "❌ Not enough holdings"

    def portfolio_value(self, current_price):
        return self.cash + (self.holdings * current_price)

    def status(self, current_price):
        return {
            "Cash": round(self.cash, 2),
            "Holdings": round(self.holdings, 2),
            "Current Price": round(current_price, 2),
            "Portfolio Value": round(self.portfolio_value(current_price), 2)
        }

    def get_trade_dataframe(self):
        columns = ["Timestamp", "Action", "Price", "Quantity", "Cash", "Holdings"]
        return pd.DataFrame(self.trades, columns=columns)
