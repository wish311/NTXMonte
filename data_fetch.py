import yfinance as yf
import numpy as np

def fetch_stock_data(symbol, start, end):
    """Fetches historical stock data from Yahoo Finance."""
    try:
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data found for {symbol}. Check the ticker symbol.")
        return data['Close']  # 'Adj Close' might not exist, fallback to 'Close'
    except Exception as e:
        print(f"[-] Error fetching data: {e}")
        return None


def calculate_mu_sigma(prices):
    """Calculates annualized mean return (mu) and volatility (sigma)."""
    try:
        log_returns = np.log(prices / prices.shift(1)).dropna()
        mu = log_returns.mean() * 252
        sigma = log_returns.std() * np.sqrt(252)
        return mu, sigma
    except Exception as e:
        print(f"[-] Error calculating mu/sigma: {e}")
        return None, None
