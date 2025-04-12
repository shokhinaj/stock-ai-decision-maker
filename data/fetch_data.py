import yfinance as yf
import pandas as pd

# Fetches stock data and computes key features for AI state representation
def get_stock_data(ticker='AAPL', period='6mo'):
    df = yf.download(ticker, period=period)

    # Compute additional columns for state features
    df['Return'] = df['Close'].pct_change()
    df['Short_MA'] = df['Close'].rolling(window=5).mean()
    df['Long_MA'] = df['Close'].rolling(window=20).mean()
    df['Volume_Norm'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()

    # Drop incomplete rows due to moving average NaNs
    df.dropna(inplace=True)
    return df.reset_index()