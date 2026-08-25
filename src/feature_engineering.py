import pandas as pd
import numpy as np
import os

# figure out where our project root is
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
processed_dir = os.path.join(project_root, "data", "processed")

# load the cleaned btc data
df = pd.read_csv(os.path.join(processed_dir, "BTC_15min_cleaned.csv"))
print(f"Loaded {len(df)} rows")
print(df.columns.tolist())

df["Open time"] = pd.to_datetime(df["Open time"])

# --- basic price features ---

# percentage return between consecutive candles
df["Return"] = df["Close"].pct_change()

# simple moving averages at different windows
df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()

# difference between fast and slow SMA gives us a crossover signal
df["SMA_Cross"] = df["SMA_10"] - df["SMA_20"]

# raw price change (in dollars)
df["Price_Change"] = df["Close"].diff()

# how much volume changed compared to last candle
df["Volume_Change"] = df["Volume"].pct_change()

# range of the candle
df["High_Low"] = df["High"] - df["Low"]

# rolling standard deviation of returns as a volatility measure
df["Volatility"] = df["Return"].rolling(window=20).std()


# --- RSI calculation (14 period) ---
# this is a common momentum indicator

price_delta = df["Close"].diff()
gains = price_delta.where(price_delta > 0, 0.0)
losses = (-price_delta).where(price_delta < 0, 0.0)

avg_gain = gains.rolling(window=14).mean()
avg_loss = losses.rolling(window=14).mean()

relative_strength = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + relative_strength))


# --- MACD (Moving Average Convergence Divergence) ---
# uses 12 and 26 period exponential moving averages

ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
ema_slow = df["Close"].ewm(span=26, adjust=False).mean()

df["MACD"] = ema_fast - ema_slow
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]


# --- lagged returns ---
# give the model access to recent past returns

for lag in [1, 2, 3]:
    df[f"Return_Lag{lag}"] = df["Return"].shift(lag)


# --- momentum features ---

df["Momentum_5"] = df["Close"].pct_change(periods=5)
df["Momentum_10"] = df["Close"].pct_change(periods=10)


# --- cleanup ---
# pct_change and rolling windows create inf and nan values
# need to handle those before feeding to any model

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print(f"\nAfter feature engineering: {len(df)} rows")
print("Columns:", df.columns.tolist())
print(df.tail(3))

# save
output_path = os.path.join(processed_dir, "BTC_15min_features.csv")
df.to_csv(output_path, index=False)
print(f"\nFeatures saved to {output_path}")
