import pandas as pd
import os

# path to my raw bitcoin data file
filepath = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw",
    "btc_15m_data_2018_to_2025.csv"
)

df = pd.read_csv(filepath)
print("Loaded raw data")
print(df.shape)
print(df.head(3))

# make sure the timestamp column is proper datetime
df["Open time"] = pd.to_datetime(df["Open time"])

# sort chronologically just in case its not already
df = df.sort_values("Open time").reset_index(drop=True)

# get rid of any duplicate rows that might have been scraped twice
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"\nRemoved {before - after} duplicate rows")

# drop rows where any value is missing
df = df.dropna()

print(f"Final cleaned rows: {len(df)}")
print(df.dtypes)

# save to processed folder
save_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed",
    "BTC_15min_cleaned.csv"
)
df.to_csv(save_path, index=False)
print(f"\nSaved cleaned data to {save_path}")