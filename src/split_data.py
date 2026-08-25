import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, "..", "data", "processed")

# load features dataset
df = pd.read_csv(os.path.join(processed_dir, "BTC_15min_features.csv"))
print(f"Total samples: {len(df)}")

# chronological 80/20 split
# important: we cant shuffle because this is time series data
# if we shuffle, future data leaks into training set
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

train_df.to_csv(os.path.join(processed_dir, "BTC_train.csv"), index=False)
test_df.to_csv(os.path.join(processed_dir, "BTC_test.csv"), index=False)

print(f"Train: {len(train_df)} rows")
print(f"Test:  {len(test_df)} rows")
print("Done!")