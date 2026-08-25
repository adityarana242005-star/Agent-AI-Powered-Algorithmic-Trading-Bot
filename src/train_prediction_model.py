"""
Direction prediction for BTC 15-min candles.

Tries to answer: will the next candle close higher or lower?
Uses Random Forest and Gradient Boosting, picks whichever works better.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
processed_dir = os.path.join(project_root, "data", "processed")
model_dir = os.path.join(project_root, "models")

os.makedirs(model_dir, exist_ok=True)

# load our train/test splits
train_df = pd.read_csv(os.path.join(processed_dir, "BTC_train.csv"))
test_df = pd.read_csv(os.path.join(processed_dir, "BTC_test.csv"))
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# create the target: 1 if next candle closes higher, 0 otherwise
train_df["Target"] = (train_df["Close"].shift(-1) > train_df["Close"]).astype(int)
test_df["Target"] = (test_df["Close"].shift(-1) > test_df["Close"]).astype(int)

# last row has no "next candle" so drop it
train_df = train_df.iloc[:-1]
test_df = test_df.iloc[:-1]

# these are the features i found most useful after experimenting
features = [
    "Return", "SMA_Cross", "Volume_Change", "Volatility",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist", "High_Low",
    "Return_Lag1", "Return_Lag2", "Return_Lag3",
    "Momentum_5", "Momentum_10",
]

# only use columns that actually exist in our data
features = [f for f in features if f in train_df.columns]
print(f"\nUsing {len(features)} features: {features}")

X_train = train_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = train_df["Target"]
X_test = test_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_test = test_df["Target"]

print(f"Training on {len(X_train)} samples")
print(f"Class balance: {dict(y_train.value_counts())}")


# ---- Model 1: Random Forest ----

print("\n--- Random Forest ---")

rf = RandomForestClassifier(
    n_estimators=200, max_depth=12,
    min_samples_split=20, min_samples_leaf=10,
    max_features="sqrt", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds) * 100

print(f"Accuracy: {rf_acc:.2f}%")
print(classification_report(y_test, rf_preds, target_names=["DOWN", "UP"]))

# show which features matter most
print("Feature importance:")
for name, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:20s} {imp:.4f}")


# ---- Model 2: Gradient Boosting ----

print("\n--- Gradient Boosting ---")

gb = HistGradientBoostingClassifier(
    max_iter=300, max_depth=6, learning_rate=0.05,
    min_samples_leaf=50, l2_regularization=1.0,
    max_bins=128, random_state=42
)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_preds) * 100

print(f"Accuracy: {gb_acc:.2f}%")
print(classification_report(y_test, gb_preds, target_names=["DOWN", "UP"]))


# ---- pick the winner ----

print(f"\nRF: {rf_acc:.2f}% vs GB: {gb_acc:.2f}%")

if gb_acc >= rf_acc:
    best_preds, best_name, best_acc = gb_preds, "Gradient Boosting", gb_acc
    best_model = gb
else:
    best_preds, best_name, best_acc = rf_preds, "Random Forest", rf_acc
    best_model = rf

print(f"Winner: {best_name} ({best_acc:.2f}%)")


# ---- save everything ----

# predictions csv for dashboard
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": best_preds,
    "Actual_Direction": y_test.map({1: "UP", 0: "DOWN"}).values,
    "Predicted_Direction": pd.Series(best_preds).map({1: "UP", 0: "DOWN"}).values,
    "RF_Predicted": rf_preds,
    "GB_Predicted": gb_preds,
})
results_df.to_csv(os.path.join(processed_dir, "direction_predictions.csv"), index=False)

# summary for quick reference
summary_df = pd.DataFrame({
    "Model": [best_name], "Accuracy": [best_acc],
    "RF_Accuracy": [rf_acc], "GB_Accuracy": [gb_acc],
    "Total_Predictions": [len(y_test)],
    "Correct_Predictions": [int((y_test.values == best_preds).sum())]
})
summary_df.to_csv(os.path.join(processed_dir, "prediction_summary.csv"), index=False)

# save the actual trained models so dashboard can use them for live prediction
joblib.dump(rf, os.path.join(model_dir, "rf_model.joblib"))
joblib.dump(gb, os.path.join(model_dir, "gb_model.joblib"))
joblib.dump(features, os.path.join(model_dir, "feature_list.joblib"))

print(f"\nAll saved! Models in {model_dir}")
