"""
Evaluate PPO Trading Agent
============================
Loads the trained PPO model and evaluates it on the test dataset.

Saves:
- data/processed/evaluation_results.csv (per-step portfolio + actions)
- data/processed/evaluation_summary.csv (summary metrics for dashboard)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trading_environment import BitcoinTradingEnv
from performance_metrics import (
    get_return,
    get_max_drawdown,
    get_sharpe_ratio,
    print_results
)


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data" / "processed"
model_dir = BASE_DIR / "models"


# =====================
# Load test data
# =====================


data = pd.read_csv(data_dir / "BTC_test.csv")

print(f"Test data: {len(data)} rows")


# =====================
# Create environment and load model
# =====================


env = BitcoinTradingEnv(data)

model = PPO.load(str(model_dir / "bitcoin_ppo"))

print("Model loaded successfully!")


# =====================
# Run evaluation
# =====================


observation, info = env.reset()

done = False

buy_count = 0
hold_count = 0
sell_count = 0

portfolio_values = []
actions = []


while not done:

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    action = int(action)

    observation, reward, done, truncated, info = env.step(
        action
    )

    # Save portfolio value
    portfolio_values.append(
        info["portfolio_value"]
    )

    # Save action
    actions.append(action)

    # Count actions
    if action == 0:
        hold_count += 1
    elif action == 1:
        buy_count += 1
    elif action == 2:
        sell_count += 1


# =====================
# Results
# =====================


starting_money = 10000
final_money = info["portfolio_value"]
trades = buy_count + sell_count


print()
print("=" * 40)
print("AI ACTIONS")
print("=" * 40)

print(f"BUY :  {buy_count}")
print(f"HOLD:  {hold_count}")
print(f"SELL:  {sell_count}")
print(f"Total Trades: {trades}")


# Performance results
print_results(
    starting_money,
    final_money,
    portfolio_values,
    trades
)


# Buy and Hold
first_price = data["Close"].iloc[0]
last_price = data["Close"].iloc[-1]

buy_hold_return = (
    (last_price - first_price)
    / first_price
) * 100


print()
print("=" * 40)
print("BUY & HOLD BENCHMARK")
print("=" * 40)
print(f"Bitcoin Return: {buy_hold_return:.2f}%")


# =====================
# Save per-step results
# =====================


results = pd.DataFrame({
    "Portfolio_Value": portfolio_values,
    "Action": actions
})

results.to_csv(
    data_dir / "evaluation_results.csv",
    index=False
)


# =====================
# Save summary for dashboard
# =====================


total_return = get_return(starting_money, final_money)
max_drawdown = get_max_drawdown(portfolio_values)
sharpe = get_sharpe_ratio(portfolio_values)

summary = pd.DataFrame({
    "Starting_Money": [starting_money],
    "Final_Money": [round(final_money, 2)],
    "Return": [round(total_return, 2)],
    "Max_Drawdown": [round(max_drawdown, 2)],
    "Sharpe_Ratio": [round(sharpe, 2)],
    "Number_of_Trades": [trades],
    "BUY_Count": [buy_count],
    "HOLD_Count": [hold_count],
    "SELL_Count": [sell_count],
    "Buy_Hold_Return": [round(buy_hold_return, 2)]
})

summary.to_csv(
    data_dir / "evaluation_summary.csv",
    index=False
)


print()
print("Evaluation results saved!")
print("Summary saved!")