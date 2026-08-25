import pandas as pd
from trading_environment import BitcoinTradingEnv

data = pd.read_csv("data/processed/BTC_15min_features.csv")
print("data loaded:")
print(data.head())
# 2. Create our trading environment
env = BitcoinTradingEnv(data)

print("\nTrading environment created!")


# 3. Start the environment
observation, info = env.reset()

print("\n Initial observation:")
print(observation)


# 4. Take a BUY action
# 0 = HOLD
# 1 = BUY
# 2 = SELL

observation, reward, done, truncated, info = env.step(1)

print("\nAfter BUY:")
print("Reward:", reward)
print("Balance:", info["balance"])
print("BTC held:", info["btc_held"])
print("Portfolio value:", info["portfolio_value"])