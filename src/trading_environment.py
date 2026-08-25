"""
Custom gym environment for bitcoin trading.
The agent sees market data and decides to buy, hold, or sell.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BitcoinTradingEnv(gym.Env):

    def __init__(self, data):
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = 10000
        self.trading_fee = 0.001  # 0.1% per trade

        # these get reset at the start of each episode
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.entry_price = 0.0
        self.step_idx = 0

        # action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # observation: 10 normalized market + portfolio features
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(10,), dtype=np.float32
        )

    def _get_val(self, row, col, fallback=0.0):
        """safely grab a column value, return fallback if missing or inf"""
        if col in row.index:
            v = row[col]
            if np.isfinite(v):
                return float(v)
        return fallback

    def _build_observation(self):
        """construct the 10-dim state vector the agent sees"""
        row = self.data.iloc[self.step_idx]
        close = row["Close"]
        sma20 = self._get_val(row, "SMA_20", close)

        # 1) where is price relative to 20-period moving average
        price_vs_sma = (close - sma20) / sma20 if sma20 != 0 else 0

        # 2) log-scaled volume (raw volume numbers are huge)
        vol = np.log1p(self._get_val(row, "Volume", 0)) / 20

        # 3) last candle's return
        ret = self._get_val(row, "Return", 0)

        # 4) price change as fraction of current price
        pchange = self._get_val(row, "Price_Change", 0) / close if close != 0 else 0

        # 5) what fraction of portfolio is currently in btc
        total = self.balance + self.btc_held * close
        position_ratio = (self.btc_held * close) / total if total > 0 else 0

        # 6) RSI centered around 0 (50 becomes 0, 70 becomes 0.4, etc)
        rsi = self._get_val(row, "RSI", 50)
        rsi_norm = (rsi - 50) / 50

        # 7) MACD histogram normalized by price
        macd_h = self._get_val(row, "MACD_Hist", 0)
        macd_norm = (macd_h / close * 100) if close != 0 else 0

        # 8) recent volatility
        volatility = self._get_val(row, "Volatility", 0) * 100

        # 9) SMA crossover signal
        sma_x = self._get_val(row, "SMA_Cross", 0)
        sma_x_norm = (sma_x / close * 100) if close != 0 else 0

        # 10) unrealized profit/loss if we're holding btc
        if self.btc_held > 0 and self.entry_price > 0:
            unrealized = (close - self.entry_price) / self.entry_price
        else:
            unrealized = 0

        obs = np.array([
            price_vs_sma, vol, ret, pchange, position_ratio,
            rsi_norm, macd_norm, volatility, sma_x_norm, unrealized
        ], dtype=np.float32)

        return np.clip(obs, -10, 10)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.entry_price = 0.0
        self.step_idx = 0
        return self._build_observation(), {}

    def step(self, action):
        price = self.data.loc[self.step_idx, "Close"]

        # portfolio value before taking action
        old_value = self.balance + self.btc_held * price

        reward = 0.0

        # execute the action
        if action == 1:  # BUY
            if self.balance > 0:
                amount_after_fee = self.balance * (1 - self.trading_fee)
                self.btc_held = amount_after_fee / price
                self.balance = 0
                self.entry_price = price
            else:
                reward = -0.05  # tried to buy but no cash

        elif action == 2:  # SELL
            if self.btc_held > 0:
                proceeds = self.btc_held * price
                self.balance = proceeds * (1 - self.trading_fee)
                self.btc_held = 0
                self.entry_price = 0
            else:
                reward = -0.05  # tried to sell but nothing to sell

        # advance to next candle
        self.step_idx += 1
        done = self.step_idx >= len(self.data) - 1

        next_price = self.data.loc[self.step_idx, "Close"]

        # portfolio value after price moved
        new_value = self.balance + self.btc_held * next_price

        # main reward: did our portfolio go up or down?
        if old_value > 0:
            portfolio_change = (new_value - old_value) / old_value
            reward += portfolio_change * 100

        # small nudge: penalize sitting in cash during significant moves
        # without this the agent just learns to hold cash forever
        if action == 0 and self.btc_held == 0:
            price_move = abs(next_price - price) / price
            if price_move > 0.002:
                reward -= 0.02

        info = {
            "balance": self.balance,
            "btc_held": self.btc_held,
            "portfolio_value": new_value,
            "trade_executed": action in [1, 2] and reward >= -0.04,
        }

        return self._build_observation(), reward, done, False, info