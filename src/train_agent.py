"""
Train PPO Trading Agent
========================
Trains the PPO agent on historical Bitcoin training data.

Improvements:
- 2M timesteps (was 500K)
- Entropy coefficient for exploration (was 0)
- Better hyperparameters
- Progress logging
"""

import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trading_environment import BitcoinTradingEnv


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data" / "processed"
model_dir = BASE_DIR / "models"


# Ensure model directory exists
model_dir.mkdir(exist_ok=True)


# =====================
# Progress callback
# =====================


class ProgressCallback(BaseCallback):
    """Prints progress every N steps."""

    def __init__(self, print_freq=50000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self):
        if self.num_timesteps % self.print_freq == 0:
            print(
                f"  Step: {self.num_timesteps:,}"
            )
        return True


# =====================
# Load training data
# =====================


data = pd.read_csv(data_dir / "BTC_train.csv")

print("Training data loaded!")
print(f"Rows: {len(data)}")


# =====================
# Create environment
# =====================


env = BitcoinTradingEnv(data)

print("Trading environment ready!")
print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.n} actions")


# =====================
# Create PPO model
# =====================


model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,         # Entropy bonus — encourages exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    policy_kwargs=dict(
        net_arch=dict(
            pi=[128, 128],  # Policy network
            vf=[128, 128]   # Value network
        )
    )
)


print("\nTraining started (2,000,000 timesteps)...")
print("This will take a few minutes.\n")


# =====================
# Train
# =====================


model.learn(
    total_timesteps=2_000_000,
    callback=ProgressCallback(print_freq=100000)
)

print("\nTraining completed!")


# =====================
# Save model
# =====================


model.save(str(model_dir / "bitcoin_ppo"))

print(f"Model saved to: {model_dir / 'bitcoin_ppo.zip'}")