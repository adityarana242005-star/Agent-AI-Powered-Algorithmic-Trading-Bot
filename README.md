# 🤖 Agent AI-Powered Algorithmic Trading Bot

An autonomous Bitcoin (BTC/USDT) algorithmic trading framework built using **Deep Reinforcement Learning (PPO)** and ensemble **Supervised Machine Learning (Random Forest & Gradient Boosting)** on 15-minute candlestick market data.

Includes a custom **Gymnasium** trading environment with realistic fee friction (0.1%), comprehensive technical indicator feature engineering, and an interactive **Streamlit & Plotly** research dashboard for live strategy backtesting and analytics.

---

## 🌟 Key Features

* **Custom Gymnasium Trading Environment (`BitcoinTradingEnv`)**: Simulates 15-minute BTC/USDT candlestick trading with a 10-dimensional state vector (RSI, MACD, SMA Crossovers, Volatility, Lags, Portfolio Ratio, Unrealized PnL) and 0.1% trade execution fees.
* **Deep Reinforcement Learning Agent**: Trained a Proximal Policy Optimization (**PPO**) agent using `Stable-Baselines3` over **2M+ timesteps** with dynamic entropy exploration and custom MLP architecture `[128, 128]`.
* **Directional ML Prediction Model**: Hybrid supervised model using **Random Forest & Gradient Boosting** with 15 engineered technical features to forecast next-candle price movements.
* **Backtesting & Financial Analytics**: Evaluates strategy performance via Sharpe Ratio, Max Drawdown, Win Rate, and Total Return relative to a Buy-and-Hold benchmark.
* **Interactive Strategy Dashboard (`BTC Strategy Lab`)**: Streamlit web interface with Plotly interactive candlestick charts, live execution logs, direction prediction tables, and feature importance visualizers.

---

## 🛠️ Tech Stack

* **Language**: Python 3.x
* **Deep Reinforcement Learning**: `Stable-Baselines3` (PPO)
* **RL Environment**: `Gymnasium`
* **Supervised Machine Learning**: `Scikit-Learn` (`RandomForestClassifier`, `HistGradientBoostingClassifier`)
* **Data Engineering & Analysis**: `Pandas`, `NumPy`
* **Dashboard & Visualization**: `Streamlit`, `Plotly`
* **Model Persistence**: `Joblib`

---

## 📂 Repository Structure

```
├── dashboard/
│   └── app.py                     # Streamlit interactive strategy lab dashboard
├── src/
│   ├── data_loader.py             # Historical BTC data loading & cleaning
│   ├── feature_engineering.py     # Technical indicator computation (RSI, MACD, SMA)
│   ├── split_data.py              # Train/test split utility
│   ├── trading_environment.py     # Custom Gymnasium BTC trading environment
│   ├── train_agent.py             # PPO agent training script
│   ├── evaluate_agent.py          # Backtesting and performance evaluation script
│   ├── train_prediction_model.py  # Random Forest / Gradient Boosting classifier training
│   ├── price_prediction.py        # Direction prediction inference pipeline
│   └── performance_metrics.py     # Sharpe ratio, Drawdown & Win rate analytics
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules for data & model artifacts
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/adityarana242005-star/Agent-AI-Powered-Algorithmic-Trading-Bot.git
cd Agent-AI-Powered-Algorithmic-Trading-Bot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Or install core dependencies directly:
pip install stable-baselines3 gymnasium pandas numpy scikit-learn streamlit plotly joblib
```

### 3. Run the Interactive Strategy Dashboard
```bash
streamlit run dashboard/app.py
```

### 4. Train Models
```bash
# Feature Engineering & Splitting
python src/feature_engineering.py
python src/split_data.py

# Train Directional Prediction Model
python src/train_prediction_model.py

# Train Deep RL Trading Agent (PPO)
python src/train_agent.py

# Evaluate Agent Backtest Performance
python src/evaluate_agent.py
```

---

## 📊 Backtest Performance Metrics

* **Evaluation Benchmark**: Buy & Hold strategy vs. DRL PPO Agent
* **Key Metrics Analyzed**:
  * **Sharpe Ratio**: Risk-adjusted return performance
  * **Maximum Drawdown (MDD)**: Peak-to-trough portfolio loss metric
  * **Win Rate**: Percentage of profitable trades executed
  * **Cumulative Return**: Total return percentage over the test dataset

---

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
