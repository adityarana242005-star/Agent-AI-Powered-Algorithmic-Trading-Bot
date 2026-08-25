
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path



# Page setup



st.set_page_config(
    page_title="BTC Strategy Lab",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Project paths


ROOT = Path(__file__).resolve().parents[1]

data_dir = ROOT / "data" / "processed"

data_file = data_dir / "BTC_test.csv"
results_file = data_dir / "evaluation_results.csv"
summary_file = data_dir / "evaluation_summary.csv"
prediction_file = data_dir / "direction_predictions.csv"
prediction_summary_file = data_dir / "prediction_summary.csv"

model_dir = ROOT / "models"


# Styling



st.markdown("""
<style>

    /* Main page */

    .stApp {
        background-color: #f2f3f5;
        color: #111111;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        background-color: #171717;
    }

    section[data-testid="stSidebar"] * {
        color: #f5f5f5 !important;
    }


    /* Header */

    .brand {
        font-size: 30px;
        font-weight: 750;
        color: #111111;
        letter-spacing: -1px;
    }

    .subtitle {
        color: #666666;
        font-size: 15px;
        margin-top: -8px;
        margin-bottom: 25px;
    }


    /* Cards */

    .card {
        background: #ffffff;
        border: 1px solid #dedede;
        border-radius: 14px;
        padding: 20px;
        min-height: 120px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .card-title {
        font-size: 13px;
        color: #666666;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 28px;
        font-weight: 700;
        color: #111111;
    }

    .card-small {
        font-size: 13px;
        color: #777777;
        margin-top: 5px;
    }

    .card-green {
        color: #0d9f4f;
        font-size: 28px;
        font-weight: 700;
    }

    .card-red {
        color: #d93025;
        font-size: 28px;
        font-weight: 700;
    }


    /* Section headings */

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #111111;
        margin-top: 25px;
        margin-bottom: 12px;
    }


    /* Status */

    .status {
        background: #ffffff;
        border: 1px solid #dedede;
        border-radius: 10px;
        padding: 12px 15px;
        color: #222222;
        font-size: 14px;
    }


    /* Calculator */

    .calculator {
        background: #ffffff;
        border: 1px solid #dedede;
        border-radius: 14px;
        padding: 22px;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #777777;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #dddddd;
    }

</style>
""", unsafe_allow_html=True)



# Load data



@st.cache_data
def load_data():

    market_data = pd.read_csv(data_file)
    evaluation = pd.read_csv(results_file)

    # Load summary (new format)
    if summary_file.exists():
        summary = pd.read_csv(summary_file)
    else:
        summary = pd.DataFrame()

    # Load predictions
    if prediction_file.exists():
        predictions = pd.read_csv(prediction_file)
    else:
        predictions = pd.DataFrame()

    # Load prediction summary
    if prediction_summary_file.exists():
        pred_summary = pd.read_csv(prediction_summary_file)
    else:
        pred_summary = pd.DataFrame()

    return market_data, evaluation, summary, predictions, pred_summary


try:

    data, evaluation, summary, predictions, pred_summary = load_data()

except Exception as error:

    st.error("Could not load the project data.")

    st.write(error)

    st.stop()



# Basic calculations



starting_money = 10000

portfolio_values = evaluation["Portfolio_Value"].dropna().values

if len(portfolio_values) > 0:
    final_money = float(portfolio_values[-1])
else:
    final_money = starting_money


strategy_return = (
    (final_money - starting_money)
    / starting_money
) * 100


# Buy and hold

first_price = float(data["Close"].iloc[0])
last_price = float(data["Close"].iloc[-1])

buy_hold_return = (
    (last_price - first_price)
    / first_price
) * 100


# Maximum drawdown

if len(portfolio_values) > 0:

    running_high = np.maximum.accumulate(
        portfolio_values
    )

    drawdowns = (
        (running_high - portfolio_values)
        / running_high
    )

    max_drawdown = float(
        np.max(drawdowns) * 100
    )

else:

    max_drawdown = 0


# Sharpe ratio

if len(portfolio_values) > 1:

    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    if np.std(returns) != 0:

        sharpe_ratio = (
            np.mean(returns)
            / np.std(returns)
        ) * np.sqrt(252)

    else:

        sharpe_ratio = 0

else:

    sharpe_ratio = 0


# Action counts


action_counts = {
    0: 0,
    1: 0,
    2: 0
}

if "Action" in evaluation.columns:

    counts = evaluation["Action"].value_counts()

    for action in [0, 1, 2]:

        action_counts[action] = int(
            counts.get(action, 0)
        )


hold_count = action_counts[0]
buy_count = action_counts[1]
sell_count = action_counts[2]

trade_count = buy_count + sell_count



# Sidebar



with st.sidebar:

    st.markdown(
        "## ₿ BTC Strategy Lab"
    )

    st.markdown(
        "Reinforcement learning research terminal"
    )

    st.divider()

    st.markdown("### Dataset")

    st.write(
        f"Test candles: **{len(data):,}**"
    )

    st.write(
        "Timeframe: **15 minutes**"
    )

    st.write(
        "Asset: **Bitcoin / USDT**"
    )

    st.divider()

    st.markdown("### Models")

    st.write(
        "Prediction: **Random Forest**"
    )

    st.write(
        "Trading: **PPO**"
    )

    st.write(
        "Framework: **Stable-Baselines3**"
    )

    st.write(
        "Environment: **Gymnasium**"
    )

    st.divider()

    # Show prediction accuracy in sidebar
    if not predictions.empty:
        accuracy = (
            (predictions["Actual"] == predictions["Predicted"])
            .mean() * 100
        )
        st.write(
            f"Direction Accuracy: **{accuracy:.2f}%**"
        )

    st.write(
        f"Strategy Return: **{strategy_return:.2f}%**"
    )

    st.write(
        f"Buy & Hold: **{buy_hold_return:.2f}%**"
    )

    st.divider()

    st.caption(
        "This dashboard reports the actual "
        "results generated by the trading project."
    )



# Header



st.markdown(
    '<div class="brand">BTC Strategy Lab</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Bitcoin reinforcement-learning strategy research terminal'
    '</div>',
    unsafe_allow_html=True
)


# Model status

st.markdown(
    '<div class="status">'
    '● Model evaluation loaded &nbsp; | &nbsp; '
    'PPO + Random Forest &nbsp; | &nbsp; 15-minute BTC data'
    '</div>',
    unsafe_allow_html=True
)



# Top metrics



st.markdown(
    '<div class="section-title">Strategy Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">BTC Price</div>
            <div class="card-value">
                ${last_price:,.2f}
            </div>
            <div class="card-small">
                Last test-set price
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    return_color = "card-green" if strategy_return >= 0 else "card-red"

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Strategy Return</div>
            <div class="{return_color}">
                {strategy_return:.2f}%
            </div>
            <div class="card-small">
                Starting capital: $10,000
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    bh_color = "card-green" if buy_hold_return >= 0 else "card-red"

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Buy & Hold</div>
            <div class="{bh_color}">
                {buy_hold_return:.2f}%
            </div>
            <div class="card-small">
                Benchmark return
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Trades</div>
            <div class="card-value">
                {trade_count:,}
            </div>
            <div class="card-small">
                BUY + SELL actions
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



# Price Direction Prediction


if not predictions.empty:

    st.markdown(
        '<div class="section-title">Price Direction Prediction</div>',
        unsafe_allow_html=True
    )

    prediction_accuracy = (
        (predictions["Actual"] == predictions["Predicted"])
        .mean() * 100
    )

    latest_prediction = predictions[
        "Predicted_Direction"
    ].iloc[-1]

    up_count = (predictions["Predicted_Direction"] == "UP").sum()
    down_count = (predictions["Predicted_Direction"] == "DOWN").sum()

    pc1, pc2, pc3, pc4 = st.columns(4)

    with pc1:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Model</div>
                <div class="card-value" style="font-size:22px;">
                    Random Forest
                </div>
                <div class="card-small">
                    Direction classifier
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with pc2:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Direction Accuracy</div>
                <div class="card-value">
                    {prediction_accuracy:.2f}%
                </div>
                <div class="card-small">
                    {int((predictions["Actual"] == predictions["Predicted"]).sum()):,} correct
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with pc3:
        pred_color = "card-green" if latest_prediction == "UP" else "card-red"
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Latest Prediction</div>
                <div class="{pred_color}">
                    {latest_prediction}
                </div>
                <div class="card-small">
                    Most recent candle
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with pc4:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Predictions</div>
                <div class="card-value" style="font-size:20px;">
                    ↑ {up_count:,} &nbsp; ↓ {down_count:,}
                </div>
                <div class="card-small">
                    UP vs DOWN calls
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )



# Price chart



st.markdown(
    '<div class="section-title">Market Performance</div>',
    unsafe_allow_html=True
)


# Reduce number of points for a smoother browser chart

chart_data = data.copy()

if len(chart_data) > 3000:

    step = len(chart_data) // 3000

    chart_data = chart_data.iloc[::step].copy()


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=chart_data.index,
        y=chart_data["Close"],
        name="BTC Price",
        mode="lines",
        line=dict(
            width=2
        )
    )
)


if "SMA_20" in chart_data.columns:

    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["SMA_20"],
            name="SMA 20",
            mode="lines",
            line=dict(
                width=1
            )
        )
    )


fig.update_layout(
    height=430,
    margin=dict(
        l=10,
        r=10,
        t=30,
        b=10
    ),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    hovermode="x unified",
    xaxis=dict(
        showgrid=False
    ),
    yaxis=dict(
        gridcolor="#eeeeee",
        title=""
    ),
    legend=dict(
        orientation="h",
        y=1.08
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)



# Portfolio + AI actions


left, right = st.columns([2, 1])


with left:

    st.markdown(
        '<div class="section-title">'
        'Portfolio Value'
        '</div>',
        unsafe_allow_html=True
    )

    portfolio_chart = go.Figure()

    portfolio_chart.add_trace(
        go.Scatter(
            y=portfolio_values,
            mode="lines",
            name="Portfolio",
            line=dict(
                width=2
            )
        )
    )

    portfolio_chart.add_hline(
        y=starting_money,
        line_dash="dot",
        annotation_text="Starting capital"
    )

    portfolio_chart.update_layout(
        height=360,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        yaxis=dict(
            tickprefix="$",
            gridcolor="#eeeeee"
        ),
        xaxis=dict(
            title="Test step",
            showgrid=False
        )
    )

    st.plotly_chart(
        portfolio_chart,
        use_container_width=True
    )


with right:

    st.markdown(
        '<div class="section-title">'
        'AI Decisions'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="card">

            <div class="card-title">BUY</div>
            <div class="card-green">{buy_count:,}</div>

            <hr>

            <div class="card-title">HOLD</div>
            <div class="card-value">{hold_count:,}</div>

            <hr>

            <div class="card-title">SELL</div>
            <div class="card-red">{sell_count:,}</div>

        </div>
        """,
        unsafe_allow_html=True
    )



# Risk section



st.markdown(
    '<div class="section-title">Risk & Performance</div>',
    unsafe_allow_html=True
)

r1, r2, r3, r4 = st.columns(4)


with r1:
    st.metric(
        "Final Portfolio",
        f"${final_money:,.2f}"
    )


with r2:
    st.metric(
        "Maximum Drawdown",
        f"{max_drawdown:.2f}%"
    )


with r3:
    st.metric(
        "Sharpe Ratio",
        f"{sharpe_ratio:.2f}"
    )


with r4:
    st.metric(
        "Test Candles",
        f"{len(data):,}"
    )



# Position simulator



st.markdown(
    '<div class="section-title">'
    'Position Simulator'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="calculator">
    Estimate the result of a simple BTC position
    before placing a trade.
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)


with c1:

    investment = st.number_input(
        "Investment ($)",
        min_value=1.0,
        value=10000.0,
        step=500.0
    )


with c2:

    entry_price = st.number_input(
        "Entry Price ($)",
        min_value=0.01,
        value=float(first_price),
        step=100.0
    )


with c3:

    exit_price = st.number_input(
        "Exit Price ($)",
        min_value=0.01,
        value=float(last_price),
        step=100.0
    )


with c4:

    fee_percent = st.number_input(
        "Trading Fee (%)",
        min_value=0.0,
        value=0.10,
        step=0.05
    )


# Position calculation

btc_amount = investment / entry_price

buy_fee = investment * (
    fee_percent / 100
)

gross_exit_value = btc_amount * exit_price

sell_fee = gross_exit_value * (
    fee_percent / 100
)

net_exit_value = (
    gross_exit_value
    - sell_fee
)

net_profit = (
    net_exit_value
    - investment
    - buy_fee
)

position_return = (
    net_profit / investment
) * 100


st.markdown("")


p1, p2, p3, p4 = st.columns(4)


with p1:

    st.metric(
        "BTC Quantity",
        f"{btc_amount:.6f}"
    )


with p2:

    st.metric(
        "Exit Value",
        f"${net_exit_value:,.2f}"
    )


with p3:

    st.metric(
        "Net P/L",
        f"${net_profit:,.2f}"
    )


with p4:

    st.metric(
        "Position Return",
        f"{position_return:.2f}%"
    )



# Live Prediction Section



st.markdown(
    '<div class="section-title">Live Prediction Engine</div>',
    unsafe_allow_html=True
)


# Load saved model
rf_model_path = model_dir / "rf_model.joblib"
feature_list_path = model_dir / "feature_list.joblib"

if rf_model_path.exists() and feature_list_path.exists():

    @st.cache_resource
    def load_model():
        model = joblib.load(rf_model_path)
        features = joblib.load(feature_list_path)
        return model, features

    rf_model, feature_list = load_model()

    st.markdown(
        '<div class="status">'
        '● ML Model loaded &nbsp; | &nbsp; '
        'Random Forest &nbsp; | &nbsp; '
        f'{len(feature_list)} features'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("")

    # Let user choose how many recent candles to predict
    num_candles = st.slider(
        "Number of recent candles to predict",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )

    # Get the most recent candles from test data
    recent_data = data.tail(num_candles + 1).copy()

    # Prepare features for prediction
    available_feats = [
        f for f in feature_list
        if f in recent_data.columns
    ]

    if len(available_feats) > 0:

        X_recent = recent_data[available_feats].iloc[:-1].copy()
        X_recent = X_recent.replace([np.inf, -np.inf], np.nan)
        X_recent = X_recent.fillna(0)

        # Run live predictions
        live_predictions = rf_model.predict(X_recent)
        live_probabilities = rf_model.predict_proba(X_recent)

        # Actual directions (next candle compared to current)
        closes = recent_data["Close"].values
        actual_directions = []
        for i in range(len(closes) - 1):
            if closes[i + 1] > closes[i]:
                actual_directions.append(1)
            else:
                actual_directions.append(0)

        # Build results table
        pred_table = pd.DataFrame({
            "BTC Price": [f"${c:,.2f}" for c in closes[:-1]],
            "Actual": ["↑ UP" if d == 1 else "↓ DOWN" for d in actual_directions],
            "Predicted": ["↑ UP" if p == 1 else "↓ DOWN" for p in live_predictions],
            "Confidence": [f"{max(prob) * 100:.1f}%" for prob in live_probabilities],
            "Correct": ["Match" if a == p else "Mismatch" for a, p in zip(actual_directions, live_predictions)]
        })

        # Calculate live accuracy
        live_correct = sum(
            1 for a, p in zip(actual_directions, live_predictions)
            if a == p
        )
        live_accuracy = (live_correct / len(actual_directions)) * 100

        # Show live accuracy metrics
        lc1, lc2, lc3 = st.columns(3)

        with lc1:
            st.metric(
                "Live Predictions",
                f"{len(live_predictions)} candles"
            )

        with lc2:
            st.metric(
                "Live Accuracy",
                f"{live_accuracy:.1f}%"
            )

        with lc3:
            st.metric(
                "Correct / Total",
                f"{live_correct} / {len(actual_directions)}"
            )

        # Show predictions table
        st.dataframe(
            pred_table,
            use_container_width=True,
            hide_index=True
        )

        # Prediction accuracy chart (rolling)
        if len(actual_directions) >= 5:
            correct_series = pd.Series([
                1 if a == p else 0
                for a, p in zip(actual_directions, live_predictions)
            ])

            rolling_acc = correct_series.rolling(
                min(5, len(correct_series)),
                min_periods=1
            ).mean() * 100

            acc_fig = go.Figure()

            acc_fig.add_trace(
                go.Scatter(
                    y=rolling_acc.values,
                    mode="lines+markers",
                    name="Rolling Accuracy",
                    line=dict(width=2)
                )
            )

            acc_fig.add_hline(
                y=50,
                line_dash="dot",
                annotation_text="Random baseline (50%)"
            )

            acc_fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=20, b=10),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                yaxis=dict(
                    title="Accuracy %",
                    gridcolor="#eeeeee",
                    range=[0, 100]
                ),
                xaxis=dict(
                    title="Candle",
                    showgrid=False
                )
            )

            st.plotly_chart(
                acc_fig,
                use_container_width=True
            )

    else:
        st.warning("Feature columns not found in test data.")

else:
    st.info(
        "ML model not found. Run train_prediction_model.py first "
        "to save the model for live predictions."
    )


# =====================
# Project notes
# =====================


st.markdown(
    '<div class="section-title">Model Notes</div>',
    unsafe_allow_html=True
)


note1, note2 = st.columns(2)


with note1:

    st.info(
        "The PPO agent is evaluated on unseen test data. "
        "The dashboard does not modify the model or its results."
    )


with note2:

    st.warning(
        f"Current baseline: the agent made {trade_count} trades "
        f"and returned {strategy_return:.2f}%, while Buy & Hold "
        f"returned {buy_hold_return:.2f}%."
    )


# =====================
# Footer
# =====================


st.markdown(
    """
    <div class="footer">
        BTC Strategy Lab · PPO Trading Research ·
        Built for strategy analysis and experimentation
    </div>
    """,
    unsafe_allow_html=True
)