import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path



# Basic page settings


st.set_page_config(
    page_title="Bitcoin Trading Research",
    page_icon="₿",
    layout="wide"
)


# Project paths


BASE_DIR = Path(__file__).resolve().parent.parent

data_dir = BASE_DIR / "data" / "processed"

test_file = data_dir / "BTC_test.csv"
prediction_file = data_dir / "direction_predictions.csv"
evaluation_file = data_dir / "evaluation_results.csv"



# Small amount of styling


st.markdown(
    """
    <style>

    .main {
        background-color: #f5f5f2;
    }

    h1, h2, h3 {
        color: #111111;
    }

    p, label, div {
        color: #111111;
    }

    .metric-box {
        background-color: white;
        padding: 18px;
        border-radius: 8px;
        border: 1px solid #dddddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# Load files


@st.cache_data
def load_data():

    price_data = pd.read_csv(test_file)

    predictions = pd.read_csv(prediction_file)

    if evaluation_file.exists():
        evaluation = pd.read_csv(evaluation_file)
    else:
        evaluation = pd.DataFrame()

    return price_data, predictions, evaluation


try:
    price_data, predictions, evaluation = load_data()

except FileNotFoundError as error:

    st.error("Required project data could not be found.")

    st.write(error)

    st.stop()



# Header


st.title("Bitcoin Trading Research Dashboard")

st.write(
    "A simple view of the price prediction model, "
    "PPO trading strategy and historical performance."
)



# Sidebar


st.sidebar.title("Project")

st.sidebar.write(
    "Bitcoin market analysis using "
    "machine learning and reinforcement learning."
)

st.sidebar.divider()

st.sidebar.write("Prediction model")
st.sidebar.write("Random Forest")

st.sidebar.write("Trading model")
st.sidebar.write("PPO")



# Current market information


latest_price = price_data["Close"].iloc[-1]

previous_price = price_data["Close"].iloc[-2]

price_change = (
    (latest_price - previous_price)
    / previous_price
) * 100


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Latest BTC Price",
        f"${latest_price:,.2f}"
    )

with col2:
    st.metric(
        "Last Candle Change",
        f"{price_change:.2f}%"
    )

with col3:
    st.metric(
        "Test Data",
        f"{len(price_data):,} candles"
    )


st.divider()



# BTC price chart


st.subheader("Bitcoin Price")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=price_data.index,
        y=price_data["Close"],
        mode="lines",
        name="BTC Price"
    )
)

fig.update_layout(
    height=420,
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    ),
    xaxis_title="Test Period",
    yaxis_title="Price"
)

st.plotly_chart(
    fig,
    use_container_width=True
)



# Prediction section


st.subheader("Price Direction Prediction")

prediction_accuracy = (
    (predictions["Actual"] == predictions["Predicted"])
    .mean()
    * 100
)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "Direction Accuracy",
        f"{prediction_accuracy:.2f}%"
    )

with col3:

    latest_prediction = predictions[
        "Predicted_Direction"
    ].iloc[-1]

    st.metric(
        "Latest Direction",
        latest_prediction
    )


# Prediction chart

prediction_chart = predictions.tail(500).copy()

prediction_chart["Actual_Price"] = (
    price_data["Close"]
    .tail(len(prediction_chart))
    .values
)


fig_prediction = go.Figure()

fig_prediction.add_trace(
    go.Scatter(
        y=prediction_chart["Actual_Price"],
        mode="lines",
        name="BTC Price"
    )
)

st.plotly_chart(
    fig_prediction,
    use_container_width=True
)



# PPO performance


st.divider()

st.subheader("PPO Trading Strategy")

if not evaluation.empty:

    # Use the first row because evaluation_results
    # contains the final strategy statistics.

    row = evaluation.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if "Starting_Money" in evaluation.columns:
            st.metric(
                "Starting Money",
                f"${row['Starting_Money']:,.2f}"
            )

    with col2:
        if "Final_Money" in evaluation.columns:
            st.metric(
                "Final Money",
                f"${row['Final_Money']:,.2f}"
            )

    with col3:
        if "Return" in evaluation.columns:
            st.metric(
                "Strategy Return",
                f"{row['Return']:.2f}%"
            )

    with col4:
        if "Number_of_Trades" in evaluation.columns:
            st.metric(
                "Trades",
                int(row["Number_of_Trades"])
            )

else:

    st.info(
        "Evaluation results have not been generated yet."
    )


# Buy and hold comparison


st.subheader("Strategy vs Buy & Hold")

if "Starting_Money" in evaluation.columns:

    starting_money = evaluation.iloc[0]["Starting_Money"]

    if "Final_Money" in evaluation.columns:

        final_money = evaluation.iloc[0]["Final_Money"]

        btc_return = (
            (
                price_data["Close"].iloc[-1]
                / price_data["Close"].iloc[0]
            ) - 1
        ) * 100

        buy_hold_money = (
            starting_money
            * (1 + btc_return / 100)
        )

        comparison = pd.DataFrame(
            {
                "Strategy": [
                    final_money
                ],
                "Buy & Hold": [
                    buy_hold_money
                ]
            }
        )

        st.bar_chart(
            comparison.T
        )

        st.write(
            f"Bitcoin buy & hold return: "
            f"**{btc_return:.2f}%**"
        )



# Prediction summary

st.divider()

st.subheader("Prediction Summary")

up_count = (
    predictions["Predicted_Direction"] == "UP"
).sum()

down_count = (
    predictions["Predicted_Direction"] == "DOWN"
).sum()

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Predicted UP",
        int(up_count)
    )

with col2:

    st.metric(
        "Predicted DOWN",
        int(down_count)
    )



# Simple calculator


st.divider()

st.subheader("Position Calculator")

st.write(
    "Estimate the result of a simple BTC position."
)

money = st.number_input(
    "Investment",
    min_value=0.0,
    value=10000.0,
    step=500.0
)

entry_price = st.number_input(
    "Entry BTC Price",
    min_value=0.0,
    value=float(latest_price),
    step=100.0
)

exit_price = st.number_input(
    "Exit BTC Price",
    min_value=0.0,
    value=float(latest_price),
    step=100.0
)


if entry_price > 0:

    btc_amount = money / entry_price

    final_value = btc_amount * exit_price

    profit = final_value - money

    return_percent = (
        profit / money
    ) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "BTC Amount",
            f"{btc_amount:.6f}"
        )

    with col2:
        st.metric(
            "Final Value",
            f"${final_value:,.2f}"
        )

    with col3:
        st.metric(
            "Profit / Loss",
            f"${profit:,.2f}",
            f"{return_percent:.2f}%"
        )



# Footer


st.divider()

st.caption(
    "Educational research project. "
    "Historical results do not guarantee future performance."
)