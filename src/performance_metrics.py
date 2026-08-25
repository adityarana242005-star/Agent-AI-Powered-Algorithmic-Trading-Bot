import numpy as np


# Calculate how much money the strategy made
def get_return(start_money, final_money):

    profit = final_money - start_money

    return (profit / start_money) * 100


# Find the biggest fall in the portfolio
def get_max_drawdown(values):

    highest_value = values[0]
    max_drop = 0

    for value in values:

        if value > highest_value:
            highest_value = value

        drop = (
            (highest_value - value)
            / highest_value
        )

        if drop > max_drop:
            max_drop = drop

    return max_drop * 100


# Calculate Sharpe ratio
def get_sharpe_ratio(values):

    values = np.array(values)

    daily_returns = (
        np.diff(values) / values[:-1]
    )

    if len(daily_returns) == 0:
        return 0

    average_return = np.mean(daily_returns)
    risk = np.std(daily_returns)

    if risk == 0:
        return 0

    sharpe = (
        average_return / risk
    ) * np.sqrt(252)

    return sharpe


# Print all results
def print_results(
    start_money,
    final_money,
    portfolio_values,
    trades
):

    total_return = get_return(
        start_money,
        final_money
    )

    max_drawdown = get_max_drawdown(
        portfolio_values
    )

    sharpe = get_sharpe_ratio(
        portfolio_values
    )

    print()
    print("PERFORMANCE RESULTS")
    print(" ")

    print(
        "Starting Money:",
        start_money
    )

    print(
        "Final Money:",
        round(final_money, 2)
    )

    print(
        "Return:",
        round(total_return, 2),
        "%"
    )

    print(
        "Maximum Drawdown:",
        round(max_drawdown, 2),
        "%"
    )

    print(
        "Sharpe Ratio:",
        round(sharpe, 2)
    )

    print(
        "Number of Trades:",
        trades
    )