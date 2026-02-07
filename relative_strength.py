"""Relative strength — compare baskets and stocks against SPY."""

import pandas as pd
import numpy as np


def basket_returns(
    prices: pd.DataFrame,
    baskets: dict[str, list[str]],
    periods: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Equal-weighted return for each basket over multiple lookback windows.
    Returns a DataFrame: rows = basket names, columns = period labels.
    """
    if periods is None:
        periods = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}

    rows = []
    for name, tickers in baskets.items():
        available = [t for t in tickers if t in prices.columns]
        if not available:
            continue
        basket_prices = prices[available]
        row = {"basket": name, "stocks": len(available)}
        for label, days in periods.items():
            if len(basket_prices) < days + 1:
                row[label] = None
                continue
            start = basket_prices.iloc[-(days + 1)]
            end = basket_prices.iloc[-1]
            rets = ((end / start) - 1) * 100
            row[label] = round(rets.mean(), 2)
        rows.append(row)

    return pd.DataFrame(rows)


def stock_vs_basket(
    prices: pd.DataFrame,
    ticker: str,
    basket_tickers: list[str],
    period_days: int = 63,
) -> dict:
    """Compare a single stock's return to its basket's equal-weighted return."""
    if ticker not in prices.columns or len(prices) < period_days + 1:
        return {}

    available = [t for t in basket_tickers if t in prices.columns and t != ticker]
    if not available:
        return {}

    start_idx = -(period_days + 1)
    stock_ret = (prices[ticker].iloc[-1] / prices[ticker].iloc[start_idx] - 1) * 100
    basket_rets = []
    for t in available:
        r = (prices[t].iloc[-1] / prices[t].iloc[start_idx] - 1) * 100
        basket_rets.append(r)
    basket_avg = np.mean(basket_rets)

    return {
        "ticker": ticker,
        "stock_return": round(stock_ret, 2),
        "basket_return": round(basket_avg, 2),
        "spread": round(stock_ret - basket_avg, 2),
    }


def vs_spy(
    prices: pd.DataFrame,
    periods: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Return each ticker's return minus SPY's return for each period."""
    if "SPY" not in prices.columns:
        return pd.DataFrame()
    if periods is None:
        periods = {"1M": 21, "3M": 63, "6M": 126}

    rows = []
    for ticker in prices.columns:
        if ticker == "SPY":
            continue
        row = {"ticker": ticker}
        for label, days in periods.items():
            if len(prices) < days + 1:
                row[label] = None
                continue
            t_ret = (prices[ticker].iloc[-1] / prices[ticker].iloc[-(days + 1)] - 1) * 100
            s_ret = (prices["SPY"].iloc[-1] / prices["SPY"].iloc[-(days + 1)] - 1) * 100
            row[label] = round(t_ret - s_ret, 2)
        rows.append(row)

    return pd.DataFrame(rows)


def basket_cumulative_returns(
    prices: pd.DataFrame,
    baskets: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Daily cumulative equal-weighted return for each basket.
    Returns a DataFrame with Date index and one column per basket.
    """
    result = pd.DataFrame(index=prices.index)
    for name, tickers in baskets.items():
        available = [t for t in tickers if t in prices.columns]
        if not available:
            continue
        basket_prices = prices[available]
        normalized = basket_prices / basket_prices.iloc[0]
        result[name] = (normalized.mean(axis=1) - 1) * 100

    if "SPY" in prices.columns:
        spy_norm = prices["SPY"] / prices["SPY"].iloc[0]
        result["SPY"] = (spy_norm - 1) * 100

    return result
