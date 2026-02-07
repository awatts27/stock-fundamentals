"""Pullback detection — find dips worth investigating.

Only called AFTER quality filtering, so every stock that reaches this
module is already a real business. We just need to know: is it cheap
relative to where it was recently?
"""

import pandas as pd
import numpy as np


def detect_pullbacks(
    prices: pd.DataFrame,
    threshold_pct: float = 20.0,
    lookback_days: int = 63,
) -> pd.DataFrame:
    """
    Flag stocks that have dropped >= threshold_pct from their rolling high
    within the lookback window.

    Returns a DataFrame with one row per triggered ticker:
      ticker, current_price, rolling_high, drop_pct
    """
    if prices.empty:
        return pd.DataFrame()

    window = min(lookback_days, len(prices))
    recent = prices.iloc[-window:]
    rolling_high = recent.max()
    current = prices.iloc[-1]
    drop_pct = ((rolling_high - current) / rolling_high * 100).round(2)

    rows = []
    for ticker in prices.columns:
        d = drop_pct.get(ticker)
        if d is None or pd.isna(d):
            continue
        if d < threshold_pct:
            continue

        rows.append({
            "ticker": ticker,
            "current_price": round(float(current.get(ticker, 0)), 2),
            "rolling_high": round(float(rolling_high.get(ticker, 0)), 2),
            "drop_pct": round(float(d), 1),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("drop_pct", ascending=False)
    return df
