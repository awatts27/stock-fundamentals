"""Pullback detection — find dips in stocks that are otherwise healthy."""

import pandas as pd
import numpy as np


def detect_pullbacks(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    threshold_pct: float = 15.0,
    lookback_days: int = 63,
) -> pd.DataFrame:
    """
    A pullback is flagged when:
      1. The stock has dropped >= threshold_pct from its rolling high
         within the lookback window.
      2. The drop is recent (happened in the last 5 trading days).

    Returns a DataFrame with one row per triggered ticker.
    """
    if prices.empty:
        return pd.DataFrame()

    window = min(lookback_days, len(prices))
    recent = prices.iloc[-window:]
    rolling_high = recent.max()
    current = prices.iloc[-1]
    drop_pct = ((rolling_high - current) / rolling_high * 100).round(2)

    # Check that the drop is fresh — price was closer to the high 5 days ago
    if len(prices) > 5:
        price_5d_ago = prices.iloc[-6]
        drop_5d_ago = ((rolling_high - price_5d_ago) / rolling_high * 100).round(2)
    else:
        drop_5d_ago = pd.Series(0, index=prices.columns)

    alerts = []
    for ticker in prices.columns:
        d = drop_pct.get(ticker)
        d5 = drop_5d_ago.get(ticker)
        if d is None or pd.isna(d):
            continue
        if d < threshold_pct:
            continue
        # The drop should have worsened in the last 5 days (fresh pullback)
        is_fresh = d5 is not None and not pd.isna(d5) and d > d5 - 3
        if not is_fresh:
            continue

        alerts.append({
            "ticker": ticker,
            "current_price": current.get(ticker),
            "rolling_high": rolling_high.get(ticker),
            "drop_pct": d,
            "lookback_days": lookback_days,
        })

    return pd.DataFrame(alerts)


def detect_pullbacks_multi_threshold(
    prices: pd.DataFrame,
    thresholds: list[float] | None = None,
    lookback_days: int = 63,
) -> pd.DataFrame:
    """Run pullback detection at multiple severity levels."""
    if thresholds is None:
        thresholds = [10.0, 15.0, 25.0]

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
        severity = "none"
        for t in sorted(thresholds, reverse=True):
            if d >= t:
                severity = f">={int(t)}%"
                break
        if severity == "none":
            continue

        rows.append({
            "ticker": ticker,
            "current_price": round(float(current.get(ticker, 0)), 2),
            "rolling_high": round(float(rolling_high.get(ticker, 0)), 2),
            "drop_pct": round(float(d), 1),
            "severity": severity,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("drop_pct", ascending=False)
    return df
