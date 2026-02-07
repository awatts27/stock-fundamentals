"""Data layer — all yfinance calls live here."""

import json
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Baskets
# ---------------------------------------------------------------------------

def load_baskets(path: str = "baskets.json") -> dict[str, list[str]]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def all_tickers(baskets: dict[str, list[str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for members in baskets.values():
        for t in members:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Fetching price history…", ttl=3600)
def fetch_price_history(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        for candidate in ["Close", "Adj Close"]:
            if candidate in data.columns.get_level_values(0):
                data = data[candidate]
                break
        else:
            data = data[data.columns.get_level_values(0)[0]]
    else:
        for col in ["Close", "Adj Close"]:
            if col in data.columns:
                data = data[[col]]
                data.columns = [tickers[0]] if len(tickers) == 1 else data.columns
                break

    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])

    data = data.dropna(how="all")
    data.columns = [str(c) for c in data.columns]
    return data


# ---------------------------------------------------------------------------
# Fundamentals snapshot for a single ticker
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fundamentals(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info or {}

    current_price = info.get("currentPrice")
    high52 = info.get("fiftyTwoWeekHigh")
    market_cap = info.get("marketCap")

    pct_below_high = None
    if current_price and high52 and high52 > 0:
        pct_below_high = round((high52 - current_price) / high52 * 100, 2)

    revenue = info.get("totalRevenue")
    fcf = info.get("freeCashflow")
    fcf_margin = None
    if fcf and revenue and revenue > 0:
        fcf_margin = round(fcf / revenue * 100, 2)

    de = info.get("debtToEquity")
    if de is not None and de < 5:
        de = de * 100

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "summary": info.get("longBusinessSummary") or "",
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "current_price": current_price,
        "market_cap": market_cap,
        "high_52w": high52,
        "low_52w": info.get("fiftyTwoWeekLow"),
        "pct_below_high": pct_below_high,
        "pe_ttm": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg": info.get("pegRatio"),
        "pb": info.get("priceToBook"),
        "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else None,
        "net_margin": round(info.get("profitMargins", 0) * 100, 2) if info.get("profitMargins") else None,
        "roe": round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else None,
        "revenue_growth": round(info.get("revenueGrowth", 0) * 100, 2) if info.get("revenueGrowth") else None,
        "earnings_growth": round(info.get("earningsGrowth", 0) * 100, 2) if info.get("earningsGrowth") else None,
        "debt_to_equity": round(de, 1) if de is not None else None,
        "current_ratio": info.get("currentRatio"),
        "fcf": fcf,
        "fcf_margin": fcf_margin,
        "beta": info.get("beta"),
        "avg_volume": info.get("averageVolume"),
        "short_pct_float": round(info.get("shortPercentOfFloat", 0) * 100, 2) if info.get("shortPercentOfFloat") else None,
        "analyst_target": info.get("targetMeanPrice"),
        "analyst_upside": round((info.get("targetMeanPrice", 0) - current_price) / current_price * 100, 1) if current_price and info.get("targetMeanPrice") else None,
        "held_pct_insiders": round(info.get("heldPercentInsiders", 0) * 100, 2) if info.get("heldPercentInsiders") else None,
        "is_etf": info.get("quoteType") in {"ETF", "MUTUALFUND", "INDEX"},
    }


# ---------------------------------------------------------------------------
# Batch fundamentals
# ---------------------------------------------------------------------------

def fetch_all_fundamentals(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        try:
            rows.append(fetch_fundamentals(tk))
        except Exception:
            rows.append({"ticker": tk, "name": tk})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Momentum & volatility helpers
# ---------------------------------------------------------------------------

def compute_momentum(prices: pd.DataFrame, periods: int) -> pd.Series:
    if prices.empty or len(prices) < periods + 1:
        return pd.Series(dtype=float)
    start = prices.iloc[-(periods + 1)]
    end = prices.iloc[-1]
    return ((end / start) - 1) * 100


def compute_volatility(prices: pd.DataFrame, periods: int) -> pd.Series:
    if prices.empty or len(prices) < periods:
        return pd.Series(dtype=float)
    returns = prices.pct_change().iloc[-periods:]
    return returns.std() * np.sqrt(252) * 100


def compute_drawdown_from_peak(prices: pd.DataFrame) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    peaks = prices.cummax()
    last_price = prices.iloc[-1]
    last_peak = peaks.iloc[-1]
    return ((last_peak - last_price) / last_peak * 100).round(2)


# ---------------------------------------------------------------------------
# Dip context — is the dip market-wide, sector, or stock-specific?
# ---------------------------------------------------------------------------

def classify_dip(
    prices: pd.DataFrame,
    ticker: str,
    basket_tickers: list[str],
    lookback: int = 21,
) -> str:
    """Classify why a stock is down.

    Returns one of:
      'market'  — SPY is also down over the window (broad selloff)
      'sector'  — majority of the basket is down (sector rotation)
      'stock'   — only this stock is down (company-specific, most dangerous)
    """
    if len(prices) < lookback + 1:
        return "unknown"

    def _ret(col):
        if col not in prices.columns:
            return None
        s = prices[col].dropna()
        if len(s) < lookback + 1:
            return None
        return (s.iloc[-1] / s.iloc[-(lookback + 1)] - 1) * 100

    spy_ret = _ret("SPY")
    if spy_ret is not None and spy_ret < -3:
        return "market"

    basket_avail = [t for t in basket_tickers if t in prices.columns and t != ticker]
    if basket_avail:
        basket_rets = [_ret(t) for t in basket_avail]
        basket_rets = [r for r in basket_rets if r is not None]
        if basket_rets:
            down_count = sum(1 for r in basket_rets if r < -3)
            if down_count / len(basket_rets) > 0.5:
                return "sector"

    return "stock"


DIP_LABELS = {
    "market": "Market-wide dip",
    "sector": "Sector dip",
    "stock": "Stock-specific",
    "unknown": "Unknown",
}
