"""Stock Watchlist v2 — Buy-the-dip alert system for thematic baskets."""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from data import (
    load_baskets,
    all_tickers,
    fetch_price_history,
    fetch_fundamentals,
    fetch_all_fundamentals,
    compute_momentum,
    compute_volatility,
    compute_drawdown_from_peak,
)
from quality import run_quality_gate, passes_quality, quality_summary
from pullbacks import detect_pullbacks_multi_threshold
from relative_strength import (
    basket_returns,
    basket_cumulative_returns,
    vs_spy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_cap(val) -> str:
    if val is None:
        return "N/A"
    if val >= 1_000_000_000_000:
        return f"${val / 1_000_000_000_000:.1f}T"
    if val >= 1_000_000_000:
        return f"${val / 1_000_000_000:.1f}B"
    if val >= 1_000_000:
        return f"${val / 1_000_000:.0f}M"
    return f"${val:,.0f}"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Stock Watchlist", layout="wide")
st.title("Stock Watchlist")
st.caption("Buy-the-dip alerts for your thematic baskets")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
baskets = load_baskets()
if not baskets:
    st.error("No baskets.json found.")
    st.stop()

tab_alerts, tab_baskets, tab_detail, tab_settings = st.tabs([
    "Alerts",
    "Basket Overview",
    "Stock Detail",
    "Baskets",
])

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
pullback_threshold = st.sidebar.slider(
    "Pullback alert threshold %", 5, 40, 15, 5,
    help="Alert when a stock drops this much from its rolling high",
)
lookback_days = st.sidebar.slider(
    "Lookback window (trading days)", 21, 126, 63, 21,
    help="How far back to look for the rolling high",
)
quality_min = st.sidebar.slider(
    "Min quality gates to pass", 1, 5, 3,
    help="Stocks must pass at least this many quality checks",
)

# Gather all unique tickers + SPY for benchmarking
unique_tickers = all_tickers(baskets)
fetch_list = list(dict.fromkeys(unique_tickers + ["SPY"]))

# Fetch data once, shared across tabs
with st.spinner("Loading market data..."):
    prices = fetch_price_history(fetch_list, period="1y")
    fundamentals_df = fetch_all_fundamentals(unique_tickers)

# ---------------------------------------------------------------------------
# TAB: Alerts
# ---------------------------------------------------------------------------
with tab_alerts:
    st.header("Pullback Alerts")
    st.markdown(
        f"Stocks in your baskets that have dropped **{pullback_threshold}%+** "
        f"from their {lookback_days}-day rolling high — filtered by quality."
    )

    if prices.empty:
        st.warning("No price data available.")
    else:
        # Detect pullbacks across all basket tickers
        basket_ticker_cols = [t for t in unique_tickers if t in prices.columns]
        pullbacks = detect_pullbacks_multi_threshold(
            prices[basket_ticker_cols],
            thresholds=[pullback_threshold],
            lookback_days=lookback_days,
        )

        if pullbacks.empty:
            st.success("No pullback alerts right now. Your baskets are holding up.")
        else:
            # Enrich with quality gates and fundamentals
            alert_rows = []
            for _, pb in pullbacks.iterrows():
                tk = pb["ticker"]
                fund_row = fundamentals_df[fundamentals_df["ticker"] == tk]
                if fund_row.empty:
                    continue
                fund = fund_row.iloc[0].to_dict()
                passes = passes_quality(fund, min_pass=quality_min)

                # Find which baskets this ticker belongs to
                membership = [name for name, members in baskets.items() if tk in members]

                alert_rows.append({
                    "Ticker": tk,
                    "Name": fund.get("name", tk),
                    "Baskets": ", ".join(membership),
                    "Price": f"${pb['current_price']:.2f}",
                    "Drop from High": f"-{pb['drop_pct']:.1f}%",
                    "Rolling High": f"${pb['rolling_high']:.2f}",
                    "Quality": quality_summary(fund),
                    "Passes Gate": passes,
                    "Net Margin": f"{fund.get('net_margin', 'N/A')}%"
                        if fund.get('net_margin') is not None else "N/A",
                    "Rev Growth": f"{fund.get('revenue_growth', 'N/A')}%"
                        if fund.get('revenue_growth') is not None else "N/A",
                    "D/E": f"{fund.get('debt_to_equity', 'N/A')}"
                        if fund.get('debt_to_equity') is not None else "N/A",
                })

            if alert_rows:
                alert_df = pd.DataFrame(alert_rows)

                # Show quality-passing alerts first
                passing = alert_df[alert_df["Passes Gate"] == True].drop(columns=["Passes Gate"])
                failing = alert_df[alert_df["Passes Gate"] == False].drop(columns=["Passes Gate"])

                if not passing.empty:
                    st.subheader("Worth investigating")
                    st.markdown(
                        "These stocks pulled back significantly **and** pass your quality checks."
                    )
                    st.dataframe(passing, use_container_width=True, hide_index=True)
                else:
                    st.info("All pullback stocks failed quality checks — nothing actionable right now.")

                if not failing.empty:
                    with st.expander(f"Failed quality gate ({len(failing)} stocks)"):
                        st.markdown("These pulled back but have weak fundamentals. Proceed with caution.")
                        st.dataframe(failing, use_container_width=True, hide_index=True)
            else:
                st.success("No pullback alerts right now.")

# ---------------------------------------------------------------------------
# TAB: Basket Overview
# ---------------------------------------------------------------------------
with tab_baskets:
    st.header("Basket Overview")
    st.markdown("How each basket is performing vs SPY across time horizons.")

    if prices.empty:
        st.warning("No price data available.")
    else:
        # Basket return table
        bret = basket_returns(prices, baskets)
        if not bret.empty:
            st.subheader("Basket Returns")
            display = bret.copy()
            display = display.set_index("basket")
            for col in ["1M", "3M", "6M", "12M"]:
                if col in display.columns:
                    display[col] = display[col].apply(
                        lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"
                    )
            st.dataframe(display, use_container_width=True)

        # Cumulative return chart
        st.subheader("Cumulative Returns (1 Year)")
        cum = basket_cumulative_returns(prices, baskets)
        if not cum.empty:
            chart_data = cum.reset_index().melt(
                id_vars=cum.index.name or "Date",
                var_name="Basket",
                value_name="Return %",
            )
            chart = (
                alt.Chart(chart_data)
                .mark_line()
                .encode(
                    x=alt.X("Date:T"),
                    y=alt.Y("Return %:Q", axis=alt.Axis(format="+.0f")),
                    color=alt.Color("Basket:N"),
                    tooltip=["Date:T", "Basket:N", alt.Tooltip("Return %:Q", format="+.1f")],
                )
                .properties(height=400)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        # Sector rotation heatmap
        st.subheader("Sector Rotation")
        st.markdown(
            "Which baskets are leading or lagging across time windows. "
            "Green = outperforming, Red = underperforming."
        )
        if not bret.empty:
            heat_data = bret.set_index("basket")[["1M", "3M", "6M"]].copy()
            for col in heat_data.columns:
                heat_data[col] = pd.to_numeric(heat_data[col], errors="coerce")
            heat_melt = heat_data.reset_index().melt(
                id_vars="basket", var_name="Period", value_name="Return %"
            )
            heatmap = (
                alt.Chart(heat_melt)
                .mark_rect()
                .encode(
                    x=alt.X("Period:N", sort=["1M", "3M", "6M"]),
                    y=alt.Y("basket:N", title="Basket", sort="-x"),
                    color=alt.Color(
                        "Return %:Q",
                        scale=alt.Scale(scheme="redgreen", domainMid=0),
                    ),
                    tooltip=["basket:N", "Period:N", alt.Tooltip("Return %:Q", format="+.1f")],
                )
                .properties(height=max(len(baskets) * 35, 200))
            )
            text = (
                alt.Chart(heat_melt)
                .mark_text(fontSize=12)
                .encode(
                    x=alt.X("Period:N", sort=["1M", "3M", "6M"]),
                    y=alt.Y("basket:N", sort="-x"),
                    text=alt.Text("Return %:Q", format="+.1f"),
                )
            )
            st.altair_chart(heatmap + text, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB: Stock Detail
# ---------------------------------------------------------------------------
with tab_detail:
    st.header("Stock Detail")
    st.markdown("Drill into a single stock — fundamentals, quality gate, and price chart.")

    all_tk = sorted(unique_tickers)
    selected = st.selectbox("Pick a ticker", options=all_tk, index=0)

    if selected:
        fund_row = fundamentals_df[fundamentals_df["ticker"] == selected]
        if fund_row.empty:
            st.warning(f"No data for {selected}")
        else:
            fund = fund_row.iloc[0].to_dict()

            # Header
            col1, col2, col3 = st.columns(3)
            col1.metric("Price", f"${fund.get('current_price', 0):.2f}" if fund.get("current_price") else "N/A")
            col2.metric("Market Cap", _fmt_cap(fund.get("market_cap")))
            col3.metric("% Below 52W High", f"-{fund.get('pct_below_high', 0):.1f}%" if fund.get("pct_below_high") else "N/A")

            # Quality gate
            st.subheader("Quality Gate")
            gate_results = run_quality_gate(fund)
            gate_labels = {
                "profitable": "Profitable (net margin > 0)",
                "revenue_growing": "Revenue growing (YoY > 0)",
                "not_overleveraged": "Debt manageable (D/E < 250%)",
                "has_liquidity": "Liquidity OK (current ratio > 0.8)",
                "not_penny_stock": "Market cap > $500M",
            }
            gate_cols = st.columns(len(gate_results))
            for i, (name, passed) in enumerate(gate_results.items()):
                icon = "white_check_mark" if passed else "x"
                gate_cols[i].markdown(f":{icon}: **{gate_labels.get(name, name)}**")

            # Key metrics
            st.subheader("Key Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P/E (TTM)", f"{fund['pe_ttm']:.1f}" if fund.get("pe_ttm") else "N/A")
            m2.metric("Net Margin", f"{fund['net_margin']:.1f}%" if fund.get("net_margin") else "N/A")
            m3.metric("ROE", f"{fund['roe']:.1f}%" if fund.get("roe") else "N/A")
            m4.metric("Rev Growth", f"{fund['revenue_growth']:.1f}%" if fund.get("revenue_growth") else "N/A")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Forward P/E", f"{fund['forward_pe']:.1f}" if fund.get("forward_pe") else "N/A")
            m6.metric("D/E", f"{fund['debt_to_equity']:.0f}%" if fund.get("debt_to_equity") else "N/A")
            m7.metric("FCF Margin", f"{fund['fcf_margin']:.1f}%" if fund.get("fcf_margin") else "N/A")
            m8.metric("Analyst Upside", f"{fund['analyst_upside']:+.1f}%" if fund.get("analyst_upside") else "N/A")

            # Basket membership
            membership = [name for name, members in baskets.items() if selected in members]
            if membership:
                st.markdown(f"**Baskets:** {', '.join(membership)}")

            # Price chart
            st.subheader("Price (1 Year)")
            if selected in prices.columns:
                chart_prices = prices[[selected]].dropna().reset_index()
                chart_prices.columns = ["Date", "Price"]
                line = (
                    alt.Chart(chart_prices)
                    .mark_line()
                    .encode(
                        x="Date:T",
                        y=alt.Y("Price:Q", scale=alt.Scale(zero=False)),
                        tooltip=["Date:T", alt.Tooltip("Price:Q", format="$.2f")],
                    )
                    .properties(height=300)
                    .interactive()
                )
                st.altair_chart(line, use_container_width=True)
            else:
                st.info("No price history available for this ticker.")


# ---------------------------------------------------------------------------
# TAB: Baskets
# ---------------------------------------------------------------------------
with tab_settings:
    st.header("Your Baskets")
    st.markdown("Current basket definitions loaded from `baskets.json`.")

    for name, tickers in baskets.items():
        with st.expander(f"{name} ({len(tickers)} stocks)"):
            st.write(", ".join(tickers))
