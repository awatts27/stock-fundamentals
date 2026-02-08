"""Stock Watchlist v2 — Buy-the-dip alert system.

Philosophy: We can't beat Wall Street. We just want to find good businesses
and buy them when they're cheap. The app does two things:
  1. Filter out bad businesses (quality gate — all-or-nothing).
  2. Tell us when the good ones pull back (price alert).
"""

import streamlit as st
import pandas as pd
import altair as alt

from data import (
    load_baskets,
    all_tickers,
    fetch_price_history,
    fetch_fundamentals,
    fetch_all_fundamentals,
    classify_dip,
    DIP_LABELS,
)
from quality import passes_quality, quality_summary, run_quality_gate, quality_failures
from pullbacks import detect_pullbacks
from relative_strength import basket_returns, basket_cumulative_returns


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


def _fmt_pct(val, suffix="%") -> str:
    if val is None:
        return "—"
    return f"{val:+.1f}{suffix}" if val >= 0 else f"{val:.1f}{suffix}"


def _fmt_cash(val) -> str:
    if val is None:
        return "—"
    if val >= 1_000_000_000:
        return f"${val / 1_000_000_000:.1f}B"
    if val >= 1_000_000:
        return f"${val / 1_000_000:.0f}M"
    if val <= -1_000_000_000:
        return f"-${abs(val) / 1_000_000_000:.1f}B"
    if val <= -1_000_000:
        return f"-${abs(val) / 1_000_000:.0f}M"
    return f"${val:,.0f}"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Stock Watchlist", layout="wide")
st.title("Stock Watchlist")
st.caption("Good businesses on sale — nothing more, nothing less")

# ---------------------------------------------------------------------------
# Load baskets
# ---------------------------------------------------------------------------
baskets = load_baskets()
if not baskets:
    st.error("No baskets.json found.")
    st.stop()

tab_alerts, tab_detail, tab_baskets_overview, tab_baskets = st.tabs([
    "On Sale",
    "Stock Detail",
    "Basket Overview",
    "Baskets",
])

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
pullback_threshold = st.sidebar.slider(
    "Pullback threshold %", 10, 50, 20, 5,
    help="Show stocks that dropped at least this much from their rolling high",
)
lookback_days = st.sidebar.slider(
    "Lookback window (trading days)", 21, 126, 63, 21,
    help="How far back to look for the rolling high",
)

# Gather all unique tickers + SPY for benchmarking
unique_tickers = all_tickers(baskets)
fetch_list = list(dict.fromkeys(unique_tickers + ["SPY"]))

# Fetch data once, shared across tabs
with st.spinner("Loading market data..."):
    prices = fetch_price_history(fetch_list, period="1y")
    fundamentals_df = fetch_all_fundamentals(unique_tickers)


# ---------------------------------------------------------------------------
# Helper: find which baskets a ticker belongs to
# ---------------------------------------------------------------------------
def _baskets_for(ticker: str) -> list[str]:
    return [name for name, members in baskets.items() if ticker in members]


# ---------------------------------------------------------------------------
# TAB: On Sale
# ---------------------------------------------------------------------------
with tab_alerts:
    st.header("On Sale")
    st.markdown(
        "Good businesses that have pulled back. Every stock here passes **all** "
        "quality gates (profitable, cash-flow positive, growing revenue, "
        "manageable debt, liquid, not a penny stock)."
    )

    if prices.empty:
        st.warning("No price data available.")
    else:
        # Step 1: quality filter — only stocks that pass ALL gates
        quality_tickers = []
        for _, row in fundamentals_df.iterrows():
            if row.get("is_etf"):
                continue
            if passes_quality(row.to_dict()):
                quality_tickers.append(row["ticker"])

        if not quality_tickers:
            st.info("No stocks in your baskets pass all quality gates right now.")
        else:
            # Step 2: detect pullbacks among quality stocks only
            quality_price_cols = [t for t in quality_tickers if t in prices.columns]
            if not quality_price_cols:
                st.info("No price data for quality stocks.")
            else:
                pullbacks = detect_pullbacks(
                    prices[quality_price_cols],
                    threshold_pct=pullback_threshold,
                    lookback_days=lookback_days,
                )

                if pullbacks.empty:
                    st.success(
                        f"No quality stocks have pulled back {pullback_threshold}%+ right now. "
                        "Your baskets are holding up."
                    )
                else:
                    # Enrich with fundamentals and dip context
                    alert_data = []
                    for _, pb in pullbacks.iterrows():
                        tk = pb["ticker"]
                        fund_row = fundamentals_df[fundamentals_df["ticker"] == tk]
                        if fund_row.empty:
                            continue
                        fund = fund_row.iloc[0].to_dict()

                        membership = _baskets_for(tk)
                        first_basket = membership[0] if membership else None
                        basket_members = baskets.get(first_basket, []) if first_basket else []
                        dip_type = classify_dip(prices, tk, basket_members, lookback=lookback_days)

                        alert_data.append({
                            "fund": fund,
                            "pb": pb,
                            "membership": membership,
                            "dip_type": dip_type,
                        })

                    if alert_data:
                        for item in alert_data:
                            fund = item["fund"]
                            pb = item["pb"]
                            dip_type = item["dip_type"]
                            tk = fund.get("ticker", "?")
                            name = fund.get("name", tk)
                            summary = fund.get("summary", "")
                            dip_label = DIP_LABELS.get(dip_type, dip_type)

                            with st.expander(
                                f"**{tk}** — {name}  ·  ${pb['current_price']:.2f}  ·  "
                                f"**-{pb['drop_pct']:.0f}%**  ·  {dip_label}"
                            ):
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Price", f"${pb['current_price']:.2f}")
                                c2.metric("Drop", f"-{pb['drop_pct']:.0f}%")
                                c3.metric("Net Margin", _fmt_pct(fund.get("net_margin")))
                                c4.metric("Rev Growth", _fmt_pct(fund.get("revenue_growth")))

                                c5, c6, c7, c8 = st.columns(4)
                                c5.metric("FCF", _fmt_cash(fund.get("fcf")))
                                c6.metric("FCF Margin", _fmt_pct(fund.get("fcf_margin")))
                                c7.metric("P/E", f"{fund['pe_ttm']:.1f}" if fund.get("pe_ttm") else "—")
                                c8.metric("D/E", f"{fund['debt_to_equity']:.0f}%" if fund.get("debt_to_equity") else "—")

                                st.markdown(f"**Dip Type:** {dip_label}")
                                if item["membership"]:
                                    st.markdown(f"**Baskets:** {', '.join(item['membership'])}")

                                st.markdown(f"[Latest news on Yahoo Finance](https://finance.yahoo.com/quote/{tk}/news/)")

                                if summary:
                                    st.caption(summary[:300] + ("..." if len(summary) > 300 else ""))

                        st.markdown("---")
                        st.markdown(
                            "**Dip Type guide:** "
                            "*Market-wide* = SPY is also down (safest to buy). "
                            "*Sector* = the whole basket is down. "
                            "*Stock-specific* = only this stock is falling (most dangerous — dig deeper)."
                        )
                    else:
                        st.success("No actionable alerts right now.")

            # Show how many stocks were filtered out
            total = len(fundamentals_df[~fundamentals_df.get("is_etf", False)])
            filtered_out = total - len(quality_tickers)
            if filtered_out > 0:
                with st.expander(f"{filtered_out} stocks hidden (failed quality gate)"):
                    st.markdown(
                        "These stocks are in your baskets but don't pass all quality gates. "
                        "They'll never show up as buy-the-dip candidates."
                    )
                    failed_rows = []
                    for _, row in fundamentals_df.iterrows():
                        rd = row.to_dict()
                        if rd.get("is_etf") or passes_quality(rd):
                            continue
                        failures = quality_failures(rd)
                        failed_rows.append({
                            "Ticker": rd.get("ticker", "?"),
                            "Name": rd.get("name", "?"),
                            "Why": "; ".join(failures),
                        })
                    if failed_rows:
                        st.dataframe(
                            pd.DataFrame(failed_rows),
                            use_container_width=True,
                            hide_index=True,
                        )


# ---------------------------------------------------------------------------
# TAB: Stock Detail
# ---------------------------------------------------------------------------
with tab_detail:
    st.header("Stock Detail")
    st.markdown("Deep dive into any stock — quality gates, fundamentals, price chart, and dip context.")

    all_tk = sorted(unique_tickers)
    col_search, col_or, col_select = st.columns([2, 0.5, 2])
    with col_search:
        custom_ticker = st.text_input(
            "Search any ticker",
            placeholder="e.g. TSLA, DIS, NFLX...",
        ).strip().upper()
    with col_or:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**or**")
    with col_select:
        basket_pick = st.selectbox("Pick from your baskets", options=[""] + all_tk, index=0)

    selected = custom_ticker if custom_ticker else basket_pick

    if selected:
        # Check if it's already in our fetched data
        fund_row = fundamentals_df[fundamentals_df["ticker"] == selected]
        if not fund_row.empty:
            fund = fund_row.iloc[0].to_dict()
        else:
            # Fetch on the fly for any ticker
            with st.spinner(f"Looking up {selected}..."):
                try:
                    fund = fetch_fundamentals(selected)
                except Exception:
                    fund = None

            if not fund or not fund.get("current_price"):
                st.error(f"Could not find data for '{selected}'. Check the ticker symbol.")
                st.stop()

        passes = passes_quality(fund)

        # Company description
        summary = fund.get("summary", "")
        if summary:
            with st.expander(f"About {fund.get('name', selected)}"):
                if fund.get("industry"):
                    st.caption(f"{fund.get('sector', '')} / {fund['industry']}")
                st.markdown(summary)

        # Header row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"${fund.get('current_price', 0):.2f}" if fund.get("current_price") else "N/A")
        col2.metric("Market Cap", _fmt_cap(fund.get("market_cap")))
        col3.metric("% Below 52W High", f"-{fund.get('pct_below_high', 0):.1f}%" if fund.get("pct_below_high") else "N/A")
        col4.metric("Quality", quality_summary(fund))

        # Pass/fail banner
        if passes:
            st.success("This stock passes all quality gates — eligible for buy-the-dip alerts.")
        else:
            st.error("This stock FAILS quality checks — it will never appear in alerts.")

        # Quality gate breakdown
        st.subheader("Quality Gates")
        gate_results = run_quality_gate(fund)
        gate_cols = st.columns(len(gate_results))
        for i, gate in enumerate(gate_results):
            icon = ":white_check_mark:" if gate["passed"] else ":x:"
            gate_cols[i].markdown(f"{icon} **{gate['name']}**")
            if not gate["passed"]:
                gate_cols[i].caption(gate["reason"])

        # Key metrics
        st.subheader("Key Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net Margin", _fmt_pct(fund.get("net_margin")))
        m2.metric("Free Cash Flow", _fmt_cash(fund.get("fcf")))
        m3.metric("FCF Margin", _fmt_pct(fund.get("fcf_margin")))
        m4.metric("Revenue Growth", _fmt_pct(fund.get("revenue_growth")))

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("P/E (TTM)", f"{fund['pe_ttm']:.1f}" if fund.get("pe_ttm") else "—")
        m6.metric("Forward P/E", f"{fund['forward_pe']:.1f}" if fund.get("forward_pe") else "—")
        m7.metric("ROE", _fmt_pct(fund.get("roe")))
        m8.metric("D/E", f"{fund['debt_to_equity']:.0f}%" if fund.get("debt_to_equity") else "—")

        m9, m10, m11, m12 = st.columns(4)
        m9.metric("Current Ratio", f"{fund['current_ratio']:.2f}" if fund.get("current_ratio") else "—")
        m10.metric("Analyst Upside", _fmt_pct(fund.get("analyst_upside")))
        m11.metric("Short % Float", f"{fund['short_pct_float']:.1f}%" if fund.get("short_pct_float") else "—")
        m12.metric("Beta", f"{fund['beta']:.2f}" if fund.get("beta") else "—")

        # External links
        st.markdown(f"[View {selected} on Yahoo Finance](https://finance.yahoo.com/quote/{selected}/) · "
                    f"[Latest news](https://finance.yahoo.com/quote/{selected}/news/)")

        # Conviction check
        st.subheader("Conviction Check")
        st.caption("Quick signals to help you decide if this dip is worth buying. Not a recommendation — just data.")

        checks = []

        # 1. Analysts still bullish?
        upside = fund.get("analyst_upside")
        if upside is not None:
            if upside > 10:
                checks.append(("Analysts still bullish", f"Target implies {upside:+.0f}% upside", True))
            elif upside > 0:
                checks.append(("Analysts lukewarm", f"Target implies only {upside:+.0f}% upside", None))
            else:
                checks.append(("Analysts bearish", f"Target implies {upside:+.0f}% (downside)", False))
        else:
            checks.append(("No analyst data", "No price target available", None))

        # 2. Earnings direction — forward P/E vs trailing P/E
        fwd_pe = fund.get("forward_pe")
        trail_pe = fund.get("pe_ttm")
        if fwd_pe and trail_pe and trail_pe > 0:
            if fwd_pe < trail_pe * 0.9:
                checks.append(("Earnings expected to grow", f"Forward P/E ({fwd_pe:.0f}) < Trailing P/E ({trail_pe:.0f})", True))
            elif fwd_pe > trail_pe * 1.1:
                checks.append(("Earnings expected to shrink", f"Forward P/E ({fwd_pe:.0f}) > Trailing P/E ({trail_pe:.0f})", False))
            else:
                checks.append(("Earnings roughly flat", f"Forward P/E ({fwd_pe:.0f}) ≈ Trailing P/E ({trail_pe:.0f})", None))
        else:
            checks.append(("Can't assess earnings direction", "Missing P/E data", None))

        # 3. Short sellers piling in?
        short_pct = fund.get("short_pct_float")
        if short_pct is not None:
            if short_pct > 10:
                checks.append(("Heavy short interest", f"{short_pct:.1f}% of float shorted — bears are aggressive", False))
            elif short_pct > 5:
                checks.append(("Elevated short interest", f"{short_pct:.1f}% of float shorted — some bearish bets", None))
            else:
                checks.append(("Low short interest", f"{short_pct:.1f}% of float shorted — bears aren't interested", True))
        else:
            checks.append(("No short interest data", "Short % of float unavailable", None))

        # 4. Revenue AND earnings both declining?
        rev_g = fund.get("revenue_growth")
        earn_g = fund.get("earnings_growth")
        if rev_g is not None and earn_g is not None:
            if rev_g < 0 and earn_g < 0:
                checks.append(("Double decline", f"Revenue ({rev_g:+.1f}%) and earnings ({earn_g:+.1f}%) both shrinking", False))
            elif rev_g > 0 and earn_g > 0:
                checks.append(("Both growing", f"Revenue ({rev_g:+.1f}%) and earnings ({earn_g:+.1f}%) both up", True))
            elif rev_g > 0 and earn_g < 0:
                checks.append(("Revenue up, earnings down", f"Revenue ({rev_g:+.1f}%) growing but earnings ({earn_g:+.1f}%) falling — spending to grow?", None))
            else:
                checks.append(("Mixed signals", f"Revenue ({rev_g:+.1f}%) and earnings ({earn_g:+.1f}%)", None))
        elif rev_g is not None:
            if rev_g > 0:
                checks.append(("Revenue growing", f"{rev_g:+.1f}% (no earnings growth data)", True))
            else:
                checks.append(("Revenue declining", f"{rev_g:+.1f}% (no earnings growth data)", False))
        else:
            checks.append(("Growth data unavailable", "Can't assess revenue/earnings trend", None))

        # Display checks
        for label, detail, signal in checks:
            if signal is True:
                st.markdown(f":white_check_mark: **{label}** — {detail}")
            elif signal is False:
                st.markdown(f":x: **{label}** — {detail}")
            else:
                st.markdown(f":large_orange_diamond: **{label}** — {detail}")

        # Summary
        green = sum(1 for _, _, s in checks if s is True)
        red = sum(1 for _, _, s in checks if s is False)
        if green >= 3 and red == 0:
            st.success("Signals are mostly positive. The dip may be a buying opportunity — but always check the news.")
        elif red >= 3:
            st.error("Multiple warning signs. This dip might be justified. Proceed with caution.")
        elif red >= 2:
            st.warning("Mixed signals. Do your research before buying.")
        else:
            st.info("Signals are mixed. Check the Yahoo Finance link above for recent news.")

        # Fetch price data (needed for dip context and chart)
        detail_prices = prices
        if selected not in prices.columns:
            custom_prices = fetch_price_history([selected, "SPY"], period="1y")
            if not custom_prices.empty:
                detail_prices = custom_prices

        # Dip context
        membership = _baskets_for(selected)
        st.subheader("Dip Context")
        if membership:
            first_basket = membership[0]
            basket_members = baskets.get(first_basket, [])
            dip_type = classify_dip(detail_prices, selected, basket_members, lookback=lookback_days)
        else:
            # Custom ticker — can still check vs SPY
            dip_type = classify_dip(detail_prices, selected, [], lookback=lookback_days)

        dip_label = DIP_LABELS.get(dip_type, dip_type)

        if dip_type == "market":
            st.info(f"**{dip_label}** — SPY is also down. This is a broad market pullback, "
                    "not specific to this stock. Historically the safest time to buy quality.")
        elif dip_type == "sector":
            first_basket = membership[0] if membership else "sector"
            st.warning(f"**{dip_label}** — The {first_basket} basket is broadly down. "
                       "Could be sector rotation. Worth watching but not as safe as a market dip.")
        elif dip_type == "stock":
            st.error(f"**{dip_label}** — Only this stock is falling while its peers are fine. "
                     "This is the most dangerous type of dip. Dig into the news before buying.")
        else:
            st.info(f"**{dip_label}**")

        if membership:
            st.markdown(f"**Baskets:** {', '.join(membership)}")
        else:
            st.caption("This stock is not in any of your baskets.")

        # Price chart — stock vs S&P 500 (normalized to % change)
        st.subheader("Price vs S&P 500 (1 Year)")

        if selected in detail_prices.columns:
            # Build normalized % change for the stock
            stock_prices = detail_prices[[selected]].dropna()
            stock_norm = ((stock_prices[selected] / stock_prices[selected].iloc[0]) - 1) * 100

            chart_data = pd.DataFrame({"Date": stock_norm.index, selected: stock_norm.values})

            # Add SPY if available
            if "SPY" in detail_prices.columns:
                spy_prices = detail_prices[["SPY"]].dropna()
                # Align to same start date
                common_start = max(stock_prices.index[0], spy_prices.index[0])
                spy_aligned = spy_prices[spy_prices.index >= common_start]
                stock_aligned = stock_norm[stock_norm.index >= common_start]
                spy_norm = ((spy_aligned["SPY"] / spy_aligned["SPY"].iloc[0]) - 1) * 100

                chart_data = pd.DataFrame({
                    "Date": stock_aligned.index,
                    selected: (detail_prices[selected][detail_prices.index >= common_start] / detail_prices[selected][detail_prices.index >= common_start].iloc[0] - 1) * 100,
                })
                chart_data["S&P 500"] = spy_norm.values[:len(chart_data)]

            chart_melted = chart_data.melt(
                id_vars="Date", var_name="Symbol", value_name="Change %"
            )

            line = (
                alt.Chart(chart_melted)
                .mark_line()
                .encode(
                    x="Date:T",
                    y=alt.Y("Change %:Q", axis=alt.Axis(format="+.0f")),
                    color=alt.Color("Symbol:N"),
                    strokeDash=alt.condition(
                        alt.datum.Symbol == "S&P 500",
                        alt.value([5, 5]),
                        alt.value([0]),
                    ),
                    tooltip=["Date:T", "Symbol:N", alt.Tooltip("Change %:Q", format="+.1f")],
                )
                .properties(height=350)
                .interactive()
            )
            st.altair_chart(line, use_container_width=True)
            st.caption("Both lines normalized to % change from 1 year ago. S&P 500 shown as dashed line.")
        else:
            st.info("No price history available for this ticker.")


# ---------------------------------------------------------------------------
# TAB: Basket Overview
# ---------------------------------------------------------------------------
with tab_baskets_overview:
    st.header("Basket Overview")
    st.markdown("How each basket is performing vs SPY.")

    if prices.empty:
        st.warning("No price data available.")
    else:
        bret = basket_returns(prices, baskets)
        if not bret.empty:
            st.subheader("Returns by Period")
            display = bret.copy().set_index("basket")
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

        # Heatmap
        if not bret.empty:
            st.subheader("Sector Rotation")
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
                        scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
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
# TAB: Baskets
# ---------------------------------------------------------------------------
with tab_baskets:
    st.header("Your Baskets")
    st.markdown("Current basket definitions loaded from `baskets.json`.")

    for name, tickers in baskets.items():
        with st.expander(f"{name} ({len(tickers)} stocks)"):
            st.write(", ".join(tickers))
