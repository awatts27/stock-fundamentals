# streamlit run streamlit_app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import math
import json
import os
import altair as alt
from sec_api import QueryApi, ExtractorApi

st.set_page_config(layout="wide")
st.title("📊 Custom Stock Fundamentals Explorer")
# --- Tabs for dashboard sections ---
tab_fundamentals, tab_factors, tab_ownership, tab_earnings = st.tabs([
    "Fundamentals",
    "Factor-Based Smart Beta Portfolio",
    "Ownership",
    "Earnings Highlights",
])
def load_baskets(path="baskets.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

baskets = load_baskets()
basket_names = list(baskets.keys())
basket_names.insert(0, "Custom")

def get_sec_api_key() -> str:
    env_key = os.getenv("SEC_API_KEY")
    if env_key:
        return env_key

    secrets_key = None
    try:
        secrets_key = st.secrets.get("SEC_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        secrets_key = None

    if secrets_key:
        return secrets_key

    st.error("SEC API key not configured. Add it to .streamlit/secrets.toml or set the SEC_API_KEY environment variable.")
    st.stop()


@st.cache_resource
def get_sec_clients():
    key = get_sec_api_key()
    return QueryApi(api_key=key), ExtractorApi(api_key=key)


queryApi, extractorApi = get_sec_clients()

# Define categories
CATEGORY_BY_METRIC = {
    "Current Price": "Price/Volume", "52W High": "Price/Volume", "52W Low": "Price/Volume",
    "% Below 52W High": "Price/Volume", "Avg Vol (10d)": "Price/Volume", "Avg Vol (3m)": "Price/Volume",
    "P/E (TTM)": "Valuation", "P/B": "Valuation", "Dividend Yield %": "Valuation",
    "Net Profit Margin %": "Profitability", "ROE %": "Profitability",
    "Current Ratio": "Liquidity", "Quick Ratio": "Liquidity",
    "Debt/Equity (pct)": "Leverage", "Interest Coverage": "Leverage",
    "Asset Turnover": "Efficiency", "Inventory Turnover": "Efficiency",
    "Revenue YoY %": "Growth", "Earnings YoY %": "Growth",
}

COLUMNS = ["Ticker"] + list(CATEGORY_BY_METRIC.keys())


def safe_div(a, b):
    try:
        if b == 0 or b is None or a is None:
            return None
        return a / b
    except Exception:
        return None


def pct(x):
    return None if x is None else x * 100.0


def condense_text(text: str, max_chars: int = 800) -> str:
    """Squash whitespace and trim to a readable snippet."""
    if not text:
        return ""
    fragments = [part.strip() for part in text.splitlines() if part.strip()]
    combined = " ".join(fragments)
    if len(combined) <= max_chars:
        return combined
    shortened = combined[:max_chars].rsplit(" ", 1)[0]
    return f"{shortened}…"


@st.cache_data(show_spinner="Fetching price history…")
def fetch_price_history(tickers, period="2y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        adj_close = data["Adj Close"]
    else:
        adj_close = data
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(tickers[0])
    adj_close = adj_close.dropna(how="all")
    return adj_close


def sanitize_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


@st.cache_data(show_spinner="Fetching data…")
def fetch_one(tk: str) -> dict:
    t = yf.Ticker(tk)
    info = t.info or {}
    is_a = t.financials
    is_q = t.quarterly_financials
    bs_q = t.quarterly_balance_sheet

    # Price data
    current_price = info.get("currentPrice")
    high52 = info.get("fiftyTwoWeekHigh")
    low52 = info.get("fiftyTwoWeekLow")
    pct_below_high = pct((high52 - current_price) /
                         high52) if current_price and high52 else None
    avg10d = info.get("averageDailyVolume10Day")
    avg3m = info.get("averageVolume")
    pe_ttm = info.get("trailingPE")
    pb = info.get("priceToBook")
    div_y = pct(info.get("dividendYield")) if info.get(
        "dividendYield") else None
    roe = pct(info.get("returnOnEquity")) if info.get(
        "returnOnEquity") else None
    net_margin = pct(info.get("profitMargins")) if info.get(
        "profitMargins") else None

    # Liquidity
    curr_assets = curr_liabs = inventory = None
    if bs_q is not None and not bs_q.empty:
        try:
            col = bs_q.columns[0]
            curr_assets = bs_q.at["Total Current Assets",
                                  col] if "Total Current Assets" in bs_q.index else None
            curr_liabs = bs_q.at["Total Current Liabilities",
                                 col] if "Total Current Liabilities" in bs_q.index else None
            inventory = bs_q.at["Inventory",
                                col] if "Inventory" in bs_q.index else 0
        except Exception as e:
            st.warning(f"Error extracting liquidity metrics: {e}")
            curr_assets = curr_liabs = inventory = None

    current_ratio = safe_div(curr_assets, curr_liabs)
    quick_ratio = safe_div(
        curr_assets - inventory if curr_assets is not None else None, curr_liabs)

    # Leverage
    de = info.get("debtToEquity")
    debt_to_equity_pct = de if de and de > 5 else (de * 100.0 if de else None)

    operating_income = interest_expense = None
    for df in [is_a, is_q]:
        if df is not None and not df.empty:
            col = df.columns[0]
            operating_income = next((df.at[n, col] for n in [
                                    "Operating Income", "Operating Income or Loss"] if n in df.index), None)
            interest_expense = next((df.at[n, col] for n in [
                                    "Interest Expense", "Interest Expense Non-Operating"] if n in df.index), None)
            if operating_income and interest_expense:
                break
    interest_coverage = safe_div(operating_income, abs(
        interest_expense)) if operating_income and interest_expense else None

    # Efficiency
    total_rev = cogs = None
    avg_assets = avg_inventory = None
    if is_a is not None and not is_a.empty:
        cols = list(is_a.columns)
        total_rev = is_a.at["Total Revenue", cols[0]
                            ] if "Total Revenue" in is_a.index else None
        cogs = is_a.at["Cost Of Revenue", cols[0]
                       ] if "Cost Of Revenue" in is_a.index else None

    if t.balance_sheet is not None and not t.balance_sheet.empty:
        bs_cols = list(t.balance_sheet.columns)
        if "Total Assets" in t.balance_sheet.index:
            vals = [t.balance_sheet.at["Total Assets", c] for c in bs_cols[:2]
                    if not math.isnan(t.balance_sheet.at["Total Assets", c])]
            avg_assets = sum(vals) / len(vals) if vals else None
        if "Inventory" in t.balance_sheet.index:
            inv_vals = [t.balance_sheet.at["Inventory", c] for c in bs_cols[:2]
                        if not math.isnan(t.balance_sheet.at["Inventory", c])]
            avg_inventory = sum(inv_vals) / len(inv_vals) if inv_vals else None

    asset_turnover = safe_div(
        total_rev, avg_assets) if total_rev and avg_assets else None
    inv_turn = safe_div(
        cogs, avg_inventory) if cogs and avg_inventory else None

    # Growth
    rev_yoy = earn_yoy = None
    if is_a is not None and not is_a.empty:
        cols = list(is_a.columns)
        if len(cols) >= 2:
            if "Total Revenue" in is_a.index:
                r0, r1 = is_a.at["Total Revenue", cols[0]
                                 ], is_a.at["Total Revenue", cols[1]]
                rev_yoy = pct((r0 - r1) / r1) if r1 else None
            if "Net Income" in is_a.index:
                e0, e1 = is_a.at["Net Income", cols[0]
                                 ], is_a.at["Net Income", cols[1]]
                earn_yoy = pct((e0 - e1) / e1) if e1 else None

    return {
        "Ticker": tk,
        "Current Price": current_price,
        "52W High": high52,
        "52W Low": low52,
        "% Below 52W High": pct_below_high,
        "Avg Vol (10d)": avg10d,
        "Avg Vol (3m)": avg3m,
        "P/E (TTM)": pe_ttm,
        "P/B": pb,
        "Dividend Yield %": div_y,
        "Net Profit Margin %": net_margin,
        "ROE %": roe,
        "Current Ratio": current_ratio,
        "Quick Ratio": quick_ratio,
        "Debt/Equity (pct)": debt_to_equity_pct,
        "Interest Coverage": interest_coverage,
        "Asset Turnover": asset_turnover,
        "Inventory Turnover": inv_turn,
        "Revenue YoY %": rev_yoy,
        "Earnings YoY %": earn_yoy,
    }


# Sidebar: Input tickers
with st.sidebar.expander("ℹ️ What do these metrics mean?", expanded=False):
    st.markdown("""
**Price/Volume**
- **Current Price**: Latest market price of the stock. (No good/bad value; use for comparison)
- **52W High/Low**: Highest/lowest price in the last 52 weeks. (Shows price range)
- **% Below 52W High**: How far the current price is below its 52-week high. Lower values may indicate strength; higher values may signal weakness or opportunity.
- **Avg Vol (10d/3m)**: Average daily trading volume over 10 days/3 months. Higher volume means more liquidity; very low volume can mean riskier trading.

**Valuation**
- **P/E (TTM)**: Price-to-Earnings ratio (Trailing Twelve Months)
    - *Formula:* Price per share / Earnings per share (TTM)
    - *Interpretation:* 10–20 is typical; lower may mean undervalued, higher (>30) may mean overvalued or high growth expectations.
    - *Example:* P/E of 15 means investors pay $15 for every $1 of earnings.
- **P/B**: Price-to-Book ratio
    - *Formula:* Price per share / Book value per share
    - *Interpretation:* <1 may mean undervalued; 1–3 is typical; >3 may mean overvalued or lots of intangible assets.
    - *Example:* P/B of 0.8 means stock trades for less than its net assets.
- **Dividend Yield %**: Annual dividend as a percent of price
    - *Formula:* Annual dividend / Price per share × 100
    - *Interpretation:* 2–6% is typical for dividend stocks; very high yields can signal risk.
    - *Example:* 4% yield means $4 annual dividend for every $100 invested.

**Profitability**
- **Net Profit Margin %**: Net income as a percent of revenue
    - *Formula:* Net income / Revenue × 100
    - *Interpretation:* Higher is better; >10% is good for most industries.
- **ROE %**: Return on equity
    - *Formula:* Net income / Shareholder equity × 100
    - *Interpretation:* >10% is considered good; very high ROE can mean high leverage.

**Liquidity**
- **Current Ratio**: Current assets / Current liabilities
    - *Interpretation:* >1 is healthy; <1 may signal liquidity risk. 1.5–2 is typical.
- **Quick Ratio**: (Current assets - inventory) / Current liabilities
    - *Interpretation:* >1 is good; more conservative than current ratio.

**Leverage**
- **Debt/Equity (pct)**: Debt as a percent of equity
    - *Formula:* Total debt / Total equity × 100
    - *Interpretation:* <100% is conservative; >200% is highly leveraged.
- **Interest Coverage**: Operating income / Interest expense
    - *Interpretation:* >2 is safe; <1 means company may struggle to pay interest.

**Efficiency**
- **Asset Turnover**: Revenue / Average assets
    - *Interpretation:* Higher is better; >1 is typical for retail, lower for capital-intensive industries.
- **Inventory Turnover**: Cost of goods sold / Average inventory
    - *Interpretation:* Higher is better; >5 is good for most industries.

**Growth**
- **Revenue YoY %**: Year-over-year revenue growth
    - *Interpretation:* Positive growth is good; >10% is strong for established companies.
- **Earnings YoY %**: Year-over-year earnings growth
    - *Interpretation:* Positive growth is good; >10% is strong.
    """)
st.sidebar.header("Configuration")
basket_choice = st.sidebar.selectbox("Select basket", options=basket_names, index=0)
if basket_choice != "Custom":
    default_tickers = ",".join(baskets[basket_choice])
else:
    default_tickers = "BMNR,FLNC,OPEN,TEM,CRWV,RKLB,ASTS"
tickers_input = st.sidebar.text_input(
    "Enter comma-separated tickers", value=default_tickers)
if basket_choice != "Custom":
    # Always use basket tickers if basket selected
    tickers = baskets[basket_choice]
else:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Metric/category selection
all_metrics = list(CATEGORY_BY_METRIC.keys())
all_categories = sorted(set(CATEGORY_BY_METRIC.values()))
selected_categories = st.sidebar.multiselect(
    "Select categories to display", options=all_categories, default=all_categories)
filtered_metrics = [m for m in all_metrics if CATEGORY_BY_METRIC[m] in selected_categories]
selected_metrics = st.sidebar.multiselect(
    "Select metrics to display", options=filtered_metrics, default=filtered_metrics)
columns_to_show = ["Ticker"] + selected_metrics

# Fetch & display

with tab_fundamentals:
    st.header("Stock Fundamentals")
    if tickers:
        data = [fetch_one(tk) for tk in tickers]
        df = pd.DataFrame(data, columns=COLUMNS)

        # Filter columns based on user selection
        df = df[columns_to_show]

        # Sorting option (sort before formatting)
        sort_metric = st.sidebar.selectbox("Sort by metric", options=columns_to_show[1:], index=0)
        sort_ascending = st.sidebar.checkbox("Sort ascending", value=True)
        df = df.sort_values(by=sort_metric, ascending=sort_ascending, na_position='last')

        # Format numeric values to 2 decimal places
        def format_value(val, col):
            if col in ["Avg Vol (10d)", "Avg Vol (3m)"]:
                try:
                    return f"{int(float(val)):,}" if val not in [None, "None", "nan"] and str(val).replace(",", "").replace(".", "").isdigit() else val
                except Exception:
                    return val
            if isinstance(val, (int, float)) and not pd.isnull(val):
                return f"{val:.2f}"
            return val
        df = df.copy()
        for col in df.columns:
            # Only format leaf columns (not index levels)
            if isinstance(col, tuple):
                col_name = col[1]
            else:
                col_name = col
            df[col] = df[col].apply(lambda x: format_value(x, col_name))

        # Create MultiIndex columns: first row = categories, second = metric names
        metric_names = df.columns.tolist()
        categories = [""] + [CATEGORY_BY_METRIC.get(col, "") for col in metric_names[1:]]
        multi_index = pd.MultiIndex.from_tuples(
            zip(categories, metric_names), names=["Category", "Metric"])
        df.columns = multi_index

        st.success(f"Fetched data for {len(tickers)} tickers.")
        st.dataframe(df, use_container_width=True)

        # Bar chart visualization
        chart_metric = st.sidebar.selectbox("Bar chart metric", options=columns_to_show[1:], index=0)
        st.subheader(f"Bar Chart: {chart_metric}")
        # Access MultiIndex columns correctly
        ticker_col = ("", "Ticker")
        metric_col = (CATEGORY_BY_METRIC.get(chart_metric, ""), chart_metric)
        tickers_list = df[ticker_col].tolist()
        metric_values = []
        for x in df[metric_col].values:
            try:
                val = float(str(x).replace(",", ""))
            except Exception:
                val = None
            metric_values.append(val)
        # Replace missing or non-numeric values with zero (after building metric_values)
        metric_values = [v if isinstance(v, (int, float)) and not pd.isnull(v) else 0 for v in metric_values]
        chart_data = pd.DataFrame({
            "Ticker": tickers_list,
            chart_metric: metric_values
        })
        chart_data = chart_data.sort_values(by=chart_metric, ascending=False)

        # Altair bar chart for correct sorting
        bar = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Ticker', sort=list(chart_data['Ticker'])),
            y=alt.Y(chart_metric),
            tooltip=['Ticker', chart_metric]
        ).properties(width=700, height=350)
        st.altair_chart(bar, use_container_width=True)

        st.download_button("Download CSV",
                           df.to_csv(index=False),
                           file_name="fundamentals.csv",
                           mime="text/csv")
    else:
        st.info("Please enter one or more tickers in the sidebar.")

with tab_factors:
    st.header("Factor-Based Smart Beta Portfolio")
    st.markdown(
        "Use value, quality, and momentum factors to surface the strongest names in your basket. "
        "Scores are normalized so you can blend factors and highlight the top decile performers."
    )
    st.markdown(
        """
**How to use this tab**

1. Pick a basket or custom ticker list in the sidebar.
2. Adjust the factor weights below (they automatically re-scale to 100%).
3. Review the ranked table, rolling return comparison, and factor correlation heatmap.
"""
    )

    if tickers:
        weight_cols = st.columns(3)
        value_weight = weight_cols[0].slider("Value weight", 0.0, 1.0, 0.33, 0.05)
        quality_weight = weight_cols[1].slider("Quality weight", 0.0, 1.0, 0.33, 0.05)
        momentum_weight = weight_cols[2].slider("Momentum weight", 0.0, 1.0, 0.34, 0.05)
        total_weight = value_weight + quality_weight + momentum_weight
        if total_weight == 0:
            st.error("Set at least one factor weight above zero to build the portfolio.")
        else:
            value_weight /= total_weight
            quality_weight /= total_weight
            momentum_weight /= total_weight
            st.caption(
                f"Weights normalized to 100% → Value: {value_weight:.0%}, "
                f"Quality: {quality_weight:.0%}, Momentum: {momentum_weight:.0%}"
            )

            records = [fetch_one(tk) for tk in tickers]
            fundamentals_df = pd.DataFrame(records, columns=COLUMNS).set_index("Ticker")

            factor_df = pd.DataFrame(index=fundamentals_df.index)
            factor_df["P/E (TTM)"] = sanitize_numeric(fundamentals_df["P/E (TTM)"])
            factor_df["P/B"] = sanitize_numeric(fundamentals_df["P/B"])
            factor_df["ROE %"] = sanitize_numeric(fundamentals_df["ROE %"])
            factor_df["Net Profit Margin %"] = sanitize_numeric(
                fundamentals_df["Net Profit Margin %"]
            )
            factor_df["Debt/Equity (pct)"] = sanitize_numeric(
                fundamentals_df["Debt/Equity (pct)"]
            )

            score_df = pd.DataFrame(index=factor_df.index)

            pe_rank = factor_df["P/E (TTM)"].rank(pct=True, ascending=True)
            pb_rank = factor_df["P/B"].rank(pct=True, ascending=True)
            value_components = []
            if not pe_rank.isna().all():
                value_components.append(1 - pe_rank)
            if not pb_rank.isna().all():
                value_components.append(1 - pb_rank)
            if value_components:
                score_df["Value Score"] = pd.concat(value_components, axis=1).mean(axis=1)
            else:
                score_df["Value Score"] = 0

            roe_rank = factor_df["ROE %"].rank(pct=True, ascending=True)
            margin_rank = factor_df["Net Profit Margin %"].rank(pct=True, ascending=True)
            de_rank = factor_df["Debt/Equity (pct)"].rank(pct=True, ascending=True)
            quality_components = []
            if not roe_rank.isna().all():
                quality_components.append(roe_rank)
            if not margin_rank.isna().all():
                quality_components.append(margin_rank)
            if not de_rank.isna().all():
                quality_components.append(1 - de_rank)
            if quality_components:
                score_df["Quality Score"] = pd.concat(quality_components, axis=1).mean(axis=1)
            else:
                score_df["Quality Score"] = 0

            history_tickers = list(dict.fromkeys(tickers + ["^GSPC"]))
            price_history = fetch_price_history(history_tickers, period="2y")
            twelve_month_returns = {}
            if not price_history.empty:
                for tk in tickers:
                    if tk in price_history.columns:
                        series = price_history[tk].dropna()
                        if len(series) > 250:
                            series = series.iloc[-252:]
                        if len(series) >= 2:
                            twelve_month_returns[tk] = series.iloc[-1] / series.iloc[0] - 1
                if "^GSPC" in price_history.columns:
                    sp_series = price_history["^GSPC"].dropna()
                    if len(sp_series) > 250:
                        sp_series = sp_series.iloc[-252:]
                    if len(sp_series) >= 2:
                        market_return = sp_series.iloc[-1] / sp_series.iloc[0] - 1
                    else:
                        market_return = None
                else:
                    market_return = None
            else:
                market_return = None

            momentum_series = pd.Series(twelve_month_returns)
            momentum_rank = momentum_series.rank(pct=True, ascending=True)
            score_df["Momentum Score"] = momentum_rank.fillna(0)
            score_df["12M Return %"] = momentum_series.mul(100)
            score_df["Beating Market"] = False
            if market_return is not None:
                score_df.loc[momentum_series.index, "Beating Market"] = (
                    momentum_series > market_return
                )
                st.caption(
                    f"S&P 500 12-month return: {market_return * 100:.2f}%"
                    if market_return is not None
                    else "S&P 500 12-month return unavailable."
                )

            for col in ["Value Score", "Quality Score", "Momentum Score"]:
                score_df[col] = score_df[col].fillna(0)

            score_df["Composite Score"] = (
                score_df["Value Score"] * value_weight
                + score_df["Quality Score"] * quality_weight
                + score_df["Momentum Score"] * momentum_weight
            )
            score_df["Factor Rank"] = score_df["Composite Score"].rank(
                method="dense", ascending=False
            )

            score_df = score_df.join(factor_df)
            score_df = score_df.sort_values("Composite Score", ascending=False)

            top_n = max(1, math.ceil(len(score_df) * 0.10))
            top_tickers = score_df.head(top_n).index.tolist()
            score_df["Top Decile"] = score_df.index.isin(top_tickers)

            st.subheader("Factor Rankings")
            st.markdown(
                "Top 10% of tickers by composite score are flagged below. "
                "Higher scores indicate a better balance of value, quality, and momentum characteristics."
            )

            display_cols = [
                "Factor Rank",
                "Top Decile",
                "Composite Score",
                "Value Score",
                "Quality Score",
                "Momentum Score",
                "12M Return %",
                "Beating Market",
                "P/E (TTM)",
                "P/B",
                "ROE %",
                "Net Profit Margin %",
                "Debt/Equity (pct)",
            ]
            styled = score_df[display_cols].style.format(
                {
                    "Composite Score": "{:.2f}",
                    "Value Score": "{:.2f}",
                    "Quality Score": "{:.2f}",
                    "Momentum Score": "{:.2f}",
                    "12M Return %": "{:.2f}",
                    "P/E (TTM)": "{:.2f}",
                    "P/B": "{:.2f}",
                    "ROE %": "{:.2f}",
                    "Net Profit Margin %": "{:.2f}",
                    "Debt/Equity (pct)": "{:.2f}",
                }
            )
            st.dataframe(styled, use_container_width=True)

            if top_tickers and not price_history.empty:
                st.subheader("Rolling 3-Month Returns vs S&P 500")
                st.markdown(
                    "The chart compares an equal-weighted portfolio of the top decile tickers "
                    "against the S&P 500 using 3-month rolling returns. Positive spreads indicate periods "
                    "where the factor blend outperformed the broad market."
                )
                aligned_history = price_history[top_tickers + ["^GSPC"]].dropna(how="all")
                portfolio_returns = aligned_history[top_tickers].pct_change().dropna()
                if not portfolio_returns.empty and "^GSPC" in aligned_history.columns:
                    equal_weight_returns = portfolio_returns.mean(axis=1)
                    market_returns_series = aligned_history["^GSPC"].pct_change().dropna()
                    window = 63

                    def rolling_total_return(series):
                        return (
                            (1 + series).rolling(window).apply(lambda x: (1 + x).prod() - 1)
                        )

                    portfolio_rolling = rolling_total_return(equal_weight_returns).dropna()
                    market_rolling = rolling_total_return(market_returns_series).dropna()
                    returns_df = (
                        pd.concat(
                            [
                                portfolio_rolling.rename("Smart Beta"),
                                market_rolling.rename("S&P 500"),
                            ],
                            axis=1,
                        )
                        .dropna()
                    )
                    if not returns_df.empty:
                        returns_df = returns_df.reset_index().melt(
                            "Date", var_name="Portfolio", value_name="Rolling Return"
                        )
                        chart = (
                            alt.Chart(returns_df)
                            .mark_line()
                            .encode(
                                x="Date:T",
                                y=alt.Y("Rolling Return:Q", axis=alt.Axis(format="%")),
                                color="Portfolio:N",
                                tooltip=["Date:T", "Portfolio:N", alt.Tooltip("Rolling Return:Q", format=".2%")],
                            )
                            .properties(height=320)
                        )
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("Not enough price history to calculate rolling returns.")
                else:
                    st.info("Insufficient price data to build the rolling return comparison.")
            else:
                st.info("Need sufficient price history and at least one top-decile stock to chart rolling returns.")

            st.subheader("Factor Correlation Heatmap")
            st.markdown(
                "This heatmap shows how factor scores move together across your ticker list. "
                "Blue cells above zero mean two factors often rise in tandem; orange cells below zero "
                "indicate diversification benefits when blending them."
            )
            corr = score_df[["Value Score", "Quality Score", "Momentum Score"]].corr().fillna(0)
            corr_reset = corr.reset_index().melt("index", var_name="Factor", value_name="Correlation")
            heatmap = (
                alt.Chart(corr_reset)
                .mark_rect()
                .encode(
                    x="Factor:N",
                    y=alt.Y("index:N", title="Factor"),
                    color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=(-1, 1))),
                    tooltip=["index:N", "Factor:N", alt.Tooltip("Correlation:Q", format=".2f")],
                )
                .properties(height=250)
            )
            text = (
                alt.Chart(corr_reset)
                .mark_text(color="black")
                .encode(
                    x="Factor:N",
                    y=alt.Y("index:N", title="Factor"),
                    text=alt.Text("Correlation:Q", format=".2f"),
                )
            )
            st.altair_chart(heatmap + text, use_container_width=True)

            with st.expander("Factor Glossary"):
                st.markdown(
                    """
**Value** → Rewards lower price-to-earnings and price-to-book ratios, aiming to surface cheaper stocks.

**Quality** → Favors companies with high return on equity and profit margins while penalizing heavy leverage.

**Momentum** → Highlights names whose 12-month total return outpaces peers and the S&P 500 benchmark.

**Composite score** → Weighted blend of the three factor scores; higher values indicate stronger multi-factor characteristics.
                    """
                )
    else:
        st.info("Please enter one or more tickers in the sidebar.")

with tab_ownership:
    st.header("SEC Ownership Data")
    st.info("Displays insider trading activity from recent SEC filings.")
    st.markdown("""
**How to use this tab:**

- Only the first ticker in your selection is shown here. If you select multiple stocks, only the first one will be used for SEC filings.
- To view ownership data for a specific stock, filter to a single ticker using the sidebar basket or by entering just one ticker in the input box.
""")
    from datetime import datetime
    # Date range filter for insider trades
    default_start = datetime.now().replace(month=1, day=1).date()
    default_end = datetime.now().date()
    start_date = st.sidebar.date_input("Insider trades start date", value=default_start)
    end_date = st.sidebar.date_input("Insider trades end date", value=default_end)
    if tickers:
        ticker = tickers[0]
        st.subheader(f"Insider Trades for {ticker}")
        # Query for Form 4 filings (insider trades)
        query = {
            "query": {
                "query_string": {
                    "query": f"ticker:{ticker} AND formType:4 AND filedAt:[{start_date} TO {end_date}]"
                }
            },
            "from": "0",
            "size": "10",
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        try:
            result = queryApi.get_filings(query)
            filings = result.get("filings", [])
            if not filings:
                st.warning("No insider trades found in selected date range.")
            else:
                rows = []
                for f in filings:
                    filed_at = f.get("filedAt", "")
                    insider = f.get("companyNameLong", "")
                    form_type = f.get("formType", "")
                    description = f.get("description", "")
                    link = f.get("linkToFilingDetails", "")
                    rows.append({
                        "Date": filed_at[:10],
                        "Insider": insider,
                        "Form": form_type,
                        "Description": description,
                        "Filing": link
                    })
                df_filings = pd.DataFrame(rows)
                def make_link(url):
                    return f'<a href="{url}" target="_blank">View Filing</a>' if url else ""
                if not df_filings.empty:
                    df_filings["Filing"] = df_filings["Filing"].apply(make_link)
                    st.markdown(df_filings.to_html(escape=False, index=False), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error fetching SEC data: {e}")
    else:
        st.info("Please enter one or more tickers in the sidebar.")

with tab_earnings:
    st.header("Earnings Highlights")
    st.info("Pulls the latest 8-K Item 2.02 (earnings release) to surface a quick snippet and links.")
    if tickers:
        ticker = tickers[0]
        st.subheader(f"Latest Earnings Release for {ticker}")
        earnings_query = {
            "query": {
                "query_string": {
                    "query": f"ticker:{ticker} AND formType:8-K AND items:\"Item 2.02\""
                }
            },
            "from": "0",
            "size": "5",
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        filings = []
        try:
            result = queryApi.get_filings(earnings_query)
            filings = result.get("filings", []) if result else []
        except Exception as e:
            st.error(f"Error fetching earnings filings: {e}")
        if not filings:
            st.warning("No recent earnings-related 8-K filings found.")
        else:
            filing_labels = [
                f"{f.get('filedAt', '')[:10]} — {f.get('description', '8-K Filing')}" for f in filings
            ]
            selected_index = st.selectbox(
                "Select filing",
                options=list(range(len(filings))),
                format_func=lambda idx: filing_labels[idx]
            )
            filing = filings[selected_index]
            filed_at = filing.get("filedAt", "")
            st.caption(f"Filed: {filed_at[:10]} | Accession: {filing.get('accessionNo', 'N/A')}")

            section_text = ""
            link_to_html = filing.get("linkToHtml") or filing.get("linkToFilingDetails")
            if link_to_html:
                try:
                    section_text = extractorApi.get_section(link_to_html, "item 2.02", "text")
                except Exception:
                    st.info("Unable to extract Item 2.02 text automatically; showing filing links instead.")
            if not section_text:
                section_text = filing.get("description") or ""

            snippet = condense_text(section_text)
            if snippet:
                st.write(snippet)
            else:
                st.warning("No snippet available for this filing.")

            link_primary = filing.get("linkToFilingDetails")
            link_press = None
            for doc in filing.get("documentFormatFiles", []) or []:
                if doc.get("documentType", "").upper().startswith("EX-99"):
                    link_press = doc.get("documentUrl")
                    break
            links = []
            if link_primary:
                links.append(f"[Filing Detail]({link_primary})")
            if link_press:
                links.append(f"[Press Release Exhibit]({link_press})")
            if links:
                st.markdown(" | ".join(links))
    else:
        st.info("Please enter one or more tickers in the sidebar.")
