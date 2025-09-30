# streamlit run streamlit_app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import math
import json
import os
from datetime import datetime
import altair as alt
import requests
from sec_api import QueryApi, ExtractorApi

st.set_page_config(layout="wide")
st.title("📊 Custom Stock Fundamentals Explorer")

if "_welcome_shown" not in st.session_state:
    st.toast("Welcome! Pick a basket in the sidebar to explore fundamentals and factors.", icon="👋")
    st.session_state["_welcome_shown"] = True
# --- Tabs for dashboard sections ---
tab_fundamentals, tab_factors, tab_regime, tab_ownership, tab_earnings, tab_glossary = st.tabs([
    "Fundamentals",
    "Factor-Based Smart Beta Portfolio",
    "Macro Regime Insights",
    "Ownership",
    "Earnings Highlights",
    "Guide & Glossary",
])
def load_baskets(path="baskets.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

baskets = load_baskets()
basket_names = list(baskets.keys())
basket_names.insert(0, "Custom")
basket_names.insert(1, "All Baskets")

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


def get_fred_api_key() -> str:
    env_key = os.getenv("FRED_API_KEY")
    if env_key:
        return env_key

    secrets_key = None
    try:
        secrets_key = st.secrets.get("FRED_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        secrets_key = None

    if secrets_key:
        return secrets_key

    st.error("FRED API key not configured. Add it to .streamlit/secrets.toml or set the FRED_API_KEY environment variable.")
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
    "Market Cap": "Size",
    "Company Name": "Profile",
    "P/E (TTM)": "Valuation", "P/B": "Valuation", "Dividend Yield %": "Valuation",
    "Net Profit Margin %": "Profitability", "ROE %": "Profitability",
    "Current Ratio": "Liquidity", "Quick Ratio": "Liquidity",
    "Debt/Equity (pct)": "Leverage", "Interest Coverage": "Leverage",
    "Asset Turnover": "Efficiency", "Inventory Turnover": "Efficiency",
    "Revenue YoY %": "Growth", "Earnings YoY %": "Growth",
}

COLUMNS = ["Ticker"] + list(CATEGORY_BY_METRIC.keys())

METRIC_GUIDE = {
    "P/E (TTM)": {
        "summary": "Price paid for each dollar of trailing earnings.",
        "good": "10-20",
        "caution": ">30 may imply rich valuation unless growth justifies it.",
    },
    "P/B": {
        "summary": "Price relative to accounting book value.",
        "good": "1-3",
        "caution": "<1 can signal value or distress; >5 often means high optimism.",
    },
    "Dividend Yield %": {
        "summary": "Cash income as a percent of price.",
        "good": "2-5% for stable payers.",
        "caution": ">8% could be unsustainable.",
    },
    "Market Cap": {
        "summary": "Company size measured by market capitalization (shares outstanding × price).",
        "good": "Match to your risk tolerance: mega caps provide stability; smaller caps can be volatile.",
        "caution": "Thinly traded micro caps (<$500M) can be illiquid and risky.",
    },
    "ROE %": {
        "summary": "Profitability on shareholder equity.",
        "good": ">15% shows strong efficiency.",
        "caution": "<5% may trail peers.",
    },
    "Net Profit Margin %": {
        "summary": "Percent of revenue kept as profit.",
        "good": ">10% is solid for most industries.",
        "caution": "Negative margins highlight losses.",
    },
    "Debt/Equity (pct)": {
        "summary": "Financial leverage vs equity base.",
        "good": "<100% = conservative leverage.",
        "caution": ">200% means heavy debt reliance.",
    },
    "Current Ratio": {
        "summary": "Short-term assets divided by liabilities.",
        "good": "1.5-2.0 is healthy.",
        "caution": "<1 raises liquidity questions.",
    },
    "Quick Ratio": {
        "summary": "Liquidity without inventory.",
        "good": ">1.0", "caution": "<0.8 may signal tight liquidity.",
    },
    "% Below 52W High": {
        "summary": "Distance from recent high.",
        "good": "0-15% shows momentum.",
        "caution": ">30% may signal drawdown or opportunity.",
    },
    "Revenue YoY %": {
        "summary": "Annual revenue growth rate.",
        "good": ">10% = strong growth.",
        "caution": "Negative growth warrants investigation.",
    },
    "Earnings YoY %": {
        "summary": "Annual net income growth.",
        "good": ">10% suggests expanding profitability.",
        "caution": "Declines flag earnings pressure.",
    },
}


def _extract_numeric_token(text: str | None) -> str | None:
    if not text:
        return None
    token = text.strip().split()[0].rstrip('.')
    return token if token else None


def _parse_threshold(token: str, value: float) -> bool:
    token_clean = token.replace('%', '')
    if '-' in token_clean:
        low, high = token_clean.split('-', 1)
        try:
            low_val = float(low)
            high_val = float(high)
        except ValueError:
            return False
        return low_val <= value <= high_val
    if token.startswith('>='):
        try:
            return value >= float(token_clean[2:])
        except ValueError:
            return False
    if token.startswith('>'):
        try:
            return value > float(token_clean[1:])
        except ValueError:
            return False
    if token.startswith('<='):
        try:
            return value <= float(token_clean[2:])
        except ValueError:
            return False
    if token.startswith('<'):
        try:
            return value < float(token_clean[1:])
        except ValueError:
            return False
    try:
        threshold = float(token_clean)
    except ValueError:
        return False
    return value >= threshold


def metric_is_strong(metric: str, raw_value) -> bool:
    info = METRIC_GUIDE.get(metric)
    if not info:
        return False
    if raw_value is None:
        return False
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return False
    if pd.isna(value):
        return False
    token = _extract_numeric_token(info.get('good'))
    if not token:
        return False
    return _parse_threshold(token, value)


def safe_div(a, b):
    try:
        if b == 0 or b is None or a is None:
            return None
        return a / b
    except Exception:
        return None


def pct(x):
    return None if x is None else x * 100.0


def _normalize_label(label: str) -> str:
    return label.replace(" ", "").lower()


def _lookup_balance_value(frame: pd.DataFrame | None, aliases: list[str]) -> float | None:
    if frame is None or frame.empty:
        return None
    try:
        column = frame.columns[0]
    except (IndexError, KeyError):
        return None
    normalized_index = {_normalize_label(str(idx)): idx for idx in frame.index}
    for alias in aliases:
        key = _normalize_label(alias)
        if key in normalized_index:
            try:
                value = frame.at[normalized_index[key], column]
                return value
            except Exception:
                continue
    return None


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


def format_market_cap(value) -> str | None:
    """Represent market cap in friendly units (BN or M when large)."""
    if value in [None, "", "None"]:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if pd.isna(numeric):
        return None
    prefix = "-" if numeric < 0 else ""
    magnitude = abs(numeric)
    if magnitude >= 1_000_000_000:
        trimmed = f"{magnitude / 1_000_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{prefix}{trimmed}BN"
    if magnitude >= 1_000_000:
        trimmed = f"{magnitude / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{prefix}{trimmed}M"
    return f"{prefix}{magnitude:,.0f}"


def parse_display_number(value) -> float | None:
    """Convert formatted display values back to numeric for charts/calculations."""
    if value in [None, "", "None"]:
        return None
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(",", "").upper()
    multiplier = 1.0
    if cleaned.endswith("BN"):
        multiplier = 1_000_000_000.0
        cleaned = cleaned[:-2]
    elif cleaned.endswith("M"):
        multiplier = 1_000_000.0
        cleaned = cleaned[:-1]
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


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
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        primary_levels = list(data.columns.levels[0])
        target_level = None
        for candidate in ["Adj Close", "Close", "Close*"]:
            if candidate in primary_levels:
                target_level = candidate
                break
        if target_level is None and primary_levels:
            target_level = primary_levels[0]
        adj_close = data.xs(target_level, axis=1, level=0) if target_level else data
    else:
        if "Adj Close" in data.columns:
            adj_close = data["Adj Close"]
        elif "Close" in data.columns:
            adj_close = data["Close"]
        else:
            adj_close = data

    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(tickers[0])
    adj_close = adj_close.dropna(how="all")
    adj_close.columns = [str(col) for col in adj_close.columns]
    return adj_close


def sanitize_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


FACTOR_DESCRIPTIONS = {
    "Value": "Rewards lower price ratios such as P/E and P/B, seeking bargains.",
    "Quality": "Focuses on high ROE, robust margins, and manageable debt.",
    "Momentum": "Highlights 12-month winners that continue to outperform.",
}


MACRO_GUIDE = {
    "10Y-2Y Treasury Spread": {
        "summary": "Positive spread = normal growth expectations; negative (inverted) often precedes recessions.",
        "good": ">0%",
        "caution": "<0% warns of economic slowdown.",
    },
    "CPI YoY": {
        "summary": "Measures consumer inflation; rising prices can pressure profits but aid real assets.",
        "good": "2-3% aligned with Fed target.",
        "caution": ">3% suggests persistent inflation.",
    },
    "ISM Manufacturing PMI": {
        "summary": "Survey of factory activity; 50 is the expansion/contraction line.",
        "good": ">50 = expansion.",
        "caution": "<50 indicates contraction.",
    },
}


@st.cache_data(show_spinner="Fetching macro data…")
def fetch_fred_series(series_id: str, observation_start: str = "2000-01-01") -> tuple[pd.DataFrame, str | None]:
    api_key = get_fred_api_key()
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
    }
    url = "https://api.stlouisfed.org/fred/series/observations"
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("observations", [])
    except Exception as exc:
        return pd.DataFrame(), f"{series_id}: {exc}"
    df = pd.DataFrame(data)
    if df.empty:
        return df, f"{series_id}: no data returned"
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date").sort_index(), None


def latest_value(df: pd.DataFrame) -> tuple[float | None, datetime | None]:
    if df is None or df.empty:
        return None, None
    series = df["value"].dropna()
    if series.empty:
        return None, None
    return series.iloc[-1], series.index[-1]


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
    market_cap = info.get("marketCap")
    company_name = info.get("longName") or info.get("shortName") or info.get("name")
    pe_ttm = info.get("trailingPE")
    pb = info.get("priceToBook")
    div_y = pct(info.get("dividendYield")) if info.get(
        "dividendYield") else None
    roe = pct(info.get("returnOnEquity")) if info.get(
        "returnOnEquity") else None
    net_margin = pct(info.get("profitMargins")) if info.get(
        "profitMargins") else None

    # Liquidity
    curr_assets = curr_liabs = None
    inventory = 0.0
    inventory_found = False
    for sheet in [bs_q, t.balance_sheet]:
        if sheet is None or sheet.empty:
            continue
        try:
            curr_assets = curr_assets if curr_assets is not None else _lookup_balance_value(
                sheet, ["Total Current Assets", "TotalCurrentAssets", "CurrentAssets"]
            )
            curr_liabs = curr_liabs if curr_liabs is not None else _lookup_balance_value(
                sheet, ["Total Current Liabilities", "TotalCurrentLiabilities", "CurrentLiabilities"]
            )
            if not inventory_found:
                inv_val = _lookup_balance_value(
                    sheet,
                    ["Inventory", "Total Inventory", "InventoryNet", "InventoryCurrent"],
                )
                if inv_val is not None:
                    inventory = inv_val
                    inventory_found = True
        except Exception as e:
            st.warning(f"Error extracting liquidity metrics: {e}")
            curr_assets = curr_liabs = None
            inventory_found = False
        if curr_assets is not None and curr_liabs is not None:
            break

    current_ratio = safe_div(curr_assets, curr_liabs)
    quick_numerator = curr_assets - inventory if (curr_assets is not None and inventory_found) else None
    quick_ratio = safe_div(quick_numerator, curr_liabs)
    if current_ratio is None:
        current_ratio = info.get("currentRatio")
    if quick_ratio is None:
        quick_ratio = info.get("quickRatio")

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
        "Market Cap": market_cap,
        "Company Name": company_name,
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
basket_choice = st.sidebar.selectbox(
    "Select basket",
    options=basket_names,
    index=None,
    placeholder="Pick a basket to load tickers",
)

tickers_input = ""
if basket_choice is None or basket_choice == "Custom":
    tickers_input = st.sidebar.text_input(
        "Enter comma-separated tickers",
        value="",
        placeholder="e.g. AAPL,MSFT,GOOGL",
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
elif basket_choice == "All Baskets":
    combined = []
    for members in baskets.values():
        combined.extend(members)
    # Preserve first occurrence order while removing duplicates
    tickers = list(dict.fromkeys(combined))
else:
    # Always use basket tickers if a preset basket is selected
    tickers = baskets.get(basket_choice, [])

# Metric/category selection
all_metrics = list(CATEGORY_BY_METRIC.keys())
all_categories = sorted(set(CATEGORY_BY_METRIC.values()))
selected_categories = st.sidebar.multiselect(
    "Select categories to display", options=all_categories, default=all_categories)
filtered_metrics = [m for m in all_metrics if CATEGORY_BY_METRIC[m] in selected_categories]
default_metrics = [m for m in filtered_metrics if m != "Company Name"]
if not default_metrics:
    default_metrics = filtered_metrics
selected_metrics = st.sidebar.multiselect(
    "Select metrics to display", options=filtered_metrics, default=default_metrics)
columns_to_show = ["Ticker"] + selected_metrics

# Fetch & display

with tab_fundamentals:
    st.header("Stock Fundamentals")
    st.markdown(
        "**Quick start:** Pick a basket, then sort the table to compare valuation, profitability, or growth across tickers."
    )
    with st.expander("How to use this tab", expanded=False):
        st.markdown(
            """
1. **Choose a basket** in the sidebar or enter your own tickers.
2. **Filter metrics** using the sidebar checkboxes to keep only the data you care about.
3. **Sort or chart** any metric to spot leaders, laggards, or potential bargains.
            """
        )
    if tickers:
        data = [fetch_one(tk) for tk in tickers]
        df = pd.DataFrame(data, columns=COLUMNS)

        with st.expander("Metric cheat sheet", expanded=False):
            rows = []
            for metric, info in METRIC_GUIDE.items():
                baseline_text = f"Baseline: {info['good']}" if info.get('good') else ""
                caution_text = f"Caution: {info['caution']}" if info.get('caution') else ""
                details = "  \
".join([text for text in [baseline_text, caution_text] if text])
                rows.append(
                    f"**{metric}** — {info['summary']}" + (f"  \
{details}" if details else "")
                )
            st.markdown("\n".join(rows))
            st.caption("Values with a soft green highlight fall inside the healthy range noted above.")

        # Filter columns based on user selection
        df = df[columns_to_show]

        # Sorting option (sort before formatting)
        sort_metric = st.sidebar.selectbox("Sort by metric", options=columns_to_show[1:], index=0)
        sort_ascending = st.sidebar.checkbox("Sort ascending", value=True)
        df = df.sort_values(by=sort_metric, ascending=sort_ascending, na_position='last')

        df = df.copy()
        strong_flags = pd.DataFrame(False, index=df.index, columns=df.columns)
        for idx in df.index:
            for col in df.columns:
                metric_name = col
                raw_val = df.at[idx, col]
                strong = metric_is_strong(metric_name, raw_val)
                strong_flags.at[idx, col] = strong
                if metric_name in ["Avg Vol (10d)", "Avg Vol (3m)"]:
                    try:
                        formatted = (
                            f"{int(float(raw_val)):,}"
                            if raw_val not in [None, "None", "nan"]
                            and str(raw_val).replace(",", "").replace(".", "").isdigit()
                            else raw_val
                        )
                    except Exception:
                        formatted = raw_val
                elif metric_name == "Market Cap":
                    formatted = format_market_cap(raw_val)
                elif isinstance(raw_val, (int, float)) and not pd.isnull(raw_val):
                    formatted = f"{raw_val:.2f}"
                else:
                    formatted = raw_val
                df.at[idx, col] = formatted

        # Create MultiIndex columns: first row = categories, second = metric names
        metric_names = df.columns.tolist()
        categories = [""] + [CATEGORY_BY_METRIC.get(col, "") for col in metric_names[1:]]
        multi_index = pd.MultiIndex.from_tuples(
            zip(categories, metric_names), names=["Category", "Metric"])
        df.columns = multi_index
        strong_flags.columns = multi_index

        st.success(f"Fetched data for {len(tickers)} tickers.")
        def highlight_strong(row):
            return [
                "background-color: rgba(0,150,0,0.15)" if strong_flags.loc[row.name, col] else ""
                for col in df.columns
            ]

        st.dataframe(df.style.apply(highlight_strong, axis=1), use_container_width=True)

        # Bar chart visualization
        chart_options = [m for m in columns_to_show[1:] if m != "Company Name"]
        if chart_options:
            chart_metric = st.sidebar.selectbox("Bar chart metric", options=chart_options, index=0)
            st.subheader(f"Bar Chart: {chart_metric}")
            # Access MultiIndex columns correctly
            ticker_col = ("", "Ticker")
            metric_col = (CATEGORY_BY_METRIC.get(chart_metric, ""), chart_metric)
            tickers_list = df[ticker_col].tolist()
            metric_values = [parse_display_number(x) for x in df[metric_col].values]
            # Replace missing or non-numeric values with zero (after building metric_values)
            metric_values = [v if v is not None and not pd.isnull(v) else 0 for v in metric_values]
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
        else:
            st.info("Select at least one numeric metric to plot the bar chart.")

        st.download_button("Download CSV",
                           df.to_csv(index=False),
                           file_name="fundamentals.csv",
                           mime="text/csv")
    else:
        st.info("Select a basket or enter tickers in the sidebar to load data.")
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
    with st.expander("What the factors mean", expanded=False):
        for name, desc in FACTOR_DESCRIPTIONS.items():
            st.markdown(f"**{name}** — {desc}")

    if tickers:
        if "value_weight_slider" not in st.session_state:
            st.session_state["value_weight_slider"] = 0.33
        if "quality_weight_slider" not in st.session_state:
            st.session_state["quality_weight_slider"] = 0.33
        if "momentum_weight_slider" not in st.session_state:
            st.session_state["momentum_weight_slider"] = 0.34

        preset_cols = st.columns(3)
        if preset_cols[0].button("Balanced (33/33/34)"):
            st.session_state["value_weight_slider"] = 0.33
            st.session_state["quality_weight_slider"] = 0.33
            st.session_state["momentum_weight_slider"] = 0.34
        if preset_cols[1].button("Value Tilt"):
            st.session_state["value_weight_slider"] = 0.5
            st.session_state["quality_weight_slider"] = 0.3
            st.session_state["momentum_weight_slider"] = 0.2
        if preset_cols[2].button("Momentum Tilt"):
            st.session_state["value_weight_slider"] = 0.2
            st.session_state["quality_weight_slider"] = 0.2
            st.session_state["momentum_weight_slider"] = 0.6

        weight_cols = st.columns(3)
        value_weight = weight_cols[0].slider(
            "Value weight", 0.0, 1.0, st.session_state["value_weight_slider"], 0.05, key="value_weight_slider"
        )
        quality_weight = weight_cols[1].slider(
            "Quality weight", 0.0, 1.0, st.session_state["quality_weight_slider"], 0.05, key="quality_weight_slider"
        )
        momentum_weight = weight_cols[2].slider(
            "Momentum weight", 0.0, 1.0, st.session_state["momentum_weight_slider"], 0.05, key="momentum_weight_slider"
        )
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

            history_tickers = list(dict.fromkeys(tickers + ["^GSPC", "SPY"]))
            price_history = fetch_price_history(history_tickers, period="2y")
            twelve_month_returns = {}
            benchmark_used = None
            if not price_history.empty:
                for tk in tickers:
                    if tk in price_history.columns:
                        series = price_history[tk].dropna()
                        if len(series) > 250:
                            series = series.iloc[-252:]
                        if len(series) >= 2:
                            twelve_month_returns[tk] = series.iloc[-1] / series.iloc[0] - 1
                benchmark_candidates = [col for col in ["^GSPC", "SPY"] if col in price_history.columns]
                if benchmark_candidates:
                    benchmark_used = benchmark_candidates[0]
                    sp_series = price_history[benchmark_used].dropna()
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
            if market_return is not None and benchmark_used:
                score_df.loc[momentum_series.index, "Beating Market"] = (
                    momentum_series > market_return
                )
                st.caption(
                    f"Benchmark ({benchmark_used}) 12-month return: {market_return * 100:.2f}%"
                    if market_return is not None
                    else "Benchmark 12-month return unavailable."
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
            st.caption("Tip: Look for checkmarks in the Top Decile column to find the strongest multi-factor names.")

            if top_tickers and not price_history.empty:
                st.subheader("Rolling 3-Month Returns vs S&P 500")
                st.markdown(
                    "The chart compares an equal-weighted portfolio of the top decile tickers "
                    "against the S&P 500 using 3-month rolling returns. Positive spreads indicate periods "
                    "where the factor blend outperformed the broad market."
                )
                available_cols = [col for col in top_tickers if col in price_history.columns]
                benchmark_candidates = [col for col in ["^GSPC", "SPY"] if col in price_history.columns]
                benchmark_col = benchmark_candidates[0] if benchmark_candidates else None
                if not available_cols or benchmark_col is None:
                    st.info("Need benchmark data (S&P 500 or SPY) and valid top-decile price history to plot rolling returns.")
                else:
                    aligned_history = price_history[available_cols + [benchmark_col]].dropna(how="all")
                    portfolio_returns = aligned_history[available_cols].pct_change().dropna()
                    if not portfolio_returns.empty and benchmark_col in aligned_history.columns:
                        equal_weight_returns = portfolio_returns.mean(axis=1)
                        market_returns_series = aligned_history[benchmark_col].pct_change().dropna()
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
                                market_rolling.rename(benchmark_col),
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
        st.info("Select a basket or enter tickers in the sidebar to load data.")

with tab_regime:
    st.header("Macro Regime Insights")
    st.markdown(
        "Blend macro signals with your equity baskets. The indicators below highlight whether to lean "
        "defensive, cyclical, commodity-heavy, or healthcare-focused this month."
    )
    st.info(
        "Signals update with the latest data from the Federal Reserve Economic Data (FRED) API. "
        "Green tiles indicate a benign backdrop; amber or red suggest caution and potential allocation shifts."
    )
    with st.expander("How to interpret the signals", expanded=False):
        st.markdown(
            """
1. **Check the indicator tiles** – green values = supportive backdrop, amber/red = caution.
2. **Review the suggested tilt** – compare weights to the equal-weight baseline.
3. **Read the model notes** – see which macro triggers caused the shift.
            """
        )

    errors = []
    yield_curve_df, yield_err = fetch_fred_series("T10Y2Y", observation_start="2010-01-01")
    if yield_err:
        errors.append(yield_err)
    cpi_df, cpi_err = fetch_fred_series("CPIAUCSL", observation_start="2010-01-01")
    if cpi_err:
        errors.append(cpi_err)
    pmi_df, pmi_err = fetch_fred_series("NAPM", observation_start="2010-01-01")
    if pmi_err or pmi_df.empty:
        fallback_df, fallback_err = fetch_fred_series("ISM/MAN_PMI", observation_start="2010-01-01")
        if fallback_err:
            errors.append(f"PMI fallback: {fallback_err}")
        else:
            pmi_df = fallback_df
            pmi_err = None
    if pmi_err:
        errors.append(pmi_err)

    if errors:
        st.warning(
            "Some macro series were unavailable. Missing datasets were skipped so the model can still run.\n"
            + "\n".join(f"- {err}" for err in errors)
        )

    yield_value, yield_date = latest_value(yield_curve_df)
    cpi_latest, cpi_date = latest_value(cpi_df)
    pmi_value, pmi_date = latest_value(pmi_df)

    cpi_yoy = None
    if cpi_df is not None and not cpi_df.empty:
        cpi_series = cpi_df["value"].dropna()
        if len(cpi_series) > 12:
            cpi_yoy = cpi_series.iloc[-1] / cpi_series.iloc[-13] - 1

    yield_inverted = yield_value is not None and yield_value < 0
    inflation_hot = cpi_yoy is not None and cpi_yoy > 0.03
    pmi_contraction = pmi_value is not None and pmi_value < 50

    indicator_cols = st.columns(3)

    def format_date(dt: datetime | None) -> str:
        return dt.strftime("%Y-%m-%d") if dt else "N/A"

    indicator_cols[0].metric(
        "10Y-2Y Treasury Spread",
        f"{yield_value:.2f}%" if yield_value is not None else "N/A",
    )
    indicator_cols[0].write(f"Last update: {format_date(yield_date)}")
    indicator_cols[0].markdown("<small>Negative = inverted curve, historically a recession warning.</small>", unsafe_allow_html=True)

    indicator_cols[1].metric(
        "CPI YoY",
        f"{cpi_yoy * 100:.2f}%" if cpi_yoy is not None else "N/A",
    )
    indicator_cols[1].write(f"Last update: {format_date(cpi_date)}")
    indicator_cols[1].markdown("<small>>3% suggests inflation pressures favouring real assets.</small>", unsafe_allow_html=True)

    indicator_cols[2].metric(
        "ISM Manufacturing PMI",
        f"{pmi_value:.1f}" if pmi_value is not None else "N/A",
    )
    indicator_cols[2].write(f"Last update: {format_date(pmi_date)}")
    indicator_cols[2].markdown("<small><50 implies manufacturing contraction.</small>", unsafe_allow_html=True)

    allocations = {
        "Defensives": 0.2,
        "Cyclicals": 0.2,
        "Commodities & Energy": 0.2,
        "Healthcare": 0.2,
        "Growth & Innovation": 0.2,
    }
    rationale = []

    def shift_weight(source: str, target: str, pct: float):
        if allocations.get(source, 0) <= 0:
            return
        take = min(pct, allocations[source])
        allocations[source] -= take
        allocations[target] = allocations.get(target, 0) + take

    if yield_inverted:
        shift_weight("Cyclicals", "Defensives", 0.1)
        rationale.append("Yield curve inverted → tilt toward defensives and lower-volatility holdings.")

    if inflation_hot:
        shift_weight("Defensives", "Commodities & Energy", 0.1)
        rationale.append("Inflation above target → increase exposure to commodities and energy plays.")

    if pmi_contraction:
        shift_weight("Cyclicals", "Healthcare", 0.1)
        rationale.append("PMI below 50 → reallocate from cyclicals into healthcare resilience.")

    allocations = {k: max(v, 0) for k, v in allocations.items()}
    total_alloc = sum(allocations.values())
    if total_alloc > 0:
        allocations = {k: v / total_alloc for k, v in allocations.items()}

    alloc_df = (
        pd.DataFrame.from_dict(allocations, orient="index", columns=["Suggested Weight"])
        .sort_values("Suggested Weight", ascending=False)
    )

    st.subheader("Suggested Factor Tilt")
    st.markdown(
        "Weights show the model's preferred tilt given current macro readings. Start from equal weights and apply the adjustments below."
    )
    st.dataframe(
        alloc_df.style.format({"Suggested Weight": "{:.0%}"}),
        use_container_width=True,
    )

    st.subheader("Model Notes")
    if rationale:
        for note in rationale:
            st.markdown(f"- {note}")
    else:
        st.markdown("- Indicators sit in neutral ranges → stay close to baseline allocations.")

    st.markdown(
        """
**How to act:**

- **Defensives** → Think staples, utilities, large-cap quality (e.g., use the *Europe Granolas* basket).
- **Commodities & Energy** → Use resource-heavy baskets like *Nuclear* when inflation runs hot.
- **Healthcare** → Defensive demand when growth slows; consider building a dedicated basket.
- **Growth & Innovation** → Keep exposure through high-beta names such as the *AI Leaders* basket, trimming when the curve inverts.
- **Cyclicals** → Industrials, consumer discretionary; lighten up when PMI dips below 50.
"""
    )

    st.caption("Benchmarks: Yield curve (T10Y2Y), CPI (CPIAUCSL), PMI (NAPM) sourced from FRED. Updates typically monthly.")

with tab_ownership:
    st.header("SEC Ownership Data")
    st.info("Displays insider trading activity from recent SEC filings.")
    st.markdown("""
**How to use this tab:**

- Only the first ticker in your selection is shown here. If you select multiple stocks, only the first one will be used for SEC filings.
- To view ownership data for a specific stock, filter to a single ticker using the sidebar basket or by entering just one ticker in the input box.
""")
    with st.expander("Quick start", expanded=False):
        st.markdown(
            """
1. **Choose one ticker** in the sidebar (Form 4s show for the first symbol).
2. **Set a date range** to focus on recent insider trades.
3. **Open filing links** to read the actual SEC documents in a new tab.
            """
        )
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
        st.info("Select a basket or enter tickers in the sidebar to load data.")

with tab_earnings:
    st.header("Earnings Highlights")
    st.info("Pulls the latest 8-K Item 2.02 (earnings release) to surface a quick snippet and links.")
    with st.expander("Quick start", expanded=False):
        st.markdown(
            """
1. **Select a ticker** (first symbol in your basket is used).
2. **Pick an earnings filing** from the dropdown to read the highlights.
3. **Follow the links** to the SEC filing or press release for full details.
            """
        )
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
        st.info("Select a basket or enter tickers in the sidebar to load data.")

with tab_glossary:
    st.header("Guide & Glossary")
    st.markdown(
        "Use this cheat sheet to decode the metrics, factors, and macro signals used across the dashboard. "
        "Share it with teammates who are new to equity analysis so everyone speaks the same language."
    )

    st.subheader("Playbook: How to explore a basket")
    st.markdown(
        """
1. **Scan fundamentals** – Sort the table by valuation or profitability to spot standout names.
2. **Blend factors** – Use the smart beta tab to tilt toward value, quality, or momentum.
3. **Check the macro backdrop** – Note whether the regime tab suggests being defensive or risk-on.
4. **Dive into filings** – Review insider trades and earnings releases for catalysts.
        """
    )

    st.subheader("Fundamental metrics at a glance")
    metric_rows = []
    for metric, info in METRIC_GUIDE.items():
        metric_rows.append(
            {
                "Metric": metric,
                "What it means": info.get("summary", ""),
                "Healthy range": info.get("good", ""),
                "Watch out": info.get("caution", ""),
            }
        )
    metric_df = pd.DataFrame(metric_rows)
    st.dataframe(metric_df, use_container_width=True)

    st.subheader("Factor definitions")
    factor_rows = [
        {"Factor": name, "Plain-English explanation": desc}
        for name, desc in FACTOR_DESCRIPTIONS.items()
    ]
    st.table(pd.DataFrame(factor_rows))

    st.subheader("Macro signal cheat sheet")
    macro_rows = []
    for name, info in MACRO_GUIDE.items():
        macro_rows.append(
            {
                "Indicator": name,
                "Why it matters": info.get("summary", ""),
                "Healthy": info.get("good", ""),
                "Caution": info.get("caution", ""),
            }
        )
    st.table(pd.DataFrame(macro_rows))

    st.subheader("FAQ")
    st.markdown(
        """
**Where does the data come from?**  
Market and fundamentals: Yahoo Finance (`yfinance`). Filings: SEC’s `sec-api`. Macro: Federal Reserve Economic Data (FRED).

**What if a value looks odd or missing?**  
Some companies do not report certain items every quarter. Values marked `None` mean data was unavailable from the source.

**How often should I refresh the factors?**  
Monthly works well—rerun the tab after earnings season or major macro events.
        """
    )

    st.caption("Tip: Save or print this tab for onboarding new analysts.")
