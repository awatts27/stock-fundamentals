# Stock Watchlist App — Build Prompt

## Philosophy

We can't beat Wall Street. We don't try to score, rank, or outsmart the market. We just want to:
1. Identify **good businesses** (profitable, generating cash, growing, not overleveraged).
2. Buy them **when they're cheap** (pulled back from recent highs).
3. Understand **why they're cheap** (market-wide dip, sector dip, or company-specific problem).

The app should feel like a calm, opinionated filter — not a Bloomberg terminal. Show less, not more.

---

## Tech Stack

- **Python 3.11+**
- **Streamlit** (web UI framework)
- **yfinance** (free stock data — prices, fundamentals, company descriptions)
- **pandas / numpy** (data manipulation)
- **altair** (interactive charts)
- No paid APIs. No API keys. No accounts required.

Dependencies (requirements-v2.txt):
```
yfinance
pandas
numpy
streamlit
altair
```

---

## File Structure

```
app.py                  # Streamlit UI — 4 tabs
data.py                 # All yfinance calls, caching, dip classification
quality.py              # Hard pass/fail quality gates
pullbacks.py            # Pullback detection from rolling highs
relative_strength.py    # Basket vs SPY comparison
baskets.json            # Stock baskets grouped by sector
requirements-v2.txt     # Python dependencies
```

---

## Module 1: `baskets.json`

A flat JSON object mapping sector names to ticker arrays. Example:

```json
{
    "My Portfolio": ["AAPL", "MSFT", "NVDA"],
    "Technology": ["AAPL", "MSFT", "GOOG", "META", "AMZN", "CRM", "ADBE"],
    "Semiconductors": ["NVDA", "AMD", "TSM", "ASML", "AVGO", "QCOM"],
    "Healthcare": ["UNH", "JNJ", "LLY", "ABT", "TMO"],
    "Fintech": ["SQ", "PYPL", "SOFI", "COIN", "SPGI"]
}
```

Rules:
- First basket should be "My Portfolio" (personal holdings).
- Other baskets should be recognized sectors (Technology, Semiconductors, Healthcare, etc.).
- No ETFs — only individual stocks (ETFs don't have proper fundamentals in yfinance).
- Stocks can appear in multiple baskets (e.g., NVDA in both My Portfolio and Semiconductors).

---

## Module 2: `data.py`

All yfinance interaction lives here. The rest of the app never calls yfinance directly.

### Functions to implement:

**`load_baskets(path="baskets.json") -> dict[str, list[str]]`**
- Load baskets from JSON file. Return empty dict if file doesn't exist.

**`all_tickers(baskets) -> list[str]`**
- Return a deduplicated, order-preserving list of all tickers across all baskets.

**`fetch_price_history(tickers, period="1y") -> pd.DataFrame`**
- Use `yf.download()` with `auto_adjust=True, progress=False`.
- Cache with `@st.cache_data(ttl=3600)`.
- Handle yfinance's inconsistent return format:
  - MultiIndex columns for multiple tickers (extract "Close" level).
  - Single Series for one ticker (convert to DataFrame).
- Return a DataFrame with Date index and one column per ticker (column names as strings).

**`fetch_fundamentals(ticker) -> dict`**
- Use `yf.Ticker(ticker).info`.
- Cache with `@st.cache_data(ttl=3600, show_spinner=False)`.
- Extract and return these fields (calculate where needed):
  - `ticker`, `name` (longName or shortName), `summary` (longBusinessSummary)
  - `sector`, `industry`
  - `current_price`, `market_cap`, `high_52w`, `low_52w`
  - `pct_below_high` — calculated: `(high - current) / high * 100`
  - `pe_ttm`, `forward_pe`, `peg`, `pb`
  - `dividend_yield` — convert from decimal to percentage
  - `net_margin` — from `profitMargins`, convert to percentage
  - `roe` — from `returnOnEquity`, convert to percentage
  - `revenue_growth`, `earnings_growth` — convert to percentage
  - `debt_to_equity` — normalize: if value < 5, multiply by 100 (yfinance returns inconsistent formats)
  - `current_ratio`
  - `fcf` — raw freeCashflow value (for quality gate)
  - `fcf_margin` — calculated: `fcf / totalRevenue * 100`
  - `beta`, `avg_volume`
  - `short_pct_float` — convert to percentage
  - `analyst_target`, `analyst_upside` — calculated: `(target - price) / price * 100`
  - `held_pct_insiders` — convert to percentage
  - `is_etf` — True if quoteType is ETF, MUTUALFUND, or INDEX

**`fetch_all_fundamentals(tickers) -> pd.DataFrame`**
- Loop over tickers, call `fetch_fundamentals()` for each.
- Catch exceptions per ticker — add a minimal fallback row `{ticker, name}` on failure.
- Return DataFrame of all results.

**`classify_dip(prices, ticker, basket_tickers, lookback=21) -> str`**
- Determine WHY a stock is down. Return one of:
  - `"market"` — SPY is also down > 3% over the lookback window (broad selloff, safest to buy)
  - `"sector"` — more than 50% of the basket is down > 3% (sector rotation)
  - `"stock"` — only this stock is down (company-specific problem, most dangerous)
  - `"unknown"` — insufficient data
- Check SPY first, then basket peers, default to stock-specific.

**`DIP_LABELS`** — dict mapping dip type codes to display labels:
```python
{"market": "Market-wide dip", "sector": "Sector dip", "stock": "Stock-specific", "unknown": "Unknown"}
```

Also include momentum/volatility/drawdown helpers if needed for the Basket Overview tab:
- `compute_momentum(prices, periods)` — percentage return over N days
- `compute_volatility(prices, periods)` — annualized volatility
- `compute_drawdown_from_peak(prices)` — current drawdown from cumulative max

---

## Module 3: `quality.py`

Hard pass/fail quality gates. A stock must pass ALL gates to appear in alerts. No scoring, no partial credit. The goal is to exclude bad investments, not rank good ones. We'd rather miss a good stock than recommend a bad one.

### 6 Quality Gates:

1. **Profitable** — `net_margin > 0`. Not profitable = not a real business.
2. **Generates cash** — `fcf > 0`. Negative free cash flow = burning money.
3. **Revenue growing** — `revenue_growth > 0`. Shrinking revenue = declining business.
4. **Debt manageable** — `debt_to_equity < 200%` (or data missing, which passes). Overleveraged = fragile.
5. **Can pay its bills** — `current_ratio >= 1.0` OR `fcf > 0`. A low current ratio is OK if the company generates strong cash flow (e.g., Apple runs a current ratio of ~0.87 but generates $100B+ in FCF). Only fail if BOTH liquidity is low AND cash flow is negative.
6. **Not a penny stock** — `market_cap >= $500M`. Too small = too risky for this approach.

### Functions:

**`run_quality_gate(row: dict) -> list[dict]`**
- Returns `[{name, passed, reason}, ...]` for each gate.
- Wraps each test in try/except — exceptions count as failures.

**`passes_quality(row: dict) -> bool`**
- True only if ALL gates pass.

**`quality_summary(row: dict) -> str`**
- Returns "6/6" or "4/6" etc.

**`quality_failures(row: dict) -> list[str]`**
- Returns list of human-readable failure reasons for failed gates only.

---

## Module 4: `pullbacks.py`

Simple pullback detection. Only called on stocks that already passed quality gates.

### Function:

**`detect_pullbacks(prices, threshold_pct=20.0, lookback_days=63) -> pd.DataFrame`**
- Find the rolling high within the lookback window.
- Calculate drop percentage: `(rolling_high - current) / rolling_high * 100`.
- Return rows for tickers where drop >= threshold.
- Sort by drop_pct descending (biggest drops first).
- Return columns: `ticker, current_price, rolling_high, drop_pct`.

---

## Module 5: `relative_strength.py`

Basket-level performance analysis for the Basket Overview tab.

### Functions:

**`basket_returns(prices, baskets, periods=None) -> pd.DataFrame`**
- Equal-weighted return for each basket over 1M (21d), 3M (63d), 6M (126d), 12M (252d).
- Returns DataFrame with columns: basket, stocks (count), 1M, 3M, 6M, 12M.

**`basket_cumulative_returns(prices, baskets) -> pd.DataFrame`**
- Daily cumulative equal-weighted return for each basket, normalized to day 1.
- Include SPY as a benchmark column.
- Returns DataFrame indexed by date, one column per basket + SPY.

**`vs_spy(prices, periods=None) -> pd.DataFrame`** (optional, available for Stock Detail)
- Each ticker's return minus SPY return for each period.

---

## Module 6: `app.py` — Streamlit UI

### Page config:
- `page_title="Stock Watchlist"`, `layout="wide"`
- Title: "Stock Watchlist"
- Caption: "Good businesses on sale — nothing more, nothing less"

### Sidebar:
- **Pullback threshold %** — slider, range 10-50, default 20, step 5
- **Lookback window (trading days)** — slider, range 21-126, default 63, step 21

### Data loading:
- Fetch all unique tickers + SPY in one `fetch_price_history()` call (1 year).
- Fetch all fundamentals via `fetch_all_fundamentals()`.
- Show a spinner during loading.

### 4 Tabs:

#### Tab 1: "On Sale" (main tab)
The core of the app. Two-stage filter:

**Stage 1: Quality filter**
- Loop through all fundamentals. Keep only stocks that pass ALL quality gates.
- Skip ETFs.

**Stage 2: Pullback detection**
- Run `detect_pullbacks()` on quality-passing stocks only.
- For each pullback, enrich with fundamentals and classify the dip.

**Display:**
- Custom column layout (not a dataframe) so we can use popovers.
- Columns: Ticker, Name, Price, Drop, Net Margin, FCF, Rev Growth, Dip Type.
- **Ticker column**: Each ticker is a `st.popover()` button. Clicking it shows:
  - Company name (bold)
  - Sector / Industry
  - First 500 characters of business description from yfinance
- Below the table: Dip Type guide explaining market/sector/stock-specific.
- Caption: "Click any ticker to see what the company does."

**Filtered stocks expander:**
- At the bottom, show an expandable section: "X stocks hidden (failed quality gate)".
- Inside: table of failed stocks with columns: Ticker, Name, Why (semicolon-separated failure reasons).

**Empty states:**
- No quality stocks: "No stocks in your baskets pass all quality gates right now."
- No pullbacks: "No quality stocks have pulled back X%+ right now. Your baskets are holding up."

#### Tab 2: "Stock Detail"
Deep dive into any single stock.

- **Dropdown**: Sorted list of all tickers.
- **Company description**: Expandable "About [Name]" section with sector/industry and full business summary.
- **Header metrics** (4 columns): Price, Market Cap, % Below 52W High, Quality score (e.g., "6/6").
- **Pass/fail banner**: Green success if passes all gates, red error if fails (with message about alert eligibility).
- **Quality gate breakdown**: One column per gate with checkmark/X icon and failure reason caption.
- **Key metrics** (3 rows of 4 columns):
  - Row 1: Net Margin, Free Cash Flow, FCF Margin, Revenue Growth
  - Row 2: P/E (TTM), Forward P/E, ROE, D/E
  - Row 3: Current Ratio, Analyst Upside, Short % Float, Beta
- **Dip context section**:
  - Classify the dip and show a color-coded message:
    - Market-wide (blue/info): "SPY is also down. Historically the safest time to buy quality."
    - Sector (yellow/warning): "The [basket] basket is broadly down. Could be sector rotation."
    - Stock-specific (red/error): "Only this stock is falling while its peers are fine. Most dangerous type of dip."
  - Show basket membership.
- **Price chart**: 1-year interactive Altair line chart with tooltips.

#### Tab 3: "Basket Overview"
How each sector basket is performing.

- **Returns table**: Basket returns for 1M, 3M, 6M, 12M formatted as "+X.X%" or "—".
- **Cumulative returns chart**: Multi-line Altair chart showing all baskets + SPY over 1 year. Interactive with tooltips.
- **Sector rotation heatmap**: Altair rect chart colored by return (red-yellow-green scale, domainMid=0). Periods on x-axis, baskets on y-axis, with text overlay showing return values.

#### Tab 4: "Baskets"
Simple reference view.

- For each basket: expandable section showing basket name, stock count, and comma-separated tickers.

---

## Formatting helpers needed in `app.py`:

- `_fmt_cap(val)` — Market cap: "$1.5T", "$42.3B", "$850M", "N/A"
- `_fmt_pct(val)` — Percentages: "+12.3%", "-5.1%", "—"
- `_fmt_cash(val)` — Cash values: "$15.2B", "$340M", "-$2.1B", "—"

---

## Key Design Principles

1. **Quality first, price second.** Never show a bad business as a buying opportunity, no matter how far it's fallen.
2. **All-or-nothing gates.** No scoring, no weighted averages. Either a stock passes or it doesn't. We'd rather miss Apple than recommend FUBO.
3. **Context matters.** A dip during a market selloff is very different from a stock dropping alone. Label which type it is.
4. **Free data only.** Everything comes from yfinance. No API keys, no accounts, no paid services.
5. **Show less, not more.** The app should surface a handful of actionable ideas, not a wall of data. The On Sale tab might show 0-5 stocks on a normal day. That's a feature, not a bug.
6. **Simple enough to explain.** If you can't explain the methodology in 30 seconds, it's too complex. Ours is: "Is it a good business? Is it on sale? Why is it on sale?"

---

## Running the app

```bash
pip install -r requirements-v2.txt
streamlit run app.py
```

Opens at http://localhost:8501. No configuration needed.
