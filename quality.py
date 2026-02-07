"""Quality gate — hard pass/fail floor. If a stock doesn't clear this, it
never appears as a buy-the-dip candidate. Period.

Philosophy: we can't beat Wall Street, so we don't try to score or rank.
We just filter out businesses that aren't real businesses, and let the
price action tell us when the survivors are cheap.

Every gate uses data freely available from yfinance.
"""


# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------
# Each gate is (label, test_fn, fail_reason).
# A stock must pass ALL gates to appear in alerts.

GATES = [
    (
        "Profitable",
        lambda r: r.get("net_margin") is not None and r["net_margin"] > 0,
        "Negative net margin — not profitable",
    ),
    (
        "Generates cash",
        lambda r: r.get("fcf") is not None and r["fcf"] > 0,
        "Negative free cash flow — burning cash",
    ),
    (
        "Revenue growing",
        lambda r: r.get("revenue_growth") is not None and r["revenue_growth"] > 0,
        "Revenue is shrinking",
    ),
    (
        "Debt manageable",
        lambda r: r.get("debt_to_equity") is None or r["debt_to_equity"] < 200,
        "Debt/equity above 200% — overleveraged",
    ),
    (
        "Can pay its bills",
        lambda r: (r.get("current_ratio") is None or r["current_ratio"] >= 1.0)
                  or (r.get("fcf") is not None and r["fcf"] > 0),
        "Current ratio below 1 AND negative free cash flow — can't cover obligations",
    ),
    (
        "Not a penny stock",
        lambda r: r.get("market_cap") is not None and r["market_cap"] >= 500_000_000,
        "Market cap under $500M",
    ),
]


def run_quality_gate(row: dict) -> list[dict]:
    """Return a list of {name, passed, reason} for each gate."""
    results = []
    for label, test_fn, fail_reason in GATES:
        passed = False
        try:
            passed = test_fn(row)
        except Exception:
            passed = False
        results.append({
            "name": label,
            "passed": passed,
            "reason": None if passed else fail_reason,
        })
    return results


def passes_quality(row: dict) -> bool:
    """True only if the stock passes ALL gates. No exceptions."""
    return all(g["passed"] for g in run_quality_gate(row))


def quality_summary(row: dict) -> str:
    """Short string like '6/6' or '4/6'."""
    results = run_quality_gate(row)
    passed = sum(1 for g in results if g["passed"])
    return f"{passed}/{len(results)}"


def quality_failures(row: dict) -> list[str]:
    """Return list of human-readable failure reasons."""
    return [g["reason"] for g in run_quality_gate(row) if not g["passed"]]
