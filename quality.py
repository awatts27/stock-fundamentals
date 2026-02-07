"""Quality gate — simple pass/fail checks before alerting on a dip."""

import pandas as pd

# Thresholds are deliberately lenient. The goal is to filter out obvious
# garbage, not to find the "best" stock. If a company passes all gates it
# simply means: this dip is worth investigating, not that it's a buy.

GATES = {
    "profitable":       lambda r: r.get("net_margin") is not None and r["net_margin"] > 0,
    "revenue_growing":  lambda r: r.get("revenue_growth") is not None and r["revenue_growth"] > 0,
    "not_overleveraged": lambda r: r.get("debt_to_equity") is None or r["debt_to_equity"] < 250,
    "has_liquidity":    lambda r: r.get("current_ratio") is None or r["current_ratio"] > 0.8,
    "not_penny_stock":  lambda r: r.get("market_cap") is not None and r["market_cap"] > 500_000_000,
}


def run_quality_gate(row: dict) -> dict:
    """Return {gate_name: bool} for each check."""
    return {name: fn(row) for name, fn in GATES.items()}


def passes_quality(row: dict, min_pass: int = 3) -> bool:
    """True if at least `min_pass` gates are satisfied."""
    results = run_quality_gate(row)
    return sum(results.values()) >= min_pass


def quality_summary(row: dict) -> str:
    results = run_quality_gate(row)
    passed = sum(results.values())
    total = len(results)
    return f"{passed}/{total}"


def quality_detail(row: dict) -> list[str]:
    results = run_quality_gate(row)
    labels = {
        "profitable": "Profitable",
        "revenue_growing": "Revenue growing",
        "not_overleveraged": "Debt manageable",
        "has_liquidity": "Liquidity OK",
        "not_penny_stock": "Market cap > $500M",
    }
    out = []
    for name, passed in results.items():
        icon = "pass" if passed else "fail"
        out.append(f"{icon}: {labels.get(name, name)}")
    return out
