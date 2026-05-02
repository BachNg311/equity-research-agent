"""Validators for fundamental_analyst output (fundamental_analysis.md).

Each `check_*` returns a failure message string, or None on success.
"""
from __future__ import annotations
import re

REQUIRED_RATIO_LABELS = [
    "P/E", "P/B", "ROE", "ROA", "EPS", "D/E", "EV/EBITDA", "Profit Margin",
]

VERDICT_RE = re.compile(r"undervalued|overvalued|fairly\s+valued", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+\.?\d*")

SANE_RANGES = {
    "P/E":   (-100, 1000),
    "P/B":   (-10, 200),
    "ROE":   (-200, 500),     # ROE in % terms (yfinance returns 1.6 = 160%)
    "EPS":   (-1000, 1000),
    "D/E":   (-100, 1000),
}


def check_has_all_ratio_labels(text: str) -> str | None:
    missing = [label for label in REQUIRED_RATIO_LABELS if label not in text]
    if missing:
        return f"missing ratio labels: {missing}"
    return None


def check_has_valuation_verdict(text: str) -> str | None:
    if not VERDICT_RE.search(text):
        return "no valuation verdict (undervalued/overvalued/fairly valued)"
    return None


def check_word_count(text: str, minimum: int = 800) -> str | None:
    n = len(text.split())
    if n < minimum:
        return f"word count {n} < {minimum}"
    return None


def _extract_ratio_value(text: str, label: str) -> float | None:
    """Find first numeric value following a ratio label."""
    pattern = rf"{re.escape(label)}[^\d\-]*(-?\d+\.?\d*)"
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def check_ratio_values_are_sane(text: str) -> str | None:
    bad = []
    for label, (lo, hi) in SANE_RANGES.items():
        val = _extract_ratio_value(text, label)
        if val is None:
            continue
        if not (lo <= val <= hi):
            bad.append(f"{label}={val} (expected in [{lo}, {hi}])")
    if bad:
        return f"out-of-range ratios: {bad}"
    return None


def check_sector_matches_yfinance(text: str, ticker: str) -> str | None:
    """Cross-reference: sector mentioned in output should match yfinance."""
    try:
        import yfinance as yf
        actual_sector = yf.Ticker(ticker).info.get("sector", "")
    except Exception as e:
        return None  # if yfinance is down, skip rather than fail
    if not actual_sector:
        return None
    if actual_sector.lower() not in text.lower():
        return f"sector '{actual_sector}' not mentioned in output"
    return None


def check_ticker_mentioned(text: str, ticker: str) -> str | None:
    if ticker.upper() not in text.upper():
        return f"ticker {ticker} not mentioned"
    return None
