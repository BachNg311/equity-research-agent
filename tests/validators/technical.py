"""Validators for technical_analyst output (technical_analysis.md)."""
from __future__ import annotations
import re

REQUIRED_INDICATORS = [
    r"SMA[^a-zA-Z]*20",
    r"SMA[^a-zA-Z]*50",
    r"SMA[^a-zA-Z]*200",
    r"RSI",
    r"MACD",
    r"Bollinger",
]

TREND_RE = re.compile(
    r"BULLISH|BEARISH|NEUTRAL|uptrend|downtrend|sideways", re.IGNORECASE
)
PRICE_RE = re.compile(r"\$\s*\d+\.?\d*")


def check_has_all_indicators(text: str) -> str | None:
    missing = []
    for pattern in REQUIRED_INDICATORS:
        if not re.search(pattern, text, re.IGNORECASE):
            missing.append(pattern)
    if missing:
        return f"missing indicators: {missing}"
    return None


def check_has_trend_label(text: str) -> str | None:
    if not TREND_RE.search(text):
        return "no trend label (BULLISH/BEARISH/NEUTRAL/uptrend/downtrend/sideways)"
    return None


def check_has_support_resistance_levels(text: str, minimum: int = 2) -> str | None:
    prices = PRICE_RE.findall(text)
    if len(prices) < minimum:
        return f"only {len(prices)} price levels found, need >= {minimum}"
    return None


def check_ticker_mentioned(text: str, ticker: str) -> str | None:
    if ticker.upper() not in text.upper():
        return f"ticker {ticker} not mentioned"
    return None


def check_word_count(text: str, minimum: int = 600) -> str | None:
    n = len(text.split())
    if n < minimum:
        return f"word count {n} < {minimum}"
    return None


def check_indicator_values_match_tool(text: str, ticker: str, tolerance: float = 0.10) -> str | None:
    """Run USTechDataTool independently and compare SMA-20.

    Tolerance is 10% by default to allow for slight day-of-fetch differences.
    """
    try:
        from stock_advisor.tools.custom_tool import USTechDataTool
    except ImportError:
        return None

    tool_out = USTechDataTool()._run(ticker)
    m_tool = re.search(r"SMA\s*\(?20\)?[^$]*\$\s*([\d,]+\.?\d*)", tool_out)
    m_text = re.search(r"SMA\s*\(?20\)?[^$]*\$\s*([\d,]+\.?\d*)", text)
    if not m_tool or not m_text:
        return None
    try:
        v_tool = float(m_tool.group(1).replace(",", ""))
        v_text = float(m_text.group(1).replace(",", ""))
    except ValueError:
        return None
    if v_tool == 0:
        return None
    diff = abs(v_text - v_tool) / v_tool
    if diff > tolerance:
        return f"SMA-20 mismatch: agent={v_text}, tool={v_tool}, diff={diff:.1%}"
    return None
