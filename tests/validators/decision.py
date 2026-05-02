"""Validators for investment_strategist output (final_decision.json)."""
from __future__ import annotations
import re

REQUIRED_FIELDS = [
    "stock_ticker", "full_name", "industry", "today_date",
    "current_price", "target_price", "expected_return",
    "decision", "macro_reasoning", "fund_reasoning", "tech_reasoning",
]

VALID_DECISIONS = {"BUY", "HOLD", "SELL"}
NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def check_pydantic_validates(decision: dict) -> str | None:
    """The InvestmentDecision Pydantic model must accept the dict."""
    try:
        from stock_advisor.crew import InvestmentDecision
        InvestmentDecision.model_validate(decision)
    except Exception as e:
        return f"Pydantic validation failed: {e}"
    return None


def check_all_fields_present(decision: dict) -> str | None:
    missing = [f for f in REQUIRED_FIELDS if f not in decision]
    if missing:
        return f"missing fields: {missing}"
    return None


def check_decision_label(decision: dict) -> str | None:
    label = decision.get("decision")
    if label not in VALID_DECISIONS:
        return f"decision={label!r} not in {VALID_DECISIONS}"
    return None


def check_prices_positive(decision: dict) -> str | None:
    bad = []
    for field in ["current_price", "target_price"]:
        v = decision.get(field)
        if not isinstance(v, (int, float)) or v <= 0:
            bad.append(f"{field}={v}")
    if bad:
        return f"invalid prices: {bad}"
    return None


def check_ticker_matches(decision: dict, expected: str) -> str | None:
    actual = (decision.get("stock_ticker") or "").upper()
    if actual != expected.upper():
        return f"stock_ticker={actual!r}, expected {expected.upper()!r}"
    return None


def check_expected_return_consistency(decision: dict, tolerance: float = 1.0) -> str | None:
    """expected_return should ≈ (target/current - 1) * 100, within `tolerance` pct points."""
    cur = decision.get("current_price")
    tgt = decision.get("target_price")
    er = decision.get("expected_return")
    if not all(isinstance(x, (int, float)) for x in (cur, tgt, er)) or cur == 0:
        return None
    computed = (tgt / cur - 1) * 100
    if abs(er - computed) > tolerance:
        return f"expected_return={er} != computed {computed:.2f} (tolerance {tolerance})"
    return None


def check_reasoning_lengths(decision: dict, minimum: int = 50) -> str | None:
    short = []
    for field in ["macro_reasoning", "fund_reasoning", "tech_reasoning"]:
        text = decision.get(field, "") or ""
        if len(text) < minimum:
            short.append(f"{field} ({len(text)} chars)")
    if short:
        return f"reasoning too short (need >= {minimum} chars): {short}"
    return None


def check_decision_aligns_with_return(decision: dict) -> str | None:
    """Sanity: BUY -> positive return; SELL -> negative return."""
    label = decision.get("decision")
    er = decision.get("expected_return")
    if not isinstance(er, (int, float)):
        return None
    if label == "BUY" and er < -5:
        return f"BUY but expected_return={er} (strongly negative)"
    if label == "SELL" and er > 5:
        return f"SELL but expected_return={er} (strongly positive)"
    return None


def check_reasoning_grounded_in(reasoning: str, source_text: str, min_overlap: int = 1) -> str | None:
    """`reasoning` should reference at least N numeric values from `source_text`."""
    src_numbers = set(NUMBER_RE.findall(source_text))
    src_numbers = {n for n in src_numbers if len(n) >= 2}  # filter trivial single digits
    rsn_numbers = set(NUMBER_RE.findall(reasoning))
    overlap = src_numbers & rsn_numbers
    if len(overlap) < min_overlap:
        return f"reasoning shares only {len(overlap)} numbers with source (need >= {min_overlap})"
    return None
