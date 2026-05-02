"""Layer 3: cross-agent consistency.

The strategist's reasoning fields should reference values that actually appear
in the upstream fundamental and technical analyses.
"""
from __future__ import annotations
import pytest

from validators import decision as v
from conftest import TICKERS

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("ticker", TICKERS)
def test_fund_reasoning_grounded_in_fundamental(ticker, pipeline_output):
    out = pipeline_output(ticker)
    err = v.check_reasoning_grounded_in(
        reasoning=out["decision"].get("fund_reasoning", ""),
        source_text=out["fundamental"],
        min_overlap=1,
    )
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_tech_reasoning_grounded_in_technical(ticker, pipeline_output):
    out = pipeline_output(ticker)
    err = v.check_reasoning_grounded_in(
        reasoning=out["decision"].get("tech_reasoning", ""),
        source_text=out["technical"],
        min_overlap=1,
    )
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_macro_reasoning_grounded_in_news(ticker, pipeline_output):
    out = pipeline_output(ticker)
    err = v.check_reasoning_grounded_in(
        reasoning=out["decision"].get("macro_reasoning", ""),
        source_text=out["news"],
        min_overlap=1,
    )
    assert err is None, f"[{ticker}] {err}"
