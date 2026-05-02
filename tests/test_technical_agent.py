"""Layer 1+2 verification of technical_analyst output, parametrized over tickers."""
from __future__ import annotations
import pytest

from validators import technical as v
from conftest import TICKERS

pytestmark = pytest.mark.integration


@pytest.fixture
def tech_text(ticker, pipeline_output):
    return pipeline_output(ticker)["technical"]


@pytest.mark.parametrize("ticker", TICKERS)
def test_has_all_indicators(ticker, tech_text):
    err = v.check_has_all_indicators(tech_text)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_has_trend_label(ticker, tech_text):
    err = v.check_has_trend_label(tech_text)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_has_support_resistance_levels(ticker, tech_text):
    err = v.check_has_support_resistance_levels(tech_text, minimum=2)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_ticker_mentioned(ticker, tech_text):
    err = v.check_ticker_mentioned(tech_text, ticker)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_word_count(ticker, tech_text):
    err = v.check_word_count(tech_text, minimum=600)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_indicator_values_match_tool(ticker, tech_text):
    err = v.check_indicator_values_match_tool(tech_text, ticker, tolerance=0.10)
    assert err is None, f"[{ticker}] {err}"
