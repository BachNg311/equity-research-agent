"""Layer 1+2 verification of fundamental_analyst output, parametrized over tickers."""
from __future__ import annotations
import pytest

from validators import fundamental as v
from conftest import TICKERS

pytestmark = pytest.mark.integration


@pytest.fixture
def fund_text(ticker, pipeline_output):
    return pipeline_output(ticker)["fundamental"]


@pytest.mark.parametrize("ticker", TICKERS)
def test_has_all_ratio_labels(ticker, fund_text):
    err = v.check_has_all_ratio_labels(fund_text)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_has_valuation_verdict(ticker, fund_text):
    err = v.check_has_valuation_verdict(fund_text)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_word_count(ticker, fund_text):
    err = v.check_word_count(fund_text, minimum=800)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_ratio_values_sane(ticker, fund_text):
    err = v.check_ratio_values_are_sane(fund_text)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_sector_matches_yfinance(ticker, fund_text):
    err = v.check_sector_matches_yfinance(fund_text, ticker)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_ticker_mentioned(ticker, fund_text):
    err = v.check_ticker_mentioned(fund_text, ticker)
    assert err is None, f"[{ticker}] {err}"
