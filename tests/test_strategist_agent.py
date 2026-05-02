"""Layer 1+2 verification of investment_strategist (final_decision.json)."""
from __future__ import annotations
import pytest

from validators import decision as v
from conftest import TICKERS

pytestmark = pytest.mark.integration


@pytest.fixture
def decision_obj(ticker, pipeline_output):
    return pipeline_output(ticker)["decision"]


@pytest.mark.parametrize("ticker", TICKERS)
def test_pydantic_validates(ticker, decision_obj):
    err = v.check_pydantic_validates(decision_obj)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_all_fields_present(ticker, decision_obj):
    err = v.check_all_fields_present(decision_obj)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_decision_label(ticker, decision_obj):
    err = v.check_decision_label(decision_obj)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_prices_positive(ticker, decision_obj):
    err = v.check_prices_positive(decision_obj)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_ticker_matches(ticker, decision_obj):
    err = v.check_ticker_matches(decision_obj, ticker)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_expected_return_consistent(ticker, decision_obj):
    err = v.check_expected_return_consistency(decision_obj, tolerance=1.0)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_reasoning_lengths(ticker, decision_obj):
    err = v.check_reasoning_lengths(decision_obj, minimum=50)
    assert err is None, f"[{ticker}] {err}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_decision_aligns_with_return(ticker, decision_obj):
    err = v.check_decision_aligns_with_return(decision_obj)
    assert err is None, f"[{ticker}] {err}"
