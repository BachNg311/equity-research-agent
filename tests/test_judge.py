"""Layer 4: LLM-as-judge gradings, opt-in.

Run with: pytest -m judge
"""
from __future__ import annotations
import pytest

from judges import judge, assert_scores_above
from conftest import TICKERS

pytestmark = [pytest.mark.integration, pytest.mark.judge]


@pytest.mark.parametrize("ticker", TICKERS)
def test_fundamental_judged(ticker, pipeline_output):
    text = pipeline_output(ticker)["fundamental"]
    scores = judge("fundamental_analyst", ticker, text)
    err = assert_scores_above(scores, threshold=3)
    assert err is None, f"[{ticker}] {err} | scores={scores}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_technical_judged(ticker, pipeline_output):
    text = pipeline_output(ticker)["technical"]
    scores = judge("technical_analyst", ticker, text)
    err = assert_scores_above(scores, threshold=3)
    assert err is None, f"[{ticker}] {err} | scores={scores}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_strategist_judged(ticker, pipeline_output):
    import json as _json
    decision = pipeline_output(ticker)["decision"]
    scores = judge("investment_strategist", ticker, _json.dumps(decision, indent=2))
    err = assert_scores_above(scores, threshold=3)
    assert err is None, f"[{ticker}] {err} | scores={scores}"


def test_news_judged(pipeline_output):
    text = pipeline_output(TICKERS[0])["news"]
    scores = judge("stock_news_researcher", TICKERS[0], text)
    err = assert_scores_above(scores, threshold=3)
    assert err is None, f"{err} | scores={scores}"
