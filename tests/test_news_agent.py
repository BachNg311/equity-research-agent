"""Layer 1+2 verification of stock_news_researcher output.

Run with:  pytest tests/test_news_agent.py -v
Skip slow/network checks: pytest tests/test_news_agent.py -m "not network"
"""
from __future__ import annotations
import pytest

from validators import news as v
from conftest import TICKERS

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def news_text(pipeline_output):
    """News output is macro-level (ticker-independent); sample one ticker only."""
    return pipeline_output(TICKERS[0])["news"]


def test_has_min_urls(news_text):
    err = v.check_has_min_urls(news_text, minimum=5)
    assert err is None, err


def test_has_min_news_items(news_text):
    err = v.check_has_min_news_items(news_text, minimum=5)
    assert err is None, err


def test_macro_keywords_present(news_text):
    err = v.check_macro_keywords(news_text, minimum=2)
    assert err is None, err


def test_word_count(news_text):
    err = v.check_word_count(news_text, minimum=800)
    assert err is None, err


@pytest.mark.network
def test_urls_reachable(news_text):
    err = v.check_urls_reachable(news_text, sample_size=3)
    assert err is None, err
