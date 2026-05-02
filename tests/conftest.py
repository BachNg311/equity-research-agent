"""Pytest configuration: env-var setup and session fixtures.

Real Gemini/Serper/Firecrawl/Finnhub keys must be in `.env` for integration tests.
The setdefault calls below only protect against module-level import failures
(crew.py instantiates LLM and tools at import time), not enable real LLM calls.
"""
from __future__ import annotations
import os
import sys
from datetime import date
from pathlib import Path

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")
os.environ.setdefault("MODEL", "gemini/gemini-1.5-flash")
os.environ.setdefault("MODEL_REASONING", "gemini/gemini-1.5-pro")
os.environ.setdefault("SERPER_API_KEY", "test-dummy-key")
os.environ.setdefault("FIRECRAWL_API_KEY", "test-dummy-key")
os.environ.setdefault("FINNHUB_API_KEY", "test-dummy-key")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))


TICKERS = ["AAPL", "MSFT", "GOOGL"]
TEST_DATE = str(date.today())


@pytest.fixture(scope="session")
def tickers() -> list[str]:
    return TICKERS


@pytest.fixture(scope="session")
def test_date() -> str:
    return TEST_DATE


@pytest.fixture(scope="session")
def pipeline_output():
    """Return a getter `fn(ticker) -> dict` with all 4 agent outputs.

    Cached in-memory per session and on-disk at tests/.test_cache/.
    Set FORCE_REGEN=1 to bypass disk cache and re-run the LLM pipeline.
    """
    from runners import run_pipeline

    mem_cache: dict[str, dict] = {}

    def _get(ticker: str) -> dict:
        if ticker not in mem_cache:
            mem_cache[ticker] = run_pipeline(ticker, TEST_DATE)
        return mem_cache[ticker]

    return _get
