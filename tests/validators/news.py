"""Validators for stock_news_researcher output (us_market_analysis.md).

Each `check_*` returns a failure message string, or None on success.
"""
from __future__ import annotations
import re

URL_RE = re.compile(r"https?://[^\s)\]]+")
MACRO_KEYWORDS = ["fed", "s&p 500", "nasdaq", "inflation", "rate", "treasury",
                  "gdp", "fomc", "jobs", "cpi"]


def check_has_min_urls(text: str, minimum: int = 5) -> str | None:
    urls = URL_RE.findall(text)
    if len(urls) < minimum:
        return f"only {len(urls)} URLs found, need >= {minimum}"
    return None


def check_has_min_news_items(text: str, minimum: int = 5) -> str | None:
    headers = re.findall(r"(?m)^#{1,3}\s+\S", text)
    bullets = re.findall(r"(?m)^[-*]\s+\S", text)
    items = max(len(headers), len(bullets))
    if items < minimum:
        return f"only {items} news items detected, need >= {minimum}"
    return None


def check_macro_keywords(text: str, minimum: int = 2) -> str | None:
    lower = text.lower()
    hits = [k for k in MACRO_KEYWORDS if k in lower]
    if len(hits) < minimum:
        return f"only {len(hits)} macro keywords found ({hits}), need >= {minimum}"
    return None


def check_word_count(text: str, minimum: int = 800) -> str | None:
    n = len(text.split())
    if n < minimum:
        return f"word count {n} < {minimum}"
    return None


def check_urls_reachable(text: str, sample_size: int = 3, timeout: float = 5.0) -> str | None:
    """Sample up to N URLs and check they return < 400."""
    try:
        import httpx
    except ImportError:
        return None
    urls = URL_RE.findall(text)[:sample_size]
    if not urls:
        return "no URLs to check"
    failed = []
    for url in urls:
        try:
            r = httpx.head(url, timeout=timeout, follow_redirects=True)
            if r.status_code >= 400:
                failed.append(f"{url} -> {r.status_code}")
        except Exception as e:
            failed.append(f"{url} -> {type(e).__name__}")
    if failed:
        return f"unreachable URLs: {failed}"
    return None
