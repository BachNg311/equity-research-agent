"""Run the 4-agent pipeline and cache outputs to disk.

Disk cache: tests/.test_cache/{ticker}_{date}.json
Override with FORCE_REGEN=1 to invalidate.
"""
from __future__ import annotations
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).parent / ".test_cache"


@contextmanager
def _cwd(path: str | Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _extract_decision(task_output: Any) -> dict:
    """Get the strategist's structured JSON, robust to CrewAI version drift."""
    if getattr(task_output, "json_dict", None):
        return task_output.json_dict
    if getattr(task_output, "pydantic", None):
        return task_output.pydantic.model_dump()
    raw = getattr(task_output, "raw", "") or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_unparseable_raw": raw}


def run_pipeline(symbol: str, current_date: str) -> dict:
    """Run news -> fundamental -> technical -> decision and return outputs.

    Returns:
        {
            "news":        str (markdown),
            "fundamental": str (markdown),
            "technical":   str (markdown),
            "decision":    dict (parsed InvestmentDecision JSON),
        }
    """
    cache_file = CACHE_DIR / f"{symbol}_{current_date}.json"
    if cache_file.exists() and not os.getenv("FORCE_REGEN"):
        return json.loads(cache_file.read_text(encoding="utf-8"))

    from crewai import Crew, Process
    from stock_advisor.crew import USStockAdvisor

    obj = USStockAdvisor()
    crew = Crew(
        agents=[
            obj.stock_news_researcher(),
            obj.fundamental_analyst(),
            obj.technical_analyst(),
            obj.investment_strategist(),
        ],
        tasks=[
            obj.news_collecting(),
            obj.fundamental_analysis(),
            obj.technical_analysis(),
            obj.investment_decision(),
        ],
        process=Process.sequential,
        verbose=False,
    )

    with tempfile.TemporaryDirectory() as tmp, _cwd(tmp):
        result = crew.kickoff(
            inputs={"symbol": symbol, "current_date": current_date}
        )

    outputs = {
        "news":        result.tasks_output[0].raw or "",
        "fundamental": result.tasks_output[1].raw or "",
        "technical":   result.tasks_output[2].raw or "",
        "decision":    _extract_decision(result.tasks_output[3]),
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    return outputs
