"""Layer 4: LLM-as-judge for qualitative grading.

Uses MODEL_REASONING (typically a stronger model) to grade outputs from
agents that ran on MODEL (typically a faster model). Different models
reduce self-bias.

Tests using this should be marked @pytest.mark.judge so they can be
opted in: pytest -m judge.
"""
from __future__ import annotations
import json
import os
import re
from typing import Any

JUDGE_PROMPT = """You are grading the output of a {agent_role} agent for ticker {ticker}.

Output to grade:
---
{output}
---

Rate 1-5 on each criterion (5 = excellent, 1 = poor):
- relevance:    does it address what was asked?
- completeness: are all required sections present?
- coherence:    is the reasoning internally consistent?
- groundedness: are claims tied to data, not made up?

Return ONLY valid JSON in this exact form, no prose:
{{"relevance": N, "completeness": N, "coherence": N, "groundedness": N, "notes": "<one short line>"}}
"""

CRITERIA = ["relevance", "completeness", "coherence", "groundedness"]


def judge(agent_role: str, ticker: str, output: str) -> dict[str, Any]:
    """Grade `output` and return scores dict, or empty dict on failure."""
    from crewai import LLM

    model = os.environ.get("MODEL_REASONING") or os.environ.get("MODEL")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not model or not api_key:
        return {}

    llm = LLM(model=model, api_key=api_key, temperature=0, max_tokens=512)
    truncated = output[:8000]
    prompt = JUDGE_PROMPT.format(agent_role=agent_role, ticker=ticker, output=truncated)

    try:
        raw = llm.call([{"role": "user", "content": prompt}])
    except Exception as e:
        return {"_error": f"LLM call failed: {e}"}

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"_error": f"no JSON in judge response: {raw[:200]}"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return {"_error": f"judge JSON parse failed: {e}"}


def assert_scores_above(scores: dict, threshold: int = 3) -> str | None:
    """Returns failure message if any criterion scored below threshold."""
    if "_error" in scores:
        return scores["_error"]
    bad = [(k, scores.get(k)) for k in CRITERIA if (scores.get(k) or 0) < threshold]
    if bad:
        return f"low scores (need >= {threshold}): {bad}; notes={scores.get('notes')}"
    return None
