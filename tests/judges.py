"""Layer 4: LLM-as-judge for qualitative grading.

Judge model: Claude Sonnet 4.6 (different provider from Gemini generators).
Using a different provider eliminates cross-model self-bias — same-provider
judges systematically inflate scores for their own family's outputs.

Requires ANTHROPIC_API_KEY in .env.
Tests using this should be marked @pytest.mark.judge: pytest -m judge.
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

# Claude Sonnet 4.6 — strong reasoning, cost-effective, different provider from Gemini generators.
# Switch to claude-opus-4-7 for highest judgment quality (higher cost).
JUDGE_MODEL = "claude-sonnet-4-6"


def judge(agent_role: str, ticker: str, output: str) -> dict[str, Any]:
    """Grade `output` with Claude and return scores dict, or error dict on failure."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"_error": "ANTHROPIC_API_KEY not set"}

    try:
        import anthropic
    except ImportError:
        return {"_error": "anthropic package not installed: pip install anthropic"}

    client = anthropic.Anthropic(api_key=api_key)
    truncated = output[:8000]
    prompt = JUDGE_PROMPT.format(agent_role=agent_role, ticker=ticker, output=truncated)

    try:
        message = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
    except Exception as e:
        return {"_error": f"Claude API call failed: {e}"}

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
