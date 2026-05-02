# Automated Agent Output Verification Plan

## Goal

Verify that each of the 4 agents in `agents.yaml` produces correct, complete, and high-quality output **automatically** (no manual eyeballing) and **at scale** (across many tickers).

The 4 agents under test:

1. `stock_news_researcher` → `us_market_analysis.md`
2. `fundamental_analyst` → `fundamental_analysis.md`
3. `technical_analyst` → `technical_analysis.md`
4. `investment_strategist` → `final_decision.json`

---

## Strategy: 4 Layers of Verification

Because LLM outputs are non-deterministic, we cannot assert exact strings. Instead, we verify **shape**, **contents**, and **quality** programmatically.

```
┌─────────────────────────────────────────────────────────┐
│ Layer 4: LLM-as-judge (qualitative scoring)             │
│ Layer 3: Cross-agent consistency (integration)          │
│ Layer 2: Structural / regex assertions (must contain X) │
│ Layer 1: Schema validation (Pydantic, types)            │
└─────────────────────────────────────────────────────────┘
                       ↑
        Run all layers across N tickers in parallel
```

---

## Per-Agent Automated Assertions

### Agent 1 — `stock_news_researcher`

Output: `us_market_analysis.md` (markdown report with 5 macro news items)

| Check | Method |
|---|---|
| Contains ≥ 5 URLs | `re.findall(r'https?://[^\s)]+', text)` |
| Contains ≥ 5 distinct news items | Count `^#`/`^-` markers OR LLM judge counts |
| Mentions macro keywords | At least 2 of: `Fed`, `S&P 500`, `Nasdaq`, `inflation`, `rate`, `Treasury` |
| Word count ≥ 800 | 3-page proxy |
| URLs reachable | `httpx.head(url, timeout=5).status_code < 400` (sample 3) |
| Recency | At least one cited URL has a date within last 90 days |

### Agent 2 — `fundamental_analyst`

Output: `fundamental_analysis.md` (markdown with ratios table, peer comparison, valuation verdict)

| Check | Method |
|---|---|
| All 8 ratio labels present | Substring check: `P/E`, `P/B`, `ROE`, `ROA`, `EPS`, `D/E`, `EV/EBITDA`, `Profit Margin` |
| Each ratio has a numeric value | Regex `P/E[^\d]*(\d+\.?\d*)` then `float()` it |
| Valuation verdict present | Regex `undervalued|overvalued|fairly\s+valued` |
| Sector matches yfinance | Compare extracted sector vs `yf.Ticker(t).info["sector"]` |
| Numbers are sane | P/E in [-100, 1000], ROE in [-2, 5], etc. |
| Word count ≥ 800 | |

### Agent 3 — `technical_analyst`

Output: `technical_analysis.md` (markdown with indicators, trend, support/resistance)

| Check | Method |
|---|---|
| All indicators present | `SMA.*20`, `SMA.*50`, `SMA.*200`, `RSI`, `MACD`, `Bollinger` |
| Trend label present | Regex `BULLISH\|BEARISH\|NEUTRAL\|uptrend\|downtrend\|sideways` |
| Support/resistance levels | Regex `\$\d+\.?\d*` matches ≥ 2 times |
| Indicator values match the tool's output | Run `USTechDataTool._run(t)` separately, parse SMA-20 from both, assert within 5% |
| Ticker mentioned | `t.upper() in text` |

### Agent 4 — `investment_strategist`

Output: `final_decision.json` (structured JSON matching `InvestmentDecision`)

This is the **strongest agent to test** because it's structured.

| Check | Method |
|---|---|
| Pydantic validates | `InvestmentDecision.model_validate_json(raw)` — already enforced by `output_json=` |
| `decision ∈ {BUY, HOLD, SELL}` | `Literal` already enforces |
| `current_price > 0` | numeric assertion |
| `target_price > 0` | numeric assertion |
| `stock_ticker == input_ticker` | exact match |
| `expected_return ≈ (target/current - 1) * 100` | within 1% tolerance |
| All reasoning fields ≥ 50 chars | `len(d.fund_reasoning) >= 50` |
| Decision-return consistency | `expected_return > 10%` → expect BUY; `< -10%` → expect SELL; else HOLD acceptable |
| Cross-reference: `fund_reasoning` mentions a ratio from `fundamental_analysis.md` | substring/regex match |
| Cross-reference: `tech_reasoning` mentions an indicator from `technical_analysis.md` | substring/regex match |

---

## Layer 4: LLM-as-Judge

For qualitative checks that regex can't catch (coherence, relevance, hallucination), use a separate LLM call with a rubric:

```python
JUDGE_PROMPT = """
You are grading a {agent_role} output for ticker {ticker}.

Output to grade:
---
{output}
---

Rate 1-5 on each criterion:
- relevance: does it answer what was asked?
- completeness: are all required sections present?
- coherence: is the reasoning internally consistent?
- groundedness: are claims tied to data, not made up?

Return JSON only: {{"relevance": N, "completeness": N, "coherence": N, "groundedness": N, "notes": "..."}}
"""

def judge(agent_role, ticker, output) -> dict:
    response = gemini.generate(JUDGE_PROMPT.format(...))
    return json.loads(response)

# In test:
scores = judge("fundamental_analyst", "AAPL", output)
assert all(v >= 3 for k, v in scores.items() if k != "notes")
```

**Use a different model as the judge** than the one that generated the output (e.g., generate with `gemini-1.5-flash`, judge with `gemini-1.5-pro`) to reduce self-bias.

---

## Layer 3: Scaling — Multi-Ticker Parametrization

Run every check across a basket of tickers using `pytest.parametrize`:

```python
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META", "JPM"]

@pytest.mark.parametrize("ticker", TICKERS)
def test_fundamental_output_has_all_ratios(ticker, fundamental_output):
    text = fundamental_output(ticker)
    for label in ["P/E", "P/B", "ROE", "EPS", "D/E"]:
        assert label in text, f"{ticker}: missing {label}"
```

**Run in parallel:**
```bash
pytest tests/ -n 8 --dist=loadfile
```
(requires `pytest-xdist`)

**Pass-rate gate** instead of strict pass/fail (LLMs occasionally fail):
```python
def test_fundamental_passes_at_scale(results_per_ticker):
    pass_rate = sum(r.passed for r in results_per_ticker) / len(results_per_ticker)
    assert pass_rate >= 0.80, f"Only {pass_rate:.0%} passed"
```

---

## Caching: Don't Re-run the LLM Every Test

LLM calls are slow ($0.10–$1 per ticker, 30–120s). Cache outputs to disk:

```python
@pytest.fixture(scope="session")
def fundamental_output():
    cache_dir = Path(".test_cache/fundamental")
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(ticker: str) -> str:
        cache_file = cache_dir / f"{ticker}.md"
        if cache_file.exists() and not os.getenv("FORCE_REGEN"):
            return cache_file.read_text()
        result = run_single_agent("fundamental_analyst", ticker)
        cache_file.write_text(result)
        return result
    return _get
```

| Command | Behavior |
|---|---|
| `pytest` | Uses cached outputs, just runs assertions (fast, cheap) |
| `FORCE_REGEN=1 pytest` | Regenerates outputs (slow, expensive — nightly CI) |

---

## Suggested File Layout

```
tests/
├── conftest.py                  # env vars + cache fixtures
├── runners.py                   # run_single_agent(name, ticker) helper
├── judges.py                    # LLM-as-judge wrapper
├── validators/
│   ├── news.py                  # validate_news(text) -> list[Failure]
│   ├── fundamental.py
│   ├── technical.py
│   └── decision.py              # Pydantic-aware
├── test_news_agent.py           # parametrized over TICKERS
├── test_fundamental_agent.py
├── test_technical_agent.py
├── test_strategist_agent.py
├── test_cross_agent.py          # consistency between agents
└── .test_cache/                 # gitignored, holds LLM outputs
```

---

## CI Strategy

LLM tests are slow and cost money — different cadences for different layers:

| Trigger | Tests run | Cost | Time |
|---|---|---|---|
| Every PR | Layers 1+2 on **cached** outputs | ~$0 | ~10s |
| Every PR | Tool unit tests (mocked yfinance) | ~$0 | ~5s |
| Nightly | All layers + regenerate cache, 8 tickers | ~$5 | ~10 min |
| Weekly | LLM-as-judge across 25 tickers | ~$15 | ~30 min |

---

## Quickest Path to Start

1. **Layer 1 first** — `InvestmentDecision` Pydantic validation already gives you a free win for the strategist
2. **Layer 2 next** — write `validators/*.py`, parametrize over 3 tickers, no judge yet
3. **Caching** — so iteration on validators doesn't burn tokens
4. **Layer 4 (judge)** — only after validators stabilize
5. **Layer 3 (scale)** — bump from 3 tickers to 25 once green

---

## Required Dev Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.5",     # parallel execution
    "pytest-mock>=3.14",
    "httpx>=0.27",           # URL reachability checks
]
```

Install: `pip install -e ".[dev]"`
