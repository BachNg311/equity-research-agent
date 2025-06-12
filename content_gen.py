import os
from pathlib import Path
from PIL import Image
import google.generativeai as genai
import json

# ── Load API key ────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv; load_dotenv()
except ModuleNotFoundError:
    pass

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMININI_API_KEY")
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY in .env or env variables.")

genai.configure(api_key=api_key)

# ── Read Markdown and Decision JSON ─────────────────────────────────────────
macro_md = Path("us_market_analysis.md").read_text(encoding="utf-8")
fund_md  = Path("fundamental_analysis.md").read_text(encoding="utf-8")
tech_md  = Path("technical_analysis.md").read_text(encoding="utf-8")
final_decision = json.loads(Path("final_decision.json").read_text(encoding="utf-8"))

# ── Metadata from final_decision.json ───────────────────────────────────────
ticker = final_decision["stock_ticker"]
company = final_decision["full_name"]
date = final_decision["today_date"]
decision = final_decision["decision"]
target_price = final_decision["target_price"]
expected_return = final_decision["expected_return"]
current_price = final_decision["current_price"]

# ── Load Images (used only in Fundamental section) ──────────────────────────
image_paths = [
    f"{ticker}_price_line.jpg",
    f"{ticker}_revenue_bar.jpg",
    f"{ticker}_market_share_all.jpg",
]
images = [Image.open(p) for p in image_paths if Path(p).exists()]

# ── Final Recommendation Block (safe fallback if inserted directly) ─────────
final_decision_text = f"""
### Final Investment Recommendation

We assign a **{decision}** rating for **{company} ({ticker})** as of {date}.  
Our target price is **${target_price:.2f}**, implying an expected return of **{expected_return:.1f}%** from the current price of ${current_price:.2f}.
"""

# ── Enhanced System Prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = f"""
You are the **Final-Assembly Editor** for our CrewAI pipeline.

We have three analyst Markdown files:
1. `us_market_analysis.md` — macroeconomic and policy news
2. `fundamental_analysis.md` — financial ratios, industry comparisons, and qualitative assessments
3. `technical_analysis.md` — market trend data, price momentum, and signals

We also have three charts (to be included in the Fundamental section):
- `{ticker}_price_line.jpg` — 6-month stock price trend
- `{ticker}_revenue_bar.jpg` — annual revenue trend (last 4 years)
- `{ticker}_market_share_all.jpg` — market share vs. competitors

Your task: synthesize all input into a structured Markdown report for **{company} ({ticker})** dated **{date}**.

─────────────────────────── FORMAT ─────────────────────────────

# U.S. Equity Research Report
## {ticker} — {date}

### Executive Summary
~150 words summary combining macro context, valuation, and technical outlook.

### Macroeconomic & Policy Outlook
A ~500–600 word professional summary based solely on `us_market_analysis.md`.
Must reflect tone of institutional research.

### Technical Analysis
Present all key indicators from `technical_analysis.md` in a clean table.
Use professional style with 300–400 word commentary on momentum, trend, and support/resistance levels.
Do **not** include any images here.

### Fundamental & Valuation Analysis
Insert full content from `fundamental_analysis.md`.
Then embed all three charts, each followed by a 200–300 word paragraph describing:
- Trends
- Fluctuations
- Implications for valuation
- Competitive context
Use vivid, analytical language to interpret the visual data.

Also include:
- DCF valuation model with assumptions (WACC, growth)
- P/E and EV/EBITDA comps
- SWOT or Porter’s Five Forces summary
- 3-year forecasts for revenue, net income, EPS, and FCF
- Final rating: **Buy / Hold / Sell**, target price, expected return %
- Conclude with the CrewAI final recommendation: **Buy / Hold / Sell**, target price, and expected return % from final_decision.json

### Risks & Catalysts
- *Three downside risks* (macro, execution, regulatory)
- *Two upside catalysts* (include timing & mechanisms)
Each bullet ≤50 words. Start with `-`, **bold** triggers, *italicize* affected metrics, and end with *(High / Medium / Low)* probability tag.

─────────────────────────── STYLE ───────────────────────────────
- Use **bold** sparingly — headings, ratings, key financial ratios
- Tables must be professional: headers bold, numbers right-aligned
- Image captions should be exact filenames
- Do not include any HTML, PDF, or JSON tags
- Output must be pure **Markdown**
"""

# ── Compose prompt for Gemini (text + images) ───────────────────────────────
prompt_parts = [
    SYSTEM_PROMPT,
    *images,
    f"""
### MACRO
{macro_md}

### FUNDAMENTAL
{fund_md}

{final_decision_text}

### TECHNICAL
{tech_md}
"""
]

# ── Generate Markdown Report ───────────────────────────────────────────────
model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
response = model.generate_content(prompt_parts, stream=False)
final_md = response.text
if final_md.strip().startswith("```markdown"):
    final_md = final_md.strip()
    final_md = final_md.removeprefix("```markdown").removesuffix("```").strip()


# ── Save to report.md ───────────────────────────────────────────────────────
Path("report.md").write_text(final_md, encoding="utf-8")
print("✅ Assembled Markdown written to report.md")
