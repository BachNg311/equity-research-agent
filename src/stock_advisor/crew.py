from __future__ import annotations
"""CrewAI project file – U.S. Stock Advisor.

This rewrites the original `VnStockAdvisor` crew for the U.S. market.
The key changes are:
• Swaps `vnstock`-based tools for `USFundDataTool` and `USTechDataTool` (see `custom_tool_us.py`).
• Updates Serper locale/location to U.S.
• Keeps Gemini models, but you can plug Groq/OpenAI/etc. the same way.
• Renames class and output files, but preserves the same crew/task structure.
"""

import os
import warnings
from pathlib import Path
from typing import List, Literal

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    FirecrawlScrapeWebsiteTool,
    SerperDevTool,
    WebsiteSearchTool,
)
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ──────────────────────────────── Local imports ────────────────────────────── #
from .tools.custom_tool import (
    USFundDataTool,
    USTechDataTool,
    FileReadTool,
)

###############################################################################
# ─────────────────────────── env & model bootstrap ────────────────────────── #
###############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("MODEL")
GEMINI_REASONING_MODEL = os.environ.get("MODEL_REASONING")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")

# Deterministic generation (temperature = 0)
llm_general = LLM(model=GEMINI_MODEL, api_key=GEMINI_API_KEY, temperature=0, max_tokens=4096)
llm_reasoning = LLM(
    model=GEMINI_REASONING_MODEL, api_key=GEMINI_API_KEY, temperature=0, max_tokens=4096
)

###############################################################################
# ──────────────────────────────── Tools setup ─────────────────────────────── #
###############################################################################

file_read_tool = FileReadTool(file_path="knowledge/PE_PB_industry_average_eng.json", description="A tool to read the PE/PB industry average data from a JSON file.")
fund_tool = USFundDataTool()
tech_tool = USTechDataTool(result_as_answer=True)

scrape_tool = FirecrawlScrapeWebsiteTool(timeout=60, api_key=FIRECRAWL_API_KEY)
search_tool = SerperDevTool(
    country="us",  # U.S. market news
    locale="us",
    location="New York, New York, United States"
)

web_search_tool = WebsiteSearchTool(
    config=dict(
        llm={
            "provider": "google",
            "config": {"model": GEMINI_MODEL, "api_key": GEMINI_API_KEY},
        },
        embedder={
            "provider": "google",
            "config": {"model": "models/text-embedding-004", "task_type": "retrieval_document"},
        },
    )
)

###############################################################################
# ───────────────────────── Knowledge sources & schema ─────────────────────── #
###############################################################################

json_source = JSONKnowledgeSource(file_paths=["PE_PB_industry_average_eng.json"])


class InvestmentDecision(BaseModel):
    """Structured JSON output for the final investment decision."""

    stock_ticker: str = Field(..., description="Ticker symbol, e.g. AAPL")
    full_name: str = Field(..., description="Company full name")
    industry: str = Field(..., description="Industry / sector")
    today_date: str = Field(..., description="Analysis date (YYYY‑MM‑DD)")
    decision: Literal["BUY", "HOLD", "SELL"] = Field(..., description="Investment decision")
    macro_reasoning: str = Field(..., description="Macro & news‑driven justification")
    fund_reasoning: str = Field(..., description="Fundamental analysis rationale")
    tech_reasoning: str = Field(..., description="Technical analysis rationale")

###############################################################################
# ─────────────────────────────── Crew definition ──────────────────────────── #
###############################################################################


@CrewBase
class USStockAdvisor:
    """CrewAI crew for analysing U.S. equities."""

    agents: List[BaseAgent]
    tasks: List[Task]

    # ─────────────────────────────── Agents ──────────────────────────────── #

    @agent
    def stock_news_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["stock_news_researcher"],
            verbose=True,
            llm=llm_general,
            tools=[search_tool, scrape_tool, web_search_tool],
            max_rpm=10,
        )

    @agent
    def fundamental_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["fundamental_analyst"],
            verbose=True,
            llm=llm_general,
            tools=[fund_tool, file_read_tool],
            knowledge_sources=[json_source],
            max_rpm=10,
            embedder={
                "provider": "google",
                "config": {"model": "models/text-embedding-004", "api_key": GEMINI_API_KEY},
            },
        )

    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["technical_analyst"],
            verbose=True,
            llm=llm_general,
            tools=[tech_tool],
            max_rpm=10,
        )

    @agent
    def investment_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config["investment_strategist"],
            verbose=True,
            llm=llm_reasoning,
            max_rpm=10,
        )

    # ─────────────────────────────── Tasks ───────────────────────────────── #

    @task
    def news_collecting(self) -> Task:
        return Task(
            config=self.tasks_config["news_collecting"],
            async_execution=True,
            output_file="us_market_analysis.md",
        )

    @task
    def fundamental_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["fundamental_analysis"],
            async_execution=True,
            output_file="fundamental_analysis.md",
        )

    @task
    def technical_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["technical_analysis"],
            async_execution=True,
            output_file="technical_analysis.md",
        )

    @task
    def investment_decision(self) -> Task:
        return Task(
            config=self.tasks_config["investment_decision"],
            context=[self.news_collecting(), self.fundamental_analysis(), self.technical_analysis()],
            output_json=InvestmentDecision,
            output_file="final_decision.json",
        )

    # ─────────────────────────────── Crew ────────────────────────────────── #

    @crew
    def crew(self) -> Crew:  # noqa: D401
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )