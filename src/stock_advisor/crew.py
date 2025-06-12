from __future__ import annotations
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
from .tools.custom_tool import (
    USFundDataTool,
    USTechDataTool,
    USSectorValuationTool,
    FileReadTool,
    StockPriceLineChartTool, RevenueBarChartTool, MarketShareAllPeersDonutTool,
    MarkdownRenderTool, WeasyPrintTool, PdfMergeTool, ImageToPdfTool
)

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("MODEL")
GEMINI_REASONING_MODEL = os.environ.get("MODEL_REASONING")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")


llm_general = LLM(model=GEMINI_MODEL, api_key=GEMINI_API_KEY, temperature=0, max_tokens=4096)
llm_reasoning = LLM(
    model=GEMINI_REASONING_MODEL, api_key=GEMINI_API_KEY, temperature=0, max_tokens=4096
)



# file_read_tool = FileReadTool(file_path="knowledge/PE_PB_industry_average_eng.json", description="A tool to read the PE/PB industry average data from a JSON file.")
fund_tool = USFundDataTool()
tech_tool = USTechDataTool(result_as_answer=True)
sector_valuation_tool = USSectorValuationTool()
price_chart_tool   = StockPriceLineChartTool()
revenue_chart_tool = RevenueBarChartTool()
market_share_tool = MarketShareAllPeersDonutTool()
# markdown_tool   = MarkdownRenderTool()
# weasy_tool      = WeasyPrintTool()
# pdf_merge_tool  = PdfMergeTool()
# image2pdf_tool = ImageToPdfTool()
scrape_tool = FirecrawlScrapeWebsiteTool(timeout=60, api_key=FIRECRAWL_API_KEY)
search_tool = SerperDevTool(
    country="us",  
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


# json_source = JSONKnowledgeSource(file_paths=["PE_PB_industry_average_eng.json"])


class InvestmentDecision(BaseModel):
    """Structured JSON output for the final investment decision."""

    stock_ticker: str = Field(..., description="Ticker symbol, e.g. AAPL")
    full_name: str = Field(..., description="Company full name")
    industry: str = Field(..., description="Industry / sector")
    today_date: str = Field(..., description="Analysis date (YYYY‑MM‑DD)")
    current_price: float = Field(..., description="Current stock price in USD")
    target_price: float = Field(..., description="Target price in USD")
    expected_return: float = Field(..., description="Expected return in %")
    decision: Literal["BUY", "HOLD", "SELL"] = Field(..., description="Investment decision")
    macro_reasoning: str = Field(..., description="Macro & news‑driven justification")
    fund_reasoning: str = Field(..., description="Fundamental analysis rationale")
    tech_reasoning: str = Field(..., description="Technical analysis rationale")




@CrewBase
class USStockAdvisor:
    """CrewAI crew for analysing U.S. equities."""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Agents

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
            tools=[fund_tool, sector_valuation_tool],
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
            tools=[tech_tool, revenue_chart_tool, price_chart_tool, market_share_tool],
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
    
    # @agent
    # def publishing_agent(self) -> Agent:
    #     return Agent(
    #         role="Equity Research Report Publishing Specialist",
    #         goal=(
    #             "Combine all analyst Markdown sections (macro news, fundamental memo, "
    #             "technical memo) with the three chart images into a single, polished "
    #             "equity-research PDF. Apply professional formatting: clear hierarchy "
    #             "of headings, page-breaks between major sections, beautifully styled "
    #             "tables with numeric alignment, consistent fonts, and descriptive "
    #             "captions beneath each chart. Ensure the final document is press-ready "
    #             "and suitable for institutional investors."
    #         ),
    #         backstory=(
    #             "You’re an expert in WeasyPrint, HTML/CSS paged-media, and corporate "
    #             "report production. Your mission is purely presentation—no investment "
    #             "analysis. Focus on layout quality, typography, and readability."
    #         ),
    #         llm=None,
    #         tools=[markdown_tool, weasy_tool, pdf_merge_tool, image2pdf_tool],
    #         verbose=True,
    #     )

    # Tasks

    @task
    def news_collecting(self) -> Task:
        return Task(
            config=self.tasks_config["news_collecting"],
            agent=self.stock_news_researcher(),
            async_execution=True,
            output_file="us_market_analysis.md",
        )

    @task
    def fundamental_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["fundamental_analysis"],
            agent=self.fundamental_analyst(),
            async_execution=True,
            output_file="fundamental_analysis.md",
        )

    @task
    def technical_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["technical_analysis"],
            agent=self.technical_analyst(),
            async_execution=True,
            output_file="technical_analysis.md",
        )

    @task
    def investment_decision(self) -> Task:
        return Task(
            config=self.tasks_config["investment_decision"],
            agent=self.investment_strategist(),
            context=[self.news_collecting(), self.fundamental_analysis(), self.technical_analysis()],
            output_json=InvestmentDecision,
            output_file="final_decision.json",
        )

    @task
    def price_charting(self) -> Task:
        return Task(
            description="Generate and save the latest 6-month price line chart.",
            agent=self.technical_analyst(),
            tool=[price_chart_tool],
            args={"ticker": "{{ticker}}", "period": "6mo", "interval": "1d"},
            expected_output  = "Path (string) to the saved JPEG chart returned by the tool",
            # output_file="{{ticker}}_price_line.jpg",
    )

    @task
    def revenue_charting(self) -> Task:
        """Generate a revenue-bar chart before the fundamental analyst writes a memo."""
        return Task(
            description=(
                "Generate and save a bar chart of the company's last four "
                "{{ freq | default('annual') }} revenues (in USD billions)."
            ),
            agent=self.technical_analyst(),       
            tool=revenue_chart_tool,                
            args={
                "ticker": "{{ ticker }}",           # passed in via kickoff(inputs=…)
                # let callers override frequency; default to 'annual'
                "freq": "{{ freq | default('annual') }}"
            },
            expected_output=(
                "Path (string) to the saved JPEG revenue-bar chart returned by the tool"
            ),
        )

    @task
    def market_share_charting(self) -> Task:
        """Generate a donut plot of <ticker> vs its largest peers by market-cap."""
        return Task(
            description=(
                "Generate and save a donut chart that shows {{ ticker }}’s market-"
                "cap share versus its {{ n_peers | default(5) }} largest peers.  "
                "Peers are fetched via Finnhub’s *Company Peers* API, market-caps "
                "via Yahoo Finance, and the chart is saved as a JPEG image."
            ),

            agent=self.technical_analyst(),
            tool=market_share_tool,                
            args={
                "ticker": "{{ ticker }}",           
                "n_peers": "{{ n_peers | default(5) }}",

                "api_key": "{{ api_key | default(None) }}"
            },

            expected_output=(
                "Path (string) to the saved JPEG donut chart returned by the tool"
            )

        )



    # @task
    # def render_section_pdfs(self) -> Task:
    #     return Task(
    #         description=(
    #             "Convert the three analyst markdown files and three JPEG charts to PDF. "
    #             "Use Markdown → HTML → PDF for the .md files and Image → PDF for the JPEGs. "
    #             "Write the six resulting PDF filenames to section_pdfs.json."
    #         ),
    #         agent=self.publishing_agent(),
    #         tools=[markdown_tool, weasy_tool, image2pdf_tool],
    #         args={
    #             "md_files": [
    #                 ("us_market_analysis.md",   "macro.pdf"),
    #                 ("fundamental_analysis.md", "fundamental.pdf"),
    #                 ("technical_analysis.md",   "technical.pdf"),
    #             ],
    #             "img_files": [
    #                 ("{{ ticker|upper }}_price_line.jpg",        "price_chart.pdf"),
    #                 ("{{ ticker|upper }}_revenue_bar.jpg",       "revenue_chart.pdf"),
    #                 ("{{ ticker|upper }}_market_share_all.jpg",  "market_share_chart.pdf"),
    #             ],
    #         },
    #         output_file="section_pdfs.json",
    #         expected_output="JSON list of six PDF filenames"
    #     )




    # @task
    # def merge_report(self) -> Task:
    #     return Task(
    #         description=(
    #             "Merge the six PDFs listed in section_pdfs.json into a single file "
    #             "named US_equity_research_report.pdf."
    #         ),
    #         agent=self.publishing_agent(),
    #         tool=pdf_merge_tool,
    #         args={
    #             "files": "{{ file_list_from('section_pdfs.json') }}",
    #             "out":   "US_equity_research_report.pdf"
    #         },
    #         expected_output="US_equity_research_report.pdf"
    #     )



    # Crew

    @crew
    def crew(self) -> Crew: 
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )