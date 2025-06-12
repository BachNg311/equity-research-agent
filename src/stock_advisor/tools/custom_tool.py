from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type
import numpy as np
import pandas as pd
import yfinance as yf
from crewai.tools import BaseTool
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, field_validator
import matplotlib
import matplotlib.pyplot as plt
import requests
import os
import json
matplotlib.use("Agg")  

"""
Custom tools for US equity analysis – fundamental & technical.
"""


# Stock Input  
class USStockInput(BaseModel):
    """Input schema for US stock tools."""

    ticker: str = Field(..., description="U.S. stock ticker symbol (e.g. AAPL).")


# US Fundamental Data Tool
class USFundDataTool(BaseTool):
    """Fetch quarterly fundamentals & key ratios for U.S. equities via yfinance."""

    name: str = "Fundamental data lookup (US market)"
    description: str = (
        "Retrieve key valuation ratios (P/E, P/B, ROE, etc.) and last 4 "
        "quarterly income‑statement trends for a given U.S. stock ticker using "
        "Yahoo Finance data."
    )
    args_schema: Type[BaseModel] = USStockInput


    def _run(self, ticker: str) -> str:  
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Company info
            full_name = info.get("longName", "N/A")
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")

            # Financial ratios
            pe_ratio = info.get("trailingPE", "N/A")
            pb_ratio = info.get("priceToBook", "N/A")
            roe = info.get("returnOnEquity", "N/A")
            roa = info.get("returnOnAssets", "N/A")
            eps = info.get("trailingEps", "N/A")
            de = info.get("debtToEquity", "N/A")
            profit_margin = info.get("profitMargins", "N/A")
            evebitda = info.get("enterpriseToEbitda", "N/A")

            # Quarterly income trends (use financials if available)
            try:
                income_stmt = stock.quarterly_financials
                revenue = income_stmt.loc['Total Revenue'].tolist()
                gross_profit = income_stmt.loc['Gross Profit'].tolist()
                net_income = income_stmt.loc['Net Income'].tolist()
                quarters = income_stmt.columns.tolist()
            except Exception:
                revenue = gross_profit = net_income = quarters = []

            quarterly_trends = []
            for i in range(min(4, len(quarters))):
                rev = f"${revenue[i]:,.0f}" if i < len(revenue) and revenue[i] else "N/A"
                gp = f"${gross_profit[i]:,.0f}" if i < len(gross_profit) and gross_profit[i] else "N/A"
                ni = f"${net_income[i]:,.0f}" if i < len(net_income) and net_income[i] else "N/A"

                quarter_info = f"""
Quarter T-{i + 1} ({quarters[i].strftime('%Y-%m-%d')}):
- Revenue: {rev}
- Gross Profit: {gp}
- Net Income: {ni}
"""
                quarterly_trends.append(quarter_info)

            return f"""📊 Stock Symbol: {ticker}
Company Name: {full_name}
Sector: {sector}
Industry: {industry}
P/E Ratio: {pe_ratio}
P/B Ratio: {pb_ratio}
ROE: {roe}
ROA: {roa}
Profit Margin: {profit_margin}
EPS: {eps}
D/E Ratio: {de}
EV/EBITDA: {evebitda}

📈 LATEST 4 QUARTERS TREND:
{''.join(quarterly_trends)}
"""
        except Exception as e:
            return f"Error retrieving data: {e}"


# US Technical Data Tool
class USTechDataTool(BaseTool):
    """Compute common technical indicators for a U.S. equity using yfinance."""

    name: str = "Technical data lookup (US market)"
    description: str = (
        "Retrieve OHLC price history (200 trading days) via Yahoo Finance and "
        "compute SMA/EMA (20/50/200, 12/26), RSI‑14, MACD, Bollinger Bands, and "
        "the three nearest support/resistance clusters."
    )
    args_schema: Type[BaseModel] = USStockInput

    def _run(self, ticker: str) -> str:
        try:
            end = datetime.now()
            start = end - timedelta(days=500)  

            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                return f"❌ No price history available for {ticker.upper()}."

            # Flatten MultiIndex if exists
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df = df.rename(columns=str.lower)

            tech = self.calc_indicators(df)
            s_r = self.support_resistance(df)

            current_price = df["close"].iloc[-1]
            recent_prices = df["close"].iloc[-5:-1]
            ind = tech.iloc[-1]

            tech_interpretation = self.get_technical_analysis(ind, current_price)

            result = f"""\n📈 Stock Symbol: {ticker.upper()}
Current Price: ${current_price:,.2f}

RECENT CLOSING PRICES:
- T-1: ${recent_prices.iloc[-1]:,.2f}
- T-2: ${recent_prices.iloc[-2]:,.2f}
- T-3: ${recent_prices.iloc[-3]:,.2f}
- T-4: ${recent_prices.iloc[-4]:,.2f}

TECHNICAL INDICATORS (latest):
- SMA (20):  ${ind['SMA_20']:,.2f}
- SMA (50):  ${ind['SMA_50']:,.2f}
- SMA (200): ${ind['SMA_200']:,.2f}
- EMA (12):  ${ind['EMA_12']:,.2f}
- EMA (26):  ${ind['EMA_26']:,.2f}
- RSI (14):  {ind['RSI_14']:.2f}
- MACD:       {ind['MACD']:.2f}
- MACD Signal:{ind['MACD_Signal']:.2f}
- MACD Hist.: {ind['MACD_Hist']:.2f}
- Bollinger Upper:  ${ind['BB_Upper']:,.2f}
- Bollinger Middle: ${ind['BB_Middle']:,.2f}
- Bollinger Lower:  ${ind['BB_Lower']:,.2f}

SUPPORT & RESISTANCE:
{s_r}

TECHNICAL INTERPRETATION:
{tech_interpretation}
"""
            return result

        except Exception as exc:
            return f"❌ Error fetching technical data: {exc}"

    @staticmethod
    def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Moving Averages
        data["SMA_20"] = data["close"].rolling(20).mean()
        data["SMA_50"] = data["close"].rolling(50).mean()
        data["SMA_200"] = data["close"].rolling(200).mean()
        data["EMA_12"] = data["close"].ewm(span=12, adjust=False).mean()
        data["EMA_26"] = data["close"].ewm(span=26, adjust=False).mean()

        # MACD
        data["MACD"] = data["EMA_12"] - data["EMA_26"]
        data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
        data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

        # RSI‑14
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        data["RSI_14"] = 100 - (100 / (1 + rs))
        data["RSI_14"] = data["RSI_14"].fillna(50)

        # Bollinger Bands
        data["BB_Middle"] = data["close"].rolling(20).mean()
        std = data["close"].rolling(20).std()
        data["BB_Upper"] = data["BB_Middle"] + 2 * std
        data["BB_Lower"] = data["BB_Middle"] - 2 * std

        return data

    @staticmethod
    def support_resistance(df: pd.DataFrame, window: int = 10, thresh: float = 0.03) -> str:
        data = df.copy()

        data["local_max"] = data["high"].rolling(window=window, center=True).apply(
            lambda x: x.iloc[len(x) // 2] == x.max()
        )
        data["local_min"] = data["low"].rolling(window=window, center=True).apply(
            lambda x: x.iloc[len(x) // 2] == x.min()
        )

        highs = data.loc[data["local_max"] == 1, "high"].tolist()
        lows = data.loc[data["local_min"] == 1, "low"].tolist()
        current = data["close"].iloc[-1]

        def cluster(levels):
            if len(levels) == 0:
                return []
            levels = sorted(levels)
            clusters = [[levels[0]]]
            for lvl in levels[1:]:
                if abs((lvl - clusters[-1][-1]) / clusters[-1][-1]) < thresh:
                    clusters[-1].append(lvl)
                else:
                    clusters.append([lvl])
            return [np.mean(c) for c in clusters]

        resist = [r for r in sorted(cluster(highs)) if r > current][:3]
        supp = [s for s in sorted(cluster(lows), reverse=True) if s < current][:3]

        out = []
        out.extend(f"- R{i}: ${lvl:,.2f}" for i, lvl in enumerate(resist, 1))
        if not resist:
            out.append("- (no significant resistance found)")
        out.extend(f"- S{i}: ${lvl:,.2f}" for i, lvl in enumerate(supp, 1))
        if not supp:
            out.append("- (no significant support found)")
        return "\n".join(out)
    
    @staticmethod
    def get_technical_analysis(ind: pd.Series, current: float) -> str:
        out = []

        if current > ind["SMA_200"] and ind["SMA_50"] > ind["SMA_200"]:
            out.append("- Long-term trend: BULLISH")
        elif current < ind["SMA_200"] and ind["SMA_50"] < ind["SMA_200"]:
            out.append("- Long-term trend: BEARISH")
        else:
            out.append("- Long-term trend: NEUTRAL")

        if current > ind["SMA_20"] and ind["SMA_20"] > ind["SMA_50"]:
            out.append("- Short-term trend: BULLISH")
        elif current < ind["SMA_20"] and ind["SMA_20"] < ind["SMA_50"]:
            out.append("- Short-term trend: BEARISH")
        else:
            out.append("- Short-term trend: NEUTRAL")

        if ind["RSI_14"] > 70:
            out.append("- RSI: OVERBOUGHT (>70)")
        elif ind["RSI_14"] < 30:
            out.append("- RSI: OVERSOLD (<30)")
        else:
            out.append(f"- RSI: NEUTRAL ({ind['RSI_14']:.2f})")

        out.append("- MACD: BULLISH" if ind["MACD"] > ind["MACD_Signal"] else "• MACD: BEARISH")

        if current > ind["BB_Upper"]:
            out.append("- Bollinger: OVERBOUGHT (above upper band)")
        elif current < ind["BB_Lower"]:
            out.append("- Bollinger: OVERSOLD (below lower band)")
        else:
            pct = (current - ind["BB_Lower"]) / (ind["BB_Upper"] - ind["BB_Lower"])
            if pct > 0.8:
                out.append("- Bollinger: Near overbought zone")
            elif pct < 0.2:
                out.append("- Bollinger: Near oversold zone")
            else:
                out.append("- Bollinger: Neutral zone")

        return "\n".join(out)

# Empty Input for US Sector Valuation Tool
class EmptyInput(BaseModel):
    """No input required for US Sector Valuation Tool."""
    pass

# US Sector Valuation Tool
class USSectorValuationTool(BaseTool):
    name: str = "US Sector Valuation Scraper"
    description: str = (
        "Fetches current average P/E and P/B ratios for US market sectors "
        "using ETF proxies via yfinance, and saves the result as JSON."
    )
    args_schema: Type[BaseModel] = EmptyInput

    def _run(self) -> str:
        sector_etfs = {
            "Technology": "XLK",
            "Financials": "XLF",
            "Health Care": "XLV",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Industrials": "XLI",
            "Energy": "XLE",
            "Materials": "XLB",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Communication Services": "XLC",
            "Banking": "KBE",
            "Semiconductors": "SOXX",
            "Biotechnology": "IBB",
            "Aerospace & Defense": "ITA",
            "Retail": "XRT",
            "Metals & Mining": "XME",
            "Oil & Gas Exploration": "XOP",
            "Clean Energy": "ICLN",
            "Agribusiness": "MOO",
            "Transportation": "IYT",
            "Infrastructure": "PAVE"
        }

        def try_convert_to_float(value: Optional[float]) -> Optional[float]:
            try:
                return float(value) if value is not None else None
            except (ValueError, TypeError):
                return None

        def get_sector_ratios(etf_symbol: str) -> Dict[str, Optional[float]]:
            etf = yf.Ticker(etf_symbol)
            info = etf.info
            return {
                "P/E": try_convert_to_float(info.get("trailingPE")),
                "P/B": try_convert_to_float(info.get("priceToBook"))
            }

        result = {}
        for sector, symbol in sector_etfs.items():
            print(f"🔍 Fetching {sector} ({symbol})...")
            result[sector] = get_sector_ratios(symbol)

        json_output = json.dumps(result, indent=4)
        with open("us_sector_valuation.json", "w") as f:
            f.write(json_output)

        return f"✅ Sector valuation data saved to 'us_sector_valuation.json'\n{json_output}"



# Stock Price Line Chart Input
class PriceChartInput(USStockInput):
    period: str = Field("6mo", description="Look-back window (1mo, 3mo, 6mo, 1y, etc.)")
    interval: str = Field("1d", description="Bar size (1d, 1wk, 1mo)")

# Stock Price Line Chart Tool
class StockPriceLineChartTool(BaseTool):
    name: str = "Stock price line chart (US)"
    description: str = "Draws and saves a line chart of close-price history."
    args_schema: Type[BaseModel] = PriceChartInput

    def _run(self, ticker: str, period: str = "6mo", interval: str = "1d") -> str:
        df = yf.download(ticker.upper(), period=period, interval=interval, progress=False)
        if df.empty:
            return f"❌ No data for {ticker}."

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df["Close"])
        ax.set(title=f"{ticker.upper()} close – last {period}",
               xlabel="Date", ylabel="Price (USD)")
        # ax.grid(True, ls="--", alpha=.5)
        outfile = f"{ticker.upper()}_price_line.jpg"
        plt.tight_layout(); plt.savefig(outfile, dpi=300); plt.close(fig)
        return outfile

# Revenue Bar Chart Input
class RevenueChartInput(USStockInput):
    freq: Literal["annual", "quarterly"] = Field("annual", description="Use annual or quarterly statement")

# Revenue Bar Chart Tool
class RevenueBarChartTool(BaseTool):
    name: str = "Revenue bar chart (US)"
    description: str = "Builds a bar chart of the last 4 revenues (annual/quarterly)."
    args_schema: Type[BaseModel] = RevenueChartInput

    def _run(self, ticker: str, freq: str = "annual") -> str:
        fin = yf.Ticker(ticker.upper())
        stmt = fin.financials if freq == "annual" else fin.quarterly_financials
        if stmt.empty:
            return f"❌ No {freq} income statement for {ticker}."

        key = "Total Revenue" if "Total Revenue" in stmt.index else "Revenue"
        revenue = stmt.loc[key].dropna().iloc[:4][::-1]  

        fig, ax = plt.subplots(figsize=(8, 4))
        labels = (revenue.index.year.astype(str) if freq == "annual"
                  else revenue.index.strftime("%Y-%m"))
        ax.bar(labels, revenue/1e9)  # USD bn
        ax.set(title=f"{ticker.upper()} {freq.title()} revenue",
               ylabel="USD (billions)")
        # ax.grid(axis="y", ls="--", alpha=.5)
        outfile = f"{ticker.upper()}_revenue_bar.jpg"
        plt.tight_layout(); plt.savefig(outfile, dpi=300); plt.close(fig)
        return outfile

# Market Share All Peers Input
class MarketShareAllPeersInput(USStockInput):
    """Ticker is required and API key must be provided."""
    api_key: Optional[str] = Field(
        None,
        description="Finnhub API key",
    )

class MarketShareAllPeersDonutTool(BaseTool):
    """Plot a donut chart of <ticker> vs **all** peers returned by Finnhub."""

    name: str = "Market share donut chart (all peers)"
    description: str = (
        "Fetches the full peer list from Finnhub for a U.S. ticker, gets each "
        "company’s market cap via Yahoo Finance, and draws a donut whose legend "
        "shows labels in the form 'SYM (xx.x %)'."
    )
    args_schema: Type[BaseModel] = MarketShareAllPeersInput

    _PEER_URL = "https://finnhub.io/api/v1/stock/peers"

    
    def _run(self, ticker: str, api_key: Optional[str] = None) -> str:
        api_key = os.getenv("FINHUB_API_KEY")
        if not api_key:
            return "❌ Finnhub API key not supplied (arg or FINNHUB_API_KEY env)."

        # peer symbols from Finnhub
        try:
            resp = requests.get(
                self._PEER_URL,
                params={"symbol": ticker.upper(), "token": api_key},
                timeout=8,
            )
            resp.raise_for_status()
            peers: list[str] = resp.json() or []
        except Exception as exc:
            return f"❌ Finnhub peers API error: {exc}"

        symbols = list({ticker.upper(), *peers})           

        # market caps via yfinance 
        caps: dict[str, int] = {}
        for sym in symbols:
            try:
                cap = yf.Ticker(sym).info.get("marketCap")
                if cap:
                    caps[sym] = cap
            except Exception:
                continue

        if len(caps) < 2:
            return f"❌ Not enough market-cap data for {ticker} and peers."

        # extract target, keep peers separate
        target_cap = caps.pop(ticker.upper(), None)
        if target_cap is None:                         
            target_cap, _ = caps.popitem()

        labels_syms = [ticker.upper(), *caps.keys()]
        sizes       = [target_cap,     *caps.values()]
        total       = sum(sizes)

        # legend labels “SYM (xx.x %)” 
        legend_labels = [
            f"{sym} ({100 * val / total:.1f} %)" for sym, val in zip(labels_syms, sizes)
        ]

        # plot donut 
        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, _ = ax.pie(
            sizes,
            wedgeprops=dict(width=0.35),
            startangle=90,
            labels=None,               
        )
        ax.set(
            aspect="equal",
            title=f"{ticker.upper()} vs ALL peers\nMarket-cap share",
        )

        # legend 
        ax.legend(
            wedges,
            legend_labels,
            title="Market share",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
        )

        outfile = f"{ticker.upper()}_market_share_all.jpg"
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close(fig)
        return outfile


class FilePathInput(BaseModel):
    """A single markdown file (absolute or relative) plus an optional CSS file."""
    path: str = Field(..., description="Path to a Markdown file to convert")
    css: str | None = Field(
        default=None,
        description="Optional CSS file; if omitted, a default stylesheet is used",
    )

    
    @field_validator("path")
    def file_must_exist(cls, v: str) -> str:   
        if not Path(v).is_file():
            raise FileNotFoundError(f"Markdown file not found: {v}")
        return v


class HTMLStringInput(BaseModel):
    """Raw HTML string and a target PDF filename."""
    html: str = Field(..., description="Already-rendered HTML to print to PDF")
    out: str = Field(
        ..., description="Output PDF path (e.g. 'section1.pdf' or 'report.pdf')"
    )


class MergeListInput(BaseModel):
    """List of PDFs to combine and a destination filename."""
    files: List[str] = Field(..., description="PDF paths in the order to merge")
    out: str = Field(..., description="Destination PDF (e.g. 'final_report.pdf')")

    @field_validator("files")
    def all_paths_exist(cls, v: List[str]) -> List[str]:     
        missing = [f for f in v if not Path(f).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing PDFs for merge: {', '.join(missing)}")
        return v

class MarkdownRenderTool(BaseTool):
    """Convert Markdown to HTML with heading fixes & CSS link."""
    name: str = "Markdown → HTML"
    description: str = (
        "Convert a Markdown file to raw HTML (tables, fenced code, sane lists)."
    )
    args_schema: Type[BaseModel] = FilePathInput

    def _run(self, path: str, css: str = "report.css") -> str:
        import markdown, pathlib
        txt = pathlib.Path(path).read_text(encoding="utf-8")
        html = markdown.markdown(txt, extensions=["tables", "fenced_code", "sane_lists"])
        return f"<link rel='stylesheet' href='{css}'>{html}"

class WeasyPrintTool(BaseTool):
    """Turn HTML (string) into a standalone PDF using WeasyPrint."""
    name: str = "HTML → PDF (WeasyPrint)"
    description: str = "Render an HTML string to a PDF file using WeasyPrint."
    args_schema: Type[BaseModel] = HTMLStringInput
    def _run(self, html: str, out: str) -> str:
        from weasyprint import HTML
        HTML(string=html, base_url=".").write_pdf(out)
        return out

class ImageToPdfInput(BaseModel):
    img_path: str = Field(..., description="Path to a JPEG/PNG file")
    out: str      = Field(..., description="Output PDF filename")

class ImageToPdfTool(BaseTool):
    name: str = "Image → PDF"
    description: str = "Embed a single image in HTML and export as a one-page PDF."
    args_schema: Type[BaseModel] = ImageToPdfInput

    def _run(self, img_path: str, out: str) -> str:   # names now match schema
        from weasyprint import HTML
        html = f"<img src='{img_path}' style='width:100%;'>"
        HTML(string=html, base_url='.').write_pdf(out)
        return out

class PdfMergeTool(BaseTool):
    """Merge multiple PDF files into one."""
    name: str = "Merge PDFs"
    description: str = "Concatenate multiple PDF files into a single document."
    args_schema: Type[BaseModel] = MergeListInput
    def _run(self, files: list[str], out: str) -> str:
        from pypdf import PdfReader, PdfWriter
        writer = PdfWriter()
        for f in files:
            for page in PdfReader(f).pages:
                writer.add_page(page)
        writer.write(out); writer.close()
        return out

class FileReadToolSchema(BaseModel):
    """Input for FileReadTool."""

    file_path: str = Field(..., description="Mandatory file full path to read the file")
    start_line: Optional[int] = Field(1, description="Line number to start reading from (1-indexed)")
    line_count: Optional[int] = Field(None, description="Number of lines to read. If None, reads the entire file")


class FileReadTool(BaseTool):
    """A tool for reading file contents.

    This tool inherits its schema handling from BaseTool to avoid recursive schema
    definition issues. The args_schema is set to FileReadToolSchema which defines
    the required file_path parameter. The schema should not be overridden in the
    constructor as it would break the inheritance chain and cause infinite loops.

    The tool supports two ways of specifying the file path:
    1. At construction time via the file_path parameter
    2. At runtime via the file_path parameter in the tool's input

    Args:
        file_path (Optional[str]): Path to the file to be read. If provided,
            this becomes the default file path for the tool.
        **kwargs: Additional keyword arguments passed to BaseTool.

    Example:
        >>> tool = FileReadTool(file_path="/path/to/file.txt")
        >>> content = tool.run()  # Reads /path/to/file.txt
        >>> content = tool.run(file_path="/path/to/other.txt")  # Reads other.txt
        >>> content = tool.run(file_path="/path/to/file.txt", start_line=100, line_count=50)  # Reads lines 100-149
    """

    name: str = "Read a file's content"
    description: str = "A tool that reads the content of a file. To use this tool, provide a 'file_path' parameter with the path to the file you want to read. Optionally, provide 'start_line' to start reading from a specific line and 'line_count' to limit the number of lines read."
    args_schema: Type[BaseModel] = FileReadToolSchema
    file_path: Optional[str] = None

    def __init__(self, file_path: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize the FileReadTool.

        Args:
            file_path (Optional[str]): Path to the file to be read. If provided,
                this becomes the default file path for the tool.
            **kwargs: Additional keyword arguments passed to BaseTool.
        """
        if file_path is not None:
            kwargs["description"] = (
                f"A tool that reads file content. The default file is {file_path}, but you can provide a different 'file_path' parameter to read another file. You can also specify 'start_line' and 'line_count' to read specific parts of the file."
            )

        super().__init__(**kwargs)
        self.file_path = file_path

    def _run(
        self,
        **kwargs: Any,
    ) -> str:
        file_path = kwargs.get("file_path", self.file_path)
        start_line = kwargs.get("start_line", 1)
        line_count = kwargs.get("line_count", None)

        if file_path is None:
            return (
                "Error: No file path provided. Please provide a file path either in the constructor or as an argument."
            )

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                if start_line == 1 and line_count is None:
                    return file.read()

                start_idx = max(start_line - 1, 0)

                selected_lines = [
                    line
                    for i, line in enumerate(file)
                    if i >= start_idx and (line_count is None or i < start_idx + line_count)
                ]

                if not selected_lines and start_idx > 0:
                    return f"Error: Start line {start_line} exceeds the number of lines in the file."

                return "".join(selected_lines)
        except FileNotFoundError:
            return f"Error: File not found at path: {file_path}"
        except PermissionError:
            return f"Error: Permission denied when trying to read file: {file_path}"
        except Exception as e:
            return f"Error: Failed to read file {file_path}. {str(e)}" 


# if __name__ == "__main__":
#     # Example usage of the tools
#     fund_tool = USFundDataTool()
#     print(fund_tool.run(ticker="AAPL"))

#     tech_tool = USTechDataTool()
#     print(tech_tool.run(ticker="AAPL"))

#     price_chart_tool = StockPriceLineChartTool()
#     print(price_chart_tool.run(ticker="AAPL", period="1y", interval="1d"))

#     revenue_chart_tool = RevenueBarChartTool()
#     print(revenue_chart_tool.run(ticker="AAPL", freq="quarterly"))

#     market_share_tool = MarketShareAllPeersDonutTool()
#     print(market_share_tool.run(ticker="AAPL", api_key="YOUR_FINNHUB_API_KEY"))


#     valuation_tool = USSectorValuationTool()
#     print(valuation_tool.run())