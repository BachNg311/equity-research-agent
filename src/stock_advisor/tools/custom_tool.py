from __future__ import annotations

"""Custom tools for US equity analysis – fundamental & technical.

These tools are designed to plug‑and‑play with CrewAI.
They replace the original VN‑specific implementation (based on `vnstock`) with
universally available U.S. data sources via Yahoo Finance (`yfinance`).
All public free endpoints – no API keys required.
"""

from typing import Any, Optional, Type

import numpy as np
import pandas as pd
import yfinance as yf
from crewai.tools import BaseTool
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

###############################################################################
# ────────────────────────────── Shared schema ────────────────────────────── #
###############################################################################


class USStockInput(BaseModel):
    """Input schema for US stock tools."""

    ticker: str = Field(..., description="U.S. stock ticker symbol (e.g. AAPL).")


###############################################################################
# ─────────────────────────── Fundamental data tool ────────────────────────── #
###############################################################################


class USFundDataTool(BaseTool):
    """Fetch quarterly fundamentals & key ratios for U.S. equities via yfinance."""

    name: str = "Fundamental data lookup (US market)"
    description: str = (
        "Retrieve key valuation ratios (P/E, P/B, ROE, etc.) and last 4 "
        "quarterly income‑statement trends for a given U.S. stock ticker using "
        "Yahoo Finance data."
    )
    args_schema: Type[BaseModel] = USStockInput

    ###########################################################################

    def _run(self, ticker: str) -> str:  # noqa: D401 – CrewAI naming convention
        try:
            tk = yf.Ticker(ticker.upper())

            # Basic info & sector/industry
            info = tk.info or {}
            long_name = info.get("longName") or info.get("shortName", "N/A")
            industry = info.get("industry", "N/A")
            sector = info.get("sector", "N/A")

            # Recent valuation ratios (some keys may be missing)
            pe_ratio = info.get("trailingPE", "N/A")
            pb_ratio = info.get("priceToBook", "N/A")
            roe = info.get("returnOnEquity", "N/A")
            roa = info.get("returnOnAssets", "N/A")
            eps = info.get("trailingEps", "N/A")
            # Debt/Equity ratio isn't directly provided – approximate from balance‑sheet
            try:
                bs = tk.quarterly_balance_sheet  # columns are dates, rows are items
                total_debt = bs.loc["Total Debt"].iloc[0]
                total_equity = bs.loc["Total Stockholder Equity"].iloc[0]
                de_ratio = total_debt / total_equity if total_equity else np.nan
            except Exception:
                de_ratio = np.nan

            # EV/EBITDA (ttm) – yfinance keys
            evebitda = info.get("enterpriseToEbitda", "N/A")
            profit_margin = info.get("profitMargins", "N/A")

            # Last 4 quarters income statement
            inc = tk.quarterly_income_stmt  # rows × columns (item × date)
            last_4 = inc.iloc[:, :4].T  # flip to (date × item)

            quarterly_trends: list[str] = []
            for i, (idx, row) in enumerate(last_4.iterrows(), 1):
                rev = row.get("Total Revenue", np.nan)
                gp = row.get("Gross Profit", np.nan)
                npi = row.get("Net Income", np.nan)

                fm = lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"  # noqa: E731
                quarterly_trends.append(
                    f"Quarter T‑{i}:\n"
                    f"  • Revenue: {fm(rev)}\n"
                    f"  • Gross Profit: {fm(gp)}\n"
                    f"  • Net Income: {fm(npi)}\n"
                )

            lines = [
                f"Ticker: {ticker.upper()}",
                f"Company: {long_name}",
                f"Sector / Industry: {sector} / {industry}",
                f"P/E: {pe_ratio}",
                f"P/B: {pb_ratio}",
                f"ROE: {roe}",
                f"ROA: {roa}",
                f"Profit Margin: {profit_margin}",
                f"EPS (ttm): {eps}",
                f"Debt‑to‑Equity: {de_ratio if pd.notna(de_ratio) else 'N/A'}",  # type: ignore[arg‑type]
                f"EV/EBITDA: {evebitda}",
                "\nLAST 4 QUARTERS:",
                *quarterly_trends,
            ]

            return "\n".join(lines)
        except Exception as exc:  # pragma: no cover – generic catch for tool
            return f"Error fetching fundamental data: {exc}"


###############################################################################
# ─────────────────────────── Technical data tool ─────────────────────────── #
###############################################################################


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
            end = datetime.utcnow()
            start = end - timedelta(days=500)  # ensure ≥200 candles

            df = yf.download(
                ticker.upper(),
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

            tech = self._calc_indicators(df)
            s_r = self._support_resistance(df)

            current = df["close"].iloc[-1]
            recent = df["close"].iloc[-5:-1]
            last = tech.iloc[-1]

            out = [
                f"📈 Ticker: {ticker.upper()}",
                f"Current Price: ${current:,.2f}",
                "\nRECENT CLOSES:",
                *(f"  • T‑{i}: ${p:,.2f}" for i, p in enumerate(recent[::-1], 1)),
                "\nLATEST INDICATORS:",
                f"  SMA(20): {last['SMA_20']:.2f}",
                f"  SMA(50): {last['SMA_50']:.2f}",
                f"  SMA(200): {last['SMA_200']:.2f}",
                f"  EMA(12): {last['EMA_12']:.2f}",
                f"  EMA(26): {last['EMA_26']:.2f}",
                f"  RSI(14): {last['RSI_14']:.2f}",
                f"  MACD: {last['MACD']:.2f}",
                f"  MACD Signal: {last['MACD_Signal']:.2f}",
                f"  MACD Hist: {last['MACD_Hist']:.2f}",
                f"  Bollinger Upper: {last['BB_Upper']:.2f}",
                f"  Bollinger Middle: {last['BB_Middle']:.2f}",
                f"  Bollinger Lower: {last['BB_Lower']:.2f}",
                "\nSUPPORT / RESISTANCE:",
                s_r,
            ]
            return "\n".join(out)

        except Exception as exc:
            return f"❌ Error fetching technical data: {exc}"

    @staticmethod
    def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    def _support_resistance(df: pd.DataFrame, window: int = 10, thresh: float = 0.03) -> str:
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
        out.extend(f"  • R{i}: ${lvl:,.2f}" for i, lvl in enumerate(resist, 1))
        if not resist:
            out.append("  • (no significant resistance found)")
        out.extend(f"  • S{i}: ${lvl:,.2f}" for i, lvl in enumerate(supp, 1))
        if not supp:
            out.append("  • (no significant support found)")
        return "\n".join(out)




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