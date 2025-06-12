import yfinance as yf
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from typing import List, Dict


class MarketShareInput(BaseModel):
    ticker: str = Field(..., description="Main stock ticker")
    peers: List[str] = Field(..., description="List of competitor tickers (excluding the main ticker)")


class MarketShareChartTool:
    def __init__(self, input_data: Dict):
        self.input = MarketShareInput(**input_data)

    def run(self) -> str:
        try:
            tickers = [self.input.ticker.upper()] + [p.upper() for p in self.input.peers]
            data = {t: yf.Ticker(t).info.get("marketCap", 0) for t in tickers}
            data = {k: v for k, v in data.items() if v > 0}

            if not data:
                return "❌ No market cap data found."

            labels = list(data.keys())
            sizes = list(data.values())
            total = sum(sizes)
            labels = [f"{l} ({s / total:.1%})" for l, s in zip(labels, sizes)]

            plt.figure(figsize=(6, 6))
            plt.pie(sizes, labels=labels, startangle=45, wedgeprops=dict(width=0.4))
            plt.title(f"Market Share in Industry ({self.input.ticker})")
            filename = f"{self.input.ticker.lower()}_market_share_donut.png"
            plt.savefig(filename)
            return f"✅ Donut chart saved to {filename}"

        except Exception as e:
            return f"❌ Failed to generate chart: {e}"


if __name__ == "__main__":
    input_data = {
        "ticker": "NVDA",
        "peers": ['AVGO', 'AMD', 'TXN', 'QCOM', 'MU', 'ADI', 'INTC', 'MRVL', '6697.T']
    }

    tool = MarketShareChartTool(input_data)
    result = tool.run()
    print(result)
