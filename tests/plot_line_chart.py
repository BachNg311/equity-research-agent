# -------- prerequisites ----------
# pip install yfinance matplotlib --upgrade

import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def plot_price_line(
        ticker: str = "TSLA",
        period: str = "1y",          # '1mo' | '3mo' | '6mo' | '1y' | '5y' | …
        interval: str = "1d"         # '1d' | '1wk' | '1mo'
    ) -> None:
    """
    Download price data via Yahoo Finance and draw a line chart of the close price.

    Parameters
    ----------
    ticker : str
        Stock symbol to fetch (e.g. 'TSLA').
    period : str
        Look-back window; valid yfinance periods include '1mo', '3mo', '6mo', '1y', etc.
    interval : str
        Bar size; typical values are '1d', '1wk', '1mo'.

    Returns
    -------
    None – displays a matplotlib chart.
    """
    # --- 1) get the data
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the symbol or your internet connection.")

    # --- 2) plot
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label=f"{ticker} close price")  # default line color
    plt.title(f"{ticker} closing price – last {period}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    # plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ticker.upper()}_stock_price.jpg", dpi=300)
    plt.show()

# quick demo – last 6 months of TSLA
if __name__ == "__main__":
    plot_price_line("TSLA", period="1y", interval="1d")