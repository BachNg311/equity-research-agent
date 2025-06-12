# -------- prerequisites ----------
# pip install yfinance matplotlib --upgrade

import yfinance as yf
import matplotlib.pyplot as plt
from typing import Literal

def plot_revenue_barchart(
        ticker: str,
        freq: Literal["annual", "quarterly"] = "annual",
        save: bool = True
    ) -> None:
    """
    Plot a bar chart of company revenue.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g. 'AAPL', 'TSLA').
    freq : {'annual', 'quarterly'}, default 'annual'
        Use annual or quarterly income-statement data.
    save : bool, default True
        If True, saves the figure as '<TICKER>_revenue_bar.jpg'.

    Returns
    -------
    None – shows a matplotlib figure (and optionally saves it).
    """
    tkr = yf.Ticker(ticker.upper())

    # 1) Fetch income-statement dataframe
    fin_df = tkr.financials if freq == "annual" else tkr.quarterly_financials
    if fin_df.empty:
        raise ValueError(f"No {freq} financial data found for {ticker}.")

    # 2) Locate revenue row (may appear as 'Total Revenue' or 'Revenue')
    possible_keys = ["Total Revenue", "Revenue"]
    revenue_row = next((k for k in possible_keys if k in fin_df.index), None)
    if revenue_row is None:
        raise KeyError(f"Revenue row not in {freq} income statement for {ticker}.")

    revenues = fin_df.loc[revenue_row].dropna()
    # Keep most-recent 4 columns, reverse to chronological order
    revenues = revenues.iloc[:4][::-1]

    # 3) Plot
    plt.figure(figsize=(8, 5))
    plt.bar(revenues.index.strftime("%Y-%m") if freq == "quarterly"
            else revenues.index.year.astype(str),
            revenues / 1e9)   # convert to billions for readability
    plt.title(f"{ticker.upper()} {'Quarterly' if freq=='quarterly' else 'Annual'} Revenue")
    plt.ylabel("Revenue (USD billions)")
    plt.xlabel("Period")
    plt.gca().set_axisbelow(True)
    # plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # 4) Save if requested
    if save:
        plt.savefig(f"{ticker.upper()}_revenue_bar.jpg", dpi=300)

    plt.show()

# EXAMPLE USAGE ---------------------------------------------------------------
if __name__ == "__main__":
    plot_revenue_barchart("TSLA", freq="annual")        #   4 most-recent fiscal years
    # plot_revenue_barchart("TSLA", freq="quarterly")   #   4 most-recent quarters
