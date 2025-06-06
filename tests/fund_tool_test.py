import yfinance as yf

def fund_data_tool_us(symbol: str) -> str:
    """Retrieve stock data for fundamental analysis (US market)."""
    try:
        stock = yf.Ticker(symbol)
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

        return f"""📊 Stock Symbol: {symbol}
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
        return f"❌ Error retrieving data: {e}"


def test_fund_data_tool_us():
    # Valid example: Apple Inc.
    print("Test result for AAPL:")
    print(fund_data_tool_us("MSFT"))

    # Invalid symbol
    print("\nTest result for INVALID:")
    print(fund_data_tool_us("INVALID"))

if __name__ == "__main__":
    test_fund_data_tool_us()
