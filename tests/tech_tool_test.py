import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ────────────────────────── Main wrapper ──────────────────────────
def tech_data_tool(symbol: str) -> str:
    """Get technical-analysis data for a US stock."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=500)
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            return f"❌ No historical data found for stock: {symbol}"

        # Standardize column names and drop second level if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.rename(columns={"Close": "close", "High": "high", "Low": "low", "Open": "open"})

        tech = calculate_indicators(df)
        s_r_text = find_support_resistance(df)

        current_price = df["close"].iloc[-1]
        recent_prices = df["close"].iloc[-5:-1]
        ind = tech.iloc[-1]

        result = f"""\n📈 Stock Symbol: {symbol.upper()}
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
{s_r_text}

TECHNICAL INTERPRETATION:
{get_technical_analysis(ind, current_price)}
"""
        return result

    except Exception as e:
        return f"❌ Error retrieving technical data: {e}"


# ────────────────────────── Indicators ──────────────────────────
def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["SMA_200"] = df["close"].rolling(200).mean()
    df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["RSI_14"] = df["RSI_14"].fillna(50)

    m = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["BB_Middle"] = m
    df["BB_Upper"] = m + 2 * std
    df["BB_Lower"] = m - 2 * std

    return df


# ────────────────── Support / resistance via pivots ──────────────
def find_support_resistance(df: pd.DataFrame, max_levels: int = 3, thresh: float = 0.03) -> str:
    data = df.copy()

    pivot_highs = data["high"][(data["high"].shift(1) < data["high"]) &
                               (data["high"].shift(-1) < data["high"])].tolist()
    pivot_lows = data["low"][(data["low"].shift(1) > data["low"]) &
                             (data["low"].shift(-1) > data["low"])].tolist()
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
        return [float(np.mean(c)) for c in clusters]

    resistances = [r for r in cluster(pivot_highs) if r > current][:max_levels]
    supports = [s for s in cluster(pivot_lows) if s < current][:max_levels]

    lines = ["Resistance Levels:"]
    lines += [f"  • R{i}: ${r:,.2f}" for i, r in enumerate(resistances, 1)] or ["  • (none)"]
    lines.append("")
    lines.append("Support Levels:")
    lines += [f"  • S{i}: ${s:,.2f}" for i, s in enumerate(supports, 1)] or ["  • (none)"]
    return "\n".join(lines)


# ────────────────────── Interpretation helper ───────────────────
def get_technical_analysis(ind, current):
    out = []

    if current > ind["SMA_200"] and ind["SMA_50"] > ind["SMA_200"]:
        out.append("• Long-term trend: BULLISH")
    elif current < ind["SMA_200"] and ind["SMA_50"] < ind["SMA_200"]:
        out.append("• Long-term trend: BEARISH")
    else:
        out.append("• Long-term trend: NEUTRAL")

    if current > ind["SMA_20"] and ind["SMA_20"] > ind["SMA_50"]:
        out.append("• Short-term trend: BULLISH")
    elif current < ind["SMA_20"] and ind["SMA_20"] < ind["SMA_50"]:
        out.append("• Short-term trend: BEARISH")
    else:
        out.append("• Short-term trend: NEUTRAL")

    if ind["RSI_14"] > 70:
        out.append("• RSI: OVERBOUGHT (>70)")
    elif ind["RSI_14"] < 30:
        out.append("• RSI: OVERSOLD (<30)")
    else:
        out.append(f"• RSI: NEUTRAL ({ind['RSI_14']:.2f})")

    out.append("• MACD: BULLISH" if ind["MACD"] > ind["MACD_Signal"] else "• MACD: BEARISH")

    if current > ind["BB_Upper"]:
        out.append("• Bollinger: OVERBOUGHT (above upper band)")
    elif current < ind["BB_Lower"]:
        out.append("• Bollinger: OVERSOLD (below lower band)")
    else:
        pct = (current - ind["BB_Lower"]) / (ind["BB_Upper"] - ind["BB_Lower"])
        if pct > 0.8:
            out.append("• Bollinger: Near overbought zone")
        elif pct < 0.2:
            out.append("• Bollinger: Near oversold zone")
        else:
            out.append("• Bollinger: Neutral zone")

    return "\n".join(out)


# ────────────────────────── Simple test ─────────────────────────
def test_tech_data_tool():
    print("🧪 Testing with MSFT:")
    print(tech_data_tool("MSFT"))

    print("\n🧪 Testing with INVALID:")
    print(tech_data_tool("INVALID"))


if __name__ == "__main__":
    test_tech_data_tool()
