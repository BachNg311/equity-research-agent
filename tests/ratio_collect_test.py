import yfinance as yf
import json


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

def try_convert_to_float(value):
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None

def get_sector_ratios(etf_symbol):
    etf = yf.Ticker(etf_symbol)
    info = etf.info

    pe_ratio = try_convert_to_float(info.get("trailingPE"))
    pb_ratio = try_convert_to_float(info.get("priceToBook"))

    return {
        "P/E": pe_ratio,
        "P/B": pb_ratio
    }

def scrape_us_industry_ratios():
    result = {}
    for sector, etf_symbol in sector_etfs.items():
        print(f"Fetching data for sector: {sector} (ETF: {etf_symbol})")
        ratios = get_sector_ratios(etf_symbol)
        result[sector] = ratios
    return result

if __name__ == "__main__":
    print("📊 Scraping US sector P/E and P/B ratios using yfinance ETF proxies...")
    sector_data = scrape_us_industry_ratios()

    json_output = json.dumps(sector_data, indent=4)
    print("\n--- US Sector Valuation Data ---")
    print(json_output)

    with open("us_sector_valuation.json", "w") as f:
        json.dump(sector_data, f, indent=4)
    print("\nData saved to 'us_sector_valuation.json'")
