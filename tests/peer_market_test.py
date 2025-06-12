import requests

def get_peers_finnhub(ticker, api_key):
    """
    Fetches peer companies for the given ticker using Finnhub API.

    Parameters:
    - ticker (str): The stock ticker symbol.
    - api_key (str): Your Finnhub API key.

    Returns:
    - list: A list of peer ticker symbols.
    """
    url = f"https://finnhub.io/api/v1/stock/peers?symbol={ticker}&token={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        peers = response.json()
        return peers # Remove the main ticker from the list
    else:
        print(f"Error fetching peers: {response.status_code}")
        return []
api_key = 'd11ks29r01qjtpe7i2bgd11ks29r01qjtpe7i2c0'  # Replace with your actual API key
ticker = 'TSLA'
peers = get_peers_finnhub(ticker, api_key)
print(f"Peers for {ticker}: {peers}")
