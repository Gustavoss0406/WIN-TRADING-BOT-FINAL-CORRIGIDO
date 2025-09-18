import requests

# Substitua pelos seus proxies válidos
proxies = {
    "http": "http://usuario:senha@proxy_host:proxy_port",
    "https": "http://usuario:senha@proxy_host:proxy_port",
}

def get_binance_price(base_url, symbol='BTCUSDT', proxies=None):
    url = f"{base_url}/api/v3/ticker/price"
    params = {'symbol': symbol}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10, proxies=proxies)
        r.raise_for_status()
        data = r.json()
        return float(data['price'])
    except Exception as e:
        print(f"Erro ao buscar preço em {base_url}: {e}")
        return None

if __name__ == "__main__":
    binance_global = "https://api.binance.com"
    binance_us = "https://api.binance.us"

    price_global = get_binance_price(binance_global, proxies=proxies)
    price_us = get_binance_price(binance_us, proxies=proxies)

    print(f"Preço BTCUSDT Binance Global: ${price_global}")
    print(f"Preço BTCUSDT Binance US:     ${price_us}")

    if price_global and price_us:
        diff = abs(price_global - price_us)
        print(f"Diferença absoluta: ${diff:.2f}")
        print(f"Diferença percentual: {diff / price_global * 100:.4f}%")