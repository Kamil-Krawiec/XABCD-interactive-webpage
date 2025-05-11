import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# Mapping input intervals to Binance API intervals
INTERVAL_MAPPING = {
    '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m',
    '30m': '30m', '1h': '1h', '2h': '2h', '4h': '4h',
    '6h': '6h', '12h': '12h', '1d': '1d', '3d': '3d',
    '1w': '1w', '1M': '1M'
}

# Default quote currency for trading pairs
QUOTE_ASSET = 'USDT'


def parse_date(date_input):
    """
    Parse a date string in 'YYYY-MM-DD' or 'DD-MM-YYYY' format or pass through datetime.
    """
    if isinstance(date_input, datetime):
        return date_input
    for fmt in ('%Y-%m-%d', '%d-%m-%Y'):
        try:
            return datetime.strptime(date_input, fmt)
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {date_input}")


def get_historical_data(symbol, interval, start_str, end_str=None, cache_dir='cache'):
    """
    Fetch historical OHLCV data for a crypto asset via Binance public API, with optional caching (no API key required).

    Parameters:
    - symbol (str): Ticker symbol (e.g., 'BTC').
    - interval (str): Interval key (e.g., '1m', '5m', '1h', '1d', '1w').
    - start_str (str or datetime): Start date in YYYY-MM-DD or DD-MM-YYYY.
    - end_str (str or datetime, optional): End date. Defaults to now UTC.
    - cache_dir (str): Directory to save/read cached CSVs.

    Returns:
    - pd.DataFrame: Columns [open_time, open, high, low, close, volume, close_time].
    """
    # Validate interval
    if interval not in INTERVAL_MAPPING:
        raise ValueError(f"Unsupported interval: {interval}. Supported: {list(INTERVAL_MAPPING.keys())}")

    # Prepare dates and cache
    start_dt = parse_date(start_str)
    end_dt = parse_date(end_str) if end_str else datetime.utcnow()
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"{symbol.upper()}_{interval}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
    )
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=['open_time', 'close_time'])

    # Build Binance URL and params
    pair = f"{symbol.upper()}"
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': pair,
        'interval': INTERVAL_MAPPING[interval],
        'startTime': int(start_dt.timestamp() * 1000),
        'endTime': int(end_dt.timestamp() * 1000),
        'limit': 1000  # max per request
    }

    # Fetch data (may require paging for >1000 points)
    all_data = []
    while True:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        # If fewer than limit returned, done
        if len(data) < params['limit']:
            break
        # Else advance startTime to last open time + 1ms
        last_open = data[-1][0]
        params['startTime'] = last_open + 1

    if not all_data:
        raise ValueError(f"No data returned for {pair} from Binance API.")

    # Parse into DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    # Convert numeric columns
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    result = df[['open_time','open','high','low','close','volume','close_time']].copy()
    # Cache
    result.to_csv(cache_file, index=False)
    return result
