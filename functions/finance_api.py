import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# Mapping supported intervals to Kraken's minute-based values
INTERVAL_MAPPING = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
    '1w': 10080,
    '2w': 20160,
    '1M': 43200
}


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
    Fetch historical OHLCV data for a crypto asset via Kraken public API, with optional caching (no API key required).

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

    # Parse dates
    start_dt = parse_date(start_str)
    end_dt = parse_date(end_str) if end_str else datetime.utcnow()
    if end_dt <= start_dt:
        raise ValueError("end_str must be after start_str")

    # Prepare cache
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"{symbol.upper()}_{interval}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
    )
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=['open_time', 'close_time'])

    # Kraken pair formatting: assume USD quote
    pair = f"X{symbol.upper()}ZUSD" if symbol.upper() == 'BTC' else f"{symbol.upper()}USD"
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        'pair': pair,
        'interval': INTERVAL_MAPPING[interval],
        'since': int(start_dt.timestamp())
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    json_data = resp.json()
    if json_data.get('error'):
        raise ValueError(f"Kraken API error for {pair}: {json_data['error']}")

    result_key = next(k for k in json_data['result'].keys() if k != 'last')
    raw = json_data['result'][result_key]
    if not raw:
        raise ValueError(f"No data returned for {pair} from Kraken API.")

    # Build DataFrame
    df = pd.DataFrame(raw, columns=[
        'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
    ])
    # Convert types
    df['open_time'] = pd.to_datetime(df['time'], unit='s')
    df['close_time'] = df['open_time'] + pd.to_timedelta(INTERVAL_MAPPING[interval], unit='m')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter time range
    mask = (df['open_time'] >= start_dt) & (df['open_time'] < end_dt)
    df = df.loc[mask].reset_index(drop=True)

    result = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]
    # Cache result
    result.to_csv(cache_file, index=False)
    return result
