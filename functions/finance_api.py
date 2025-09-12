import os
import time
import pandas as pd
from datetime import datetime
import ccxt

# Mapping supported intervals to minutes (used to compute close_time and step size)
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

def _normalize_symbol_for_unified(symbol: str) -> str:
    """
    Accepts 'BTC', 'BTCUSD', 'BTC/USDT', 'XBT/USD', etc.
    If only base is provided, default to USD quote.
    Uses ccxt unified symbols (so 'BTC/USD' works on Kraken, Bitfinex, Bybit, etc.).
    """
    s = symbol.upper().replace(" ", "")
    if "/" in s:
        return s
    # base+quote or just base
    known_quotes = ("USD", "USDT", "EUR", "USDC", "BTC")
    if any(s.endswith(q) for q in known_quotes) and len(s) > 3:
        # e.g. BTCUSD -> BTC/USD
        return s[:-3] + "/" + s[-3:]
    # default quote
    return f"{s}/USD"

def get_historical_data(symbol, interval, start_str, end_str=None, cache_dir='cache'):
    """
    Fetch historical OHLCV data via ccxt with pagination from start_str -> end_str (or now).

    Parameters:
    - symbol (str): e.g., 'BTC', 'BTC/USD', 'BTCUSDT'
    - interval (str): '1m','5m','15m','30m','1h','4h','1d','1w','2w','1M'
    - start_str (str or datetime)
    - end_str (str or datetime, optional): defaults to now UTC
    - cache_dir (str)

    Returns:
    - pd.DataFrame with columns [open_time, open, high, low, close, volume, close_time]
    """
    if interval not in INTERVAL_MAPPING:
        raise ValueError(f"Unsupported interval: {interval}. Supported: {list(INTERVAL_MAPPING.keys())}")

    # Parse dates (naive UTC timestamps)
    start_dt = parse_date(start_str)
    end_dt = parse_date(end_str) if end_str else datetime.utcnow()
    if end_dt <= start_dt:
        raise ValueError("end_str must be after start_str")

    # Cache path (sanitize symbol for filename)
    os.makedirs(cache_dir, exist_ok=True)
    safe_symbol = symbol.upper().replace("/", "")
    cache_file = os.path.join(
        cache_dir,
        f"{safe_symbol}_{interval}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
    )
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=['open_time', 'close_time'])

    # Exchange selection (default to kraken; override with CCXT_EXCHANGE env)
    exchange_id = os.getenv("CCXT_EXCHANGE", "kraken")
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown exchange '{exchange_id}'. Try CCXT_EXCHANGE=kraken/bitfinex2/bybit/etc.")
    exchange = getattr(ccxt, exchange_id)({
        "enableRateLimit": True,
    })

    # Load markets and validate timeframe support
    exchange.load_markets()
    if hasattr(exchange, "timeframes") and exchange.timeframes:
        if interval not in exchange.timeframes:
            raise ValueError(
                f"Timeframe '{interval}' not supported on {exchange_id}. "
                f"Available: {sorted(exchange.timeframes.keys())}"
            )

    unified_symbol = _normalize_symbol_for_unified(symbol)
    if unified_symbol not in exchange.markets:
        # Try to load markets again or give a helpful error
        # Some venues use 'XBT/USD' vs 'BTC/USD'; ccxt normally maps both to unified 'BTC/USD'
        # If still missing, let user know what exists for the base.
        bases = [m for m in exchange.markets.keys() if m.startswith(unified_symbol.split('/')[0] + "/")]
        hint = f" Examples: {', '.join(bases[:10])}" if bases else ""
        raise ValueError(f"Symbol '{unified_symbol}' not found on {exchange_id}.{hint}")

    # Convert to ms for ccxt
    since_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Reasonable per-call limit (many exchanges accept 500–1000)
    per_call_limit = 1000
    ms_per_candle = exchange.parse_timeframe(interval) * 1000

    all_rows = []
    seen = set()
    safety_loops = 200000  # hard guard

    while since_ms < end_ms and safety_loops > 0:
        candles = exchange.fetch_ohlcv(unified_symbol, interval, since=since_ms, limit=per_call_limit)
        if not candles:
            break

        for t, o, h, l, c, v in candles:
            if t >= end_ms:
                break
            if t not in seen:
                all_rows.append((t, o, h, l, c, v))
                seen.add(t)

        last_t = candles[-1][0]
        # Advance by exactly one candle to avoid duplicates
        next_since = last_t + ms_per_candle
        if next_since <= since_ms:
            next_since = since_ms + ms_per_candle
        since_ms = next_since

        safety_loops -= 1
        time.sleep(getattr(exchange, "rateLimit", 1000) / 1000.0)

        # If server returned fewer than requested, you may be near the end of history
        if len(candles) < per_call_limit and last_t + ms_per_candle >= end_ms:
            break

    if not all_rows:
        # Return empty frame with expected schema
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])

    # Build DataFrame identical to your previous shape
    df = pd.DataFrame(all_rows, columns=['time_ms', 'open', 'high', 'low', 'close', 'volume'])
    # Pandas naive UTC timestamps (to match your previous code’s behavior)
    df['open_time'] = pd.to_datetime(df['time_ms'], unit='ms')
    df['close_time'] = df['open_time'] + pd.to_timedelta(INTERVAL_MAPPING[interval], unit='m')

    # Filter exact range & sort
    mask = (df['open_time'] >= start_dt) & (df['open_time'] < end_dt)
    df = df.loc[mask].drop_duplicates(subset='time_ms').sort_values('time_ms').reset_index(drop=True)

    result = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]
    # Cache result
    result.to_csv(cache_file, index=False)
    return result
