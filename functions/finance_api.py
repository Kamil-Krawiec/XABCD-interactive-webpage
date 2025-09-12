# functions/finance_api.py
# US-safe backend using Kraken via ccxt. Same inputs/outputs as before.

import os
import time
from datetime import datetime
import pandas as pd

import ccxt  # ensure requirements.txt has: ccxt>=4.0.0

# Mapping supported intervals to minutes (used for close_time and paging step)
INTERVAL_MAPPING = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
    '1w': 10080,
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

# ----------------- helpers -----------------

def _split_symbol_any(s: str):
    """
    Accepts 'BTCUSDT', 'BTC/USDT', 'ETHBTC', 'XBT/USD', etc.
    Returns (base, quote) without slashes, with BTC/XBT normalized to BTC.
    """
    s = s.strip().upper().replace(" ", "")
    s = s.replace("XBT", "BTC")  # unify BTC alias

    if "/" in s:
        base, quote = s.split("/", 1)
    else:
        # try to split by common 3-4 letter quotes
        for qlen in (4, 3):
            if len(s) > qlen:
                base, quote = s[:-qlen], s[-qlen:]
                if quote in ("USDT", "USDC", "BUSD", "USD", "EUR", "BTC"):
                    break
        else:
            base, quote = s, "USDT"
    return base, quote

def _pick_kraken_symbol(exchange: ccxt.kraken, user_symbol: str) -> str:
    """
    Given a user symbol, choose a valid Kraken market.
    Tries sensible fallbacks: USDT -> USD -> USDC -> EUR -> BTC.
    """
    base, quote = _split_symbol_any(user_symbol)
    markets = exchange.markets

    def ok(sym: str) -> bool:
        return sym in markets

    candidates = [
        f"{base}/{quote}",
        f"{base}/USDT",
        f"{base}/USD",
        f"{base}/USDC",
        f"{base}/EUR",
        f"{base}/BTC",
    ]
    for c in candidates:
        if ok(c):
            return c

    # Build a helpful hint
    examples = [m for m in markets.keys() if m.startswith(base + "/")]
    examples.sort()
    hint = f" Available on Kraken for {base}: {', '.join(examples[:12])}" if examples else ""
    raise ValueError(f"Symbol '{user_symbol}' not found on kraken.{hint}")

def _ensure_timeframe(exchange: ccxt.kraken, interval: str) -> str:
    """
    Validate timeframe against Kraken. Kraken supports:
    1,5,15,30,60,240,1440,10080,21600 (minutes) → ccxt keys like '1m','5m','1h','4h','1d','1w','15d'
    We map your app's keys directly; for '2w' we resample '1d' later.
    """
    tfs = getattr(exchange, "timeframes", {}) or {}
    if interval in tfs:
        return interval
    if interval == "2w" and "1d" in tfs:
        return "1d"  # fetch daily and resample to 14D
    available = ", ".join(sorted(tfs.keys()))
    raise ValueError(f"Timeframe '{interval}' not supported on Kraken. Available: {available}")

# ----------------- main API -----------------

def get_historical_data(symbol, interval, start_str, end_str=None, cache_dir='cache'):
    """
    Fetch historical OHLCV via Kraken (ccxt) with pagination from start_str -> end_str (or now).

    Parameters:
    - symbol (str): e.g., 'BTCUSDT', 'ETHBTC', 'BTC/USDT'
    - interval (str): '1m','5m','15m','30m','1h','4h','1d','1w','2w','1M'
    - start_str (str or datetime)
    - end_str (str or datetime, optional): defaults to now UTC
    - cache_dir (str)

    Returns:
    - pd.DataFrame with columns [open_time, open, high, low, close, volume, close_time]
    """
    if interval not in INTERVAL_MAPPING:
        raise ValueError(f"Unsupported interval: {interval}. Supported: {list(INTERVAL_MAPPING.keys())}")

    # Parse times
    start_dt = parse_date(start_str)
    end_dt = parse_date(end_str) if end_str else datetime.utcnow()
    if end_dt <= start_dt:
        raise ValueError("end_str must be after start_str")

    # Cache
    os.makedirs(cache_dir, exist_ok=True)
    safe_symbol = symbol.upper().replace("/", "")
    cache_file = os.path.join(cache_dir, f"{safe_symbol}_{interval}_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}.csv")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=['open_time', 'close_time'])

    # Kraken exchange (US-accessible)
    exchange = ccxt.kraken({"enableRateLimit": True})
    exchange.load_markets()

    # Normalize & validate symbol on Kraken
    symbol_on_kraken = _pick_kraken_symbol(exchange, symbol)

    # Choose a fetch timeframe (resample later if 2w)
    fetch_tf = _ensure_timeframe(exchange, interval)
    actual_fetch_tf = "1d" if (interval == "2w" and fetch_tf == "1d") else fetch_tf

    ms_per_candle = exchange.parse_timeframe(actual_fetch_tf) * 1000
    since_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Kraken usually returns up to ~720 OHLC per call; 500–1000 is fine
    per_call_limit = 720

    rows = []
    seen_t = set()
    safety = 200000

    while since_ms < end_ms and safety > 0:
        batch = exchange.fetch_ohlcv(symbol_on_kraken, actual_fetch_tf, since=since_ms, limit=per_call_limit)
        if not batch:
            break

        for t, o, h, l, c, v in batch:
            if t >= end_ms:
                break
            if t not in seen_t:
                rows.append((t, o, h, l, c, v))
                seen_t.add(t)

        last_t = batch[-1][0]
        next_since = last_t + ms_per_candle
        if next_since <= since_ms:
            next_since = since_ms + ms_per_candle
        since_ms = next_since

        safety -= 1
        time.sleep(getattr(exchange, "rateLimit", 1000) / 1000.0)

        if len(batch) < per_call_limit and last_t + ms_per_candle >= end_ms:
            break

    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])

    # Build DataFrame to match your expected schema
    df = pd.DataFrame(rows, columns=["time_ms", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["time_ms"], unit="ms", utc=False)

    if interval == "2w" and actual_fetch_tf == "1d":
        # Resample daily → 14 days OHLCV
        df = df.set_index("open_time").sort_index()
        o = df["open"].resample("14D", label="left", closed="left").first()
        h = df["high"].resample("14D", label="left", closed="left").max()
        l = df["low"].resample("14D", label="left", closed="left").min()
        c = df["close"].resample("14D", label="left", closed="left").last()
        v = df["volume"].resample("14D", label="left", closed="left").sum()
        df = pd.concat([o, h, l, c, v], axis=1).dropna(how="any").reset_index()
        step_minutes = INTERVAL_MAPPING["2w"]
    else:
        df = df.sort_values("time_ms").reset_index(drop=True)
        step_minutes = INTERVAL_MAPPING[interval]

    df["close_time"] = df["open_time"] + pd.to_timedelta(step_minutes, unit="m")

    # Exact range filter
    mask = (df["open_time"] >= start_dt) & (df["open_time"] < end_dt)
    df = df.loc[mask].reset_index(drop=True)

    result = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    result.to_csv(cache_file, index=False)
    return result
