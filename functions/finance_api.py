import os
import time
import pandas as pd
from datetime import datetime, timedelta

import ccxt 
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
    """Parse a date string in 'YYYY-MM-DD' or 'DD-MM-YYYY' format or pass through datetime."""
    if isinstance(date_input, datetime):
        return date_input
    for fmt in ('%Y-%m-%d', '%d-%m-%Y'):
        try:
            return datetime.strptime(date_input, fmt)
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {date_input}")

# ---------- Binance helpers ----------

def _normalize_for_binance(symbol: str) -> str:
    """
    Accepts: 'BTC', 'BTCUSD', 'BTC/USDT', 'XBT/USD', etc.
    Returns a Binance-style unified symbol, favoring USDT quotes.
    """
    s = symbol.strip().upper().replace(" ", "")
    # Map XBT -> BTC
    s = s.replace("XBT", "BTC")

    if "/" in s:
        base, quote = s.split("/", 1)
    else:
        # If no slash, assume 3-letter quote at end; default to USDT if none.
        if len(s) > 3 and s[-4:] in ("USDT", "USDC"):
            base, quote = s[:-4], s[-4:]
        elif len(s) > 3:
            base, quote = s[:-3], s[-3:]
        else:
            base, quote = s, "USDT"

    # Binance spot markets overwhelmingly use USDT (BUSD is legacy/limited).
    if quote in ("USD", "USDC", "BUSD"):
        quote = "USDT"
    return f"{base}/{quote}"

def _pick_binance_symbol(exchange: ccxt.binance, candidate: str) -> str:
    """
    If 'candidate' is not listed, try common fallbacks (USDT/BUSD/USD/USDC).
    Returns a valid symbol present in exchange.markets or raises ValueError.
    """
    markets = exchange.markets
    if candidate in markets:
        return candidate

    base, quote = candidate.split("/")
    fallbacks = [
        f"{base}/USDT",
        f"{base}/USD",
        f"{base}/USDC",
        f"{base}/BUSD",
    ]
    for s in fallbacks:
        if s in markets:
            return s

    # Build a helpful hint
    examples = [m for m in markets.keys() if m.startswith(base + "/")]
    examples.sort()
    hint = f" Available on Binance for {base}: {', '.join(examples[:10])}" if examples else ""
    raise ValueError(f"Symbol '{candidate}' not found on binance.{hint}")

def _ensure_timeframe(exchange: ccxt.binance, interval: str) -> str:
    """
    Return a ccxt timeframe supported by Binance for the requested interval.
    - If interval is directly supported: return it
    - If interval == '2w': we will fetch '1d' and resample to 14 days later
    - Otherwise: raise with helpful message
    """
    tfs = getattr(exchange, "timeframes", {})
    if interval in tfs:
        return interval
    if interval == "2w":
        # We will fetch '1d' and resample to 14D in pandas.
        if "1d" in tfs:
            return "1d"
    # Not supported:
    available = ", ".join(sorted(tfs.keys()))
    raise ValueError(f"Timeframe '{interval}' not supported on Binance. Available: {available}")

# ---------- Main API ----------

def get_historical_data(symbol, interval, start_str, end_str=None, cache_dir='cache'):
    """
    Fetch historical OHLCV data via Binance (ccxt) with pagination from start_str -> end_str (or now).

    Parameters:
    - symbol (str): e.g., 'BTC', 'BTC/USDT', 'ETHUSDT'
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
    cache_file = os.path.join(
        cache_dir, f"{safe_symbol}_{interval}_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}.csv"
    )
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=['open_time', 'close_time'])

    # --- Exchange init (Binance spot) ---
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    exchange.load_markets()

    # Normalize & validate symbol for Binance
    unified = _normalize_for_binance(symbol)
    symbol_on_binance = _pick_binance_symbol(exchange, unified)

    # Pick fetch timeframe
    fetch_tf = _ensure_timeframe(exchange, interval)
    # If we're going to resample later (2w), we actually fetch 1d
    actual_fetch_tf = "1d" if (interval == "2w" and fetch_tf == "1d") else fetch_tf

    ms_per_candle = exchange.parse_timeframe(actual_fetch_tf) * 1000
    since_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Reasonable per-call limit for Binance (max 1000)
    per_call_limit = 1000

    rows = []
    seen_t = set()
    safety = 200000

    while since_ms < end_ms and safety > 0:
        batch = exchange.fetch_ohlcv(symbol_on_binance, actual_fetch_tf, since=since_ms, limit=per_call_limit)
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

        # If we got fewer than requested and we're near the end, we can stop
        if len(batch) < per_call_limit and last_t + ms_per_candle >= end_ms:
            break

    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])

    df = pd.DataFrame(rows, columns=["time_ms", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["time_ms"], unit="ms", utc=False)
    if interval == "2w" and actual_fetch_tf == "1d":
        # Resample daily → 14 days
        # Keep OHLCV semantics:
        df = df.set_index("open_time").sort_index()
        o = df["open"].resample("14D", label="left", closed="left").first()
        h = df["high"].resample("14D", label="left", closed="left").max()
        l = df["low"].resample("14D", label="left", closed="left").min()
        c = df["close"].resample("14D", label="left", closed="left").last()
        v = df["volume"].resample("14D", label="left", closed="left").sum()
        df = pd.concat([o, h, l, c, v], axis=1).dropna(how="any")
        df = df.reset_index()  # open_time back as column
        step_minutes = INTERVAL_MAPPING["2w"]
    else:
        # No resample
        df = df.sort_values("time_ms").reset_index(drop=True)
        step_minutes = INTERVAL_MAPPING[interval]

    # Compute close_time from open_time + step
    df["close_time"] = df["open_time"] + pd.to_timedelta(step_minutes, unit="m")

    # Exact range filter
    mask = (df["open_time"] >= start_dt) & (df["open_time"] < end_dt)
    df = df.loc[mask].reset_index(drop=True)

    result = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    result.to_csv(cache_file, index=False)
    return result
