from binance import Client
import os
from datetime import datetime
import pandas as pd
from datetime import datetime as time


def get_historical_data(client, symbol, interval, start_str=None, end_str=time.today(), cache_dir='cache'):
    """
    Fetches historical candlestick data from Binance with caching.

    :param client: Client object from the Binance API
    :param symbol: Market symbol, e.g., 'BTCUSDT'
    :param interval: Candlestick interval, e.g., '1h', '1d', '15m'
    :param start_str: Start date as a string, e.g., 'YYYY-MM-DD' or '1 Jan, 2021'
    :param end_str: End date as a string, e.g., 'YYYY-MM-DD' or '1 Jan, 2021' (optional)
    :param cache_dir: Directory to store cache files
    :return: DataFrame containing candlestick data
    """
    # Ensure the cache directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Define the cache file path
    cache_file = os.path.join(cache_dir, f"{symbol}_{interval}.csv")

    # Try to read cached data
    if os.path.exists(cache_file):
        print(f"Reading cached data from {cache_file}")
        cached_df = pd.read_csv(cache_file, parse_dates=['open_time', 'close_time'])
    else:
        print(f"No cache found at {cache_file}")
        cached_df = pd.DataFrame()

    # Convert start_str and end_str to datetime objects
    if start_str:
        start_dt = pd.to_datetime(start_str)
    else:
        start_dt = None
    if end_str:
        end_dt = pd.to_datetime(end_str)
    else:
        end_dt = None

    missing_intervals = []

    if not cached_df.empty:
        # Get the earliest and latest timestamps in the cache
        cached_start = cached_df['open_time'].min()
        cached_end = cached_df['open_time'].max()

        # Check if the cached data covers the requested range
        if start_dt:
            if start_dt < cached_start:
                # Need to fetch data from start_dt to cached_start
                missing_intervals.append(
                    (start_dt.strftime('%Y-%m-%d %H:%M:%S'), cached_start.strftime('%Y-%m-%d %H:%M:%S')))

        if end_dt:
            if end_dt > cached_end:
                # Need to fetch data from cached_end to end_dt
                missing_intervals.append(
                    (cached_end.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')))
    else:
        # Cache is empty, need to fetch all data
        if start_dt is None:
            raise ValueError("start_str must be provided when cache is empty")
        missing_intervals.append(
            (start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S') if end_dt else None))

    new_data = []

    # Fetch missing data
    for fetch_start, fetch_end in missing_intervals:
        print(f"Fetching data from Binance for {symbol} {interval} from {fetch_start} to {fetch_end}")
        candles = client.get_historical_klines(symbol, interval, fetch_start, fetch_end)
        for candle in candles:
            new_data.append({
                'open_time': datetime.fromtimestamp(candle[0] / 1000),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5]),
                'close_time': datetime.fromtimestamp(candle[6] / 1000),
                'quote_asset_volume': float(candle[7]),
                'number_of_trades': int(candle[8]),
                'taker_buy_base_asset_volume': float(candle[9]),
                'taker_buy_quote_asset_volume': float(candle[10]),
                'ignore': candle[11]
            })

    if new_data:
        new_df = pd.DataFrame(new_data)
        if not cached_df.empty:
            # Combine with cached data
            combined_df = pd.concat([cached_df, new_df], ignore_index=True)
            # Drop duplicates
            combined_df.drop_duplicates(subset=['open_time'], inplace=True)
            # Sort by open_time
            combined_df.sort_values('open_time', inplace=True)
        else:
            combined_df = new_df
        # Save the combined data back to cache
        combined_df.to_csv(cache_file, index=False)
    else:
        combined_df = cached_df

    return combined_df
