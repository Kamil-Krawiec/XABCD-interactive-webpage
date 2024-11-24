import os
import pandas as pd
from datetime import datetime
from binance.helpers import date_to_milliseconds

def get_historical_data(client, symbol, interval, start_str, end_str=None, cache_dir='cache'):
    # Ensure cache directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Define cache file path
    cache_file = os.path.join(cache_dir, f"{symbol}_{interval}.csv")

    # Initialize variables
    all_data = []
    last_cached_end_time = None

    # Try to load cached data
    if os.path.exists(cache_file):
        print(f"Reading cached data from {cache_file}")
        cached_df = pd.read_csv(cache_file)
        cached_df['open_time'] = pd.to_datetime(cached_df['open_time'])
        cached_df['close_time'] = pd.to_datetime(cached_df['close_time'])

        # Get the last timestamp from cached data
        last_cached_end_time = cached_df['open_time'].max()
        all_data.append(cached_df)
    else:
        cached_df = pd.DataFrame()

    # Convert start and end times to milliseconds
    if isinstance(start_str, str):
        start_ts = date_to_milliseconds(start_str)
    else:
        start_ts = int(start_str.timestamp() * 1000)

    if end_str:
        if isinstance(end_str, str):
            end_ts = date_to_milliseconds(end_str)
        else:
            end_ts = int(end_str.timestamp() * 1000)
    else:
        end_ts = None  # Binance will fetch up to the current time if end_ts is None

    # If cached data exists and covers the required range, return cached data
    if last_cached_end_time and start_ts >= int(last_cached_end_time.timestamp() * 1000):
        df = cached_df[cached_df['open_time'] >= pd.to_datetime(start_ts, unit='ms')]
        return df

    # Adjust start timestamp to fetch data after the cached data
    if last_cached_end_time:
        start_ts = int(last_cached_end_time.timestamp() * 1000) + 1  # Add 1 ms to avoid duplication

    # Fetch new data from Binance
    klines = []
    limit = 1000  # Max number of records per request
    while True:
        fetch_end_ts = None
        if end_ts:
            fetch_end_ts = min(end_ts, start_ts + limit * 60 * 1000)

        new_klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_ts,
            endTime=fetch_end_ts
        )

        if not new_klines:
            break

        klines.extend(new_klines)
        start_ts = new_klines[-1][0] + 1  # Start from the next timestamp

        # Break if we've reached the end timestamp
        if end_ts and start_ts >= end_ts:
            break

    if klines:
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        new_df = pd.DataFrame(klines, columns=columns)
        new_df['open_time'] = pd.to_datetime(new_df['open_time'], unit='ms')
        new_df['close_time'] = pd.to_datetime(new_df['close_time'], unit='ms')

        # Convert numeric columns to float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                        'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        new_df[numeric_cols] = new_df[numeric_cols].astype(float)
        new_df['number_of_trades'] = new_df['number_of_trades'].astype(int)

        all_data.append(new_df)

        # Combine new data with cached data
        if not cached_df.empty:
            combined_df = pd.concat([cached_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(subset=['open_time'], inplace=True)
            combined_df.sort_values('open_time', inplace=True)
        else:
            combined_df = new_df

        # Save combined data to cache
        combined_df.to_csv(cache_file, index=False)
    else:
        combined_df = cached_df

    # Filter data for the requested range
    combined_df = combined_df[combined_df['open_time'] >= pd.to_datetime(start_str)]
    if end_str:
        combined_df = combined_df[combined_df['open_time'] <= pd.to_datetime(end_str)]

    combined_df.reset_index(drop=True, inplace=True)

    return combined_df