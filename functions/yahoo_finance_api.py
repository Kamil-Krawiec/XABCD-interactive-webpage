import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from config import YAHOO_SYMBOL_MAPPING


def get_historical_data(symbol, interval, start_str, end_str=None, cache_dir='cache'):
    """
    Fetch historical market data for a given symbol and interval, with caching.

    Parameters:
    - symbol (str): The ticker symbol to fetch data for.
    - interval (str): The data interval (e.g., '1m', '5m', '1h', '1d').
    - start_str (str or datetime): The start date in 'YYYY-MM-DD' format or as a datetime object.
    - end_str (str or datetime, optional): The end date in 'YYYY-MM-DD' format or as a datetime object.
      Defaults to the current UTC time if not provided.
    - cache_dir (str): Directory to store cached data files.

    Returns:
    - pandas.DataFrame: The historical data with adjusted columns.
    """
    if interval is None:
        raise ValueError(f"Unsupported interval: {interval}")

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file path
    cache_file = os.path.join(cache_dir, f"{symbol}_{interval}.csv")

    # Try to load cached data
    cached_df, last_cached_end_time = read_cached_data(cache_file)

    # Convert start and end times to datetime objects
    start_dt = parse_date(start_str)
    end_dt = parse_date(end_str) if end_str else datetime.utcnow()

    # If cached data covers the required range, return cached data
    if last_cached_end_time and last_cached_end_time >= end_dt - timedelta(minutes=1):
        df = cached_df[cached_df['open_time'] >= start_dt].copy()
        df.reset_index(drop=True, inplace=True)
        return df

    # Adjust start date if cached data exists
    if last_cached_end_time:
        start_dt = last_cached_end_time + timedelta(minutes=1)

    # Fetch new data using yfinance
    yf_symbol = adjust_symbol_for_yfinance(symbol)
    print(f"Fetching data from Yahoo Finance for {yf_symbol} from {start_dt} to {end_dt} with interval {interval}")

    try:
        new_df = yf.download(
            tickers=yf_symbol,
            start=start_dt,
            end=end_dt,
            interval=interval,
            progress=False
        )
        print(new_df)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return cached_df

    if new_df.empty:
        print(f"No new data fetched for {symbol}.")
        df = cached_df[cached_df['open_time'] >= start_dt].copy()
        df.reset_index(drop=True, inplace=True)
        return df

    # Process and combine data
    new_df = process_new_data(new_df, interval)
    combined_df = combine_data(cached_df, new_df)
    save_data_to_cache(combined_df, cache_file)

    # Filter data for the requested date range
    combined_df = combined_df[
        (combined_df['open_time'] >= start_dt) & (combined_df['open_time'] <= end_dt)
        ].copy()
    combined_df.reset_index(drop=True, inplace=True)

    return combined_df


def adjust_symbol_for_yfinance(symbol):
    """
    Adjust the symbol to match yfinance's expected format.

    Parameters:
    - symbol (str): The original symbol.

    Returns:
    - str: The adjusted symbol for yfinance.
    """
    return YAHOO_SYMBOL_MAPPING.get(symbol.upper(), symbol)


def parse_date(date_input):
    """
    Parse a date input into a datetime object.

    Parameters:
    - date_input (str or datetime): The date input.

    Returns:
    - datetime: The parsed datetime object.
    """
    if isinstance(date_input, datetime):
        return date_input
    elif isinstance(date_input, str):
        # Try to parse in 'Y-M-D' format
        try:
            return datetime.strptime(date_input, '%Y-%m-%d')
        except ValueError:
            pass  # If it fails, try the next format

        # Try to parse in 'D-M-Y' format
        try:
            return datetime.strptime(date_input, '%d-%m-%Y')
        except ValueError:
            pass  # If it fails, raise an error

        # If both formats fail, raise an error
        raise ValueError(f"Invalid date format: {date_input}")
    else:
        raise ValueError(f"Invalid date input: {date_input}")


def read_cached_data(cache_file):
    """
    Read cached data from a CSV file.

    Parameters:
    - cache_file (str): The path to the cache file.

    Returns:
    - tuple:
        - pandas.DataFrame: The cached data.
        - datetime or None: The last cached end time.
    """
    if os.path.exists(cache_file):
        print(f"Reading cached data from {cache_file}")
        cached_df = pd.read_csv(cache_file)
        cached_df['open_time'] = pd.to_datetime(cached_df['open_time'])
        cached_df['close_time'] = pd.to_datetime(cached_df['close_time'])
        last_cached_end_time = cached_df['open_time'].max()
        return cached_df, last_cached_end_time
    else:
        return pd.DataFrame(), None


def process_new_data(new_df, yf_interval):
    """
    Process the newly fetched data from yfinance, handling MultiIndex columns.

    Parameters:
    - new_df (pandas.DataFrame): The new data fetched from yfinance.
    - yf_interval (str): The interval used in yfinance.

    Returns:
    - pandas.DataFrame: The processed data with single-level columns.
    """
    # Reset index to get 'Datetime' as a column
    new_df.reset_index(inplace=True)

    # Check if columns are MultiIndex
    if isinstance(new_df.columns, pd.MultiIndex):
        # Drop the 'Ticker' level from the columns
        # Since we're fetching data for a single ticker, we can drop the second level
        new_df.columns = new_df.columns.droplevel(1)
    else:
        # If columns are single-level, proceed as usual
        pass

    # Rename columns to match expected format
    new_df.rename(columns={
        'Datetime': 'open_time',
        'Date': 'open_time',  # In case of daily data
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    }, inplace=True)

    # Compute 'close_time' column
    interval_durations = {
        '1m': timedelta(minutes=1), '2m': timedelta(minutes=2),
        '5m': timedelta(minutes=5), '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30), '60m': timedelta(hours=1),
        '90m': timedelta(minutes=90), '1d': timedelta(days=1),
        '5d': timedelta(days=5), '1wk': timedelta(weeks=1),
        '1mo': timedelta(days=30), '3mo': timedelta(days=90)
    }
    duration = interval_durations.get(yf_interval, timedelta(minutes=1))
    new_df['close_time'] = new_df['open_time'] + duration

    # Convert numeric columns to float
    numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    existing_numeric_cols = [col for col in numeric_cols if col in new_df.columns]
    new_df[existing_numeric_cols] = new_df[existing_numeric_cols].astype(float)

    # Drop 'adj_close' if not needed
    if 'adj_close' in new_df.columns:
        new_df.drop(columns=['adj_close'], inplace=True)

    return new_df


def combine_data(cached_df, new_df):
    """
    Combine cached data with new data.

    Parameters:
    - cached_df (pandas.DataFrame): The cached data.
    - new_df (pandas.DataFrame): The new data.

    Returns:
    - pandas.DataFrame: The combined data.
    """
    if not cached_df.empty:
        combined_df = pd.concat([cached_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.drop_duplicates(subset=['open_time'], inplace=True)
    combined_df.sort_values('open_time', inplace=True)
    return combined_df


def save_data_to_cache(df, cache_file):
    """
    Save DataFrame to a cache file.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to save.
    - cache_file (str): The path to the cache file.
    """
    df.to_csv(cache_file, index=False)
