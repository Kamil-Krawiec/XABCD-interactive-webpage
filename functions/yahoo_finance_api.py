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
    # cached_df, last_cached_end_time = read_cached_data(cache_file)

    # Convert start and end times to datetime objects
    start_dt = parse_date(start_str)
    end_dt = parse_date(end_str) if end_str else datetime.utcnow()

    symbol = adjust_symbol_for_yfinance(symbol)

    # Fetch new data using yfinance
    print(f"Fetching data from Yahoo Finance for {symbol} from {start_dt} to {end_dt} with interval {interval}")

    try:
        new_df = yf.download(
            tickers=symbol,
            start=start_dt,
            end=end_dt,
            interval=interval,
            progress=False
        )
        new_df.reset_index(inplace=True)

        # Check if columns are MultiIndex
        if isinstance(new_df.columns, pd.MultiIndex):
            # Drop the 'Ticker' level from the columns
            # Since we're fetching data for a single ticker, we can drop the second level
            new_df.columns = new_df.columns.droplevel(1)

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
            '30m': timedelta(minutes=30), '60m': timedelta(minutes=60),
            '90m': timedelta(minutes=90), '1d': timedelta(days=1),
            '5d': timedelta(days=5), '1wk': timedelta(weeks=1),
            '1mo': timedelta(days=30), '3mo': timedelta(days=90)
        }
        duration = interval_durations.get(interval)

        new_df['close_time'] = new_df['open_time']+duration

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

    if new_df.empty:
        print(f"No data fetched for {symbol}.")
        return None

    return new_df


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
