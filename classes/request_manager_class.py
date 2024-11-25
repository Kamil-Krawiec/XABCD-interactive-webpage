import os
from datetime import datetime, date, timedelta
import pandas as pd
import requests


class RequestManager:
    def __init__(self, api_key: str, cache_path='./cache/sp500_cache.csv'):
        self.api_key = api_key
        self.data_cache = {}  # Cache to store the data for different symbols
        self.fear_greed_cache = None  # Cache for the Fear and Greed Index data
        self.cache_path = cache_path  # Local storage path for the S&P 500 data

    def fetch_market_index_SP500(self, symbol: str):
        # If symbol data is in the in-memory cache, return it
        if symbol in self.data_cache:
            return self.data_cache[symbol]

        # Get today's date
        today = date.today()

        # Determine if today is a weekend (Saturday=5, Sunday=6)
        is_weekend = today.weekday() >= 5

        # Determine the last trading day
        if is_weekend:
            # If today is Saturday (5), subtract one day to get Friday (4)
            # If today is Sunday (6), subtract two days to get Friday (4)
            days_since_friday = today.weekday() - 4
            last_trading_day = today - timedelta(days=days_since_friday)
        else:
            last_trading_day = today

        # Check if data exists locally
        if os.path.exists(self.cache_path):
            print(f"Loading data from local storage: {self.cache_path}")
            df = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
            df.index = df.index.date  # Convert index to date format

            # Get the last date in the CSV
            last_cached_date = df.index[-1]

            if last_cached_date >= last_trading_day:
                print(f"Data is up to date for {last_trading_day}. Using local cache.")
                self.data_cache[symbol] = df
                return df
            else:
                if is_weekend:
                    print("Today is a weekend. Market is closed. Using cached data.")
                    self.data_cache[symbol] = df
                    return df
                else:
                    print(f"Data is outdated. Last available date: {last_cached_date}. Fetching new data from API...")
        else:
            if is_weekend:
                print("Today is a weekend and no local data is available. Cannot fetch new data. Returning None.")
                return None

        # Fetch data from the API if local data is outdated or doesn't exist and today is a weekday
        try:
            response = requests.get(
                f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={self.api_key}"
            )
            response.raise_for_status()  # Raise an exception for non-200 responses

            data = response.json()
            if 'Time Series (Daily)' in data:
                # Convert the data to a pandas DataFrame
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index', dtype=float)

                # Convert index to datetime, then extract only the date
                df.index = pd.to_datetime(df.index).date

                # Sort by date
                df.sort_index(inplace=True)

                # Store the fetched data in cache
                self.data_cache[symbol] = df

                # Save the DataFrame to local storage for future use
                df.to_csv(self.cache_path)
                print(f"Data saved to local storage: {self.cache_path}")

                return df
            else:
                print("Error: No 'Time Series (Daily)' data found in response.")
                return None
        except requests.RequestException as e:
            print(f"Error fetching market index data: {e}")
            # If there's an error fetching new data and local data exists, load from local storage
            if os.path.exists(self.cache_path):
                print(f"Falling back to local storage: {self.cache_path}")
                df = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
                df.index = df.index.date  # Convert index to date format
                self.data_cache[symbol] = df
                return df

            return None

    def get_SP500_index_at_time(self, symbol: str, timestamp: pd.Timestamp):
        """
        Retrieves the S&P 500 index value at a specific timestamp.
        If the exact date is not available, it attempts to retrieve the value
        from previous trading days.

        Args:
        - symbol: The symbol for the S&P 500 index (e.g., 'SPY').
        - timestamp: The timestamp for which the index value is requested.

        Returns:
        - The S&P 500 index value or None if not available.
        """
        # Fetch the market index data for the S&P 500
        df = self.fetch_market_index_SP500(symbol)

        if df is not None:
            # Extract the target date
            target_date = timestamp.date()
            # Adjusted to use datetime.timedelta
            prev_day = target_date - timedelta(days=1)
            prev_prev_day = target_date - timedelta(days=2)

            # Ensure the index is in datetime.date format
            if not isinstance(df.index, pd.Index):
                df.index = pd.to_datetime(df.index).date

            # Check if the target date exists in the index
            if target_date in df.index:
                # Return the value in the '4. close' column for the matched date
                return df.loc[target_date, '4. close']
            elif prev_day in df.index:
                return df.loc[prev_day, '4. close']
            elif prev_prev_day in df.index:
                return df.loc[prev_prev_day, '4. close']
            else:
                # Find the last available date before the target date
                available_dates = df.index[df.index < target_date]
                if len(available_dates) > 0:
                    last_available_date = available_dates[-1]
                    return df.loc[last_available_date, '4. close']
                else:
                    print(f"No data found for {target_date} or previous dates.")
                    return None
        else:
            # Handle the case where no data is fetched
            print(f"No data available for symbol: {symbol}")
            return None

    def get_fear_greed_index_at_time(self, timestamp: pd.Timestamp):
        """
        Fetch the Fear and Greed Index for a specific date, with caching support.

        Args:
        - timestamp: The time for which the Fear and Greed Index is requested.

        Returns:
        - Fear and Greed Index at that specific date, or None if not found
        """
        target_date = timestamp.date()  # Convert timestamp to date

        # Check if data is cached
        if self.fear_greed_cache is not None:
            # Search for the matching date in the cached data
            cached_value = self._find_fear_greed_in_cache(target_date)
            if cached_value is not None:
                return cached_value

        # Fetch data from the API if not cached or date not found in cache
        try:
            url = 'https://api.alternative.me/fng/?limit=0'
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for non-200 status codes

            data = response.json()['data']

            # Cache the entire dataset
            self.fear_greed_cache = data

            # Look for the matching date in the newly fetched data
            return self._find_fear_greed_in_cache(target_date)

        except requests.RequestException as e:
            print(f"Error fetching Fear and Greed Index: {e}")
            return None

    def _find_fear_greed_in_cache(self, target_date):
        """
        Helper method to find the Fear and Greed Index for a specific date in the cache.
        """
        if self.fear_greed_cache is not None:
            for entry in self.fear_greed_cache:
                entry_date = datetime.fromtimestamp(float(entry['timestamp'])).date()
                if entry_date == target_date:
                    return float(entry['value'])  # Return the index value
                elif entry_date == target_date - timedelta(days=1):
                    return float(entry['value'])
                elif entry_date == target_date - timedelta(days=2):
                    return float(entry['value'])

        return None  # Return None if no match is found
