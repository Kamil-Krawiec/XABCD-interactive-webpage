import os
from datetime import datetime, date
import pandas as pd
import requests


class RequestManager:
    def __init__(self, api_key: str, cache_path='./cache/sp500_cache.csv'):
        self.api_key = api_key
        self.data_cache = {}  # Cache to store the data for different symbols
        self.fear_greed_cache = None  # Cache for the Fear and Greed Index data
        self.cache_path = cache_path  # Local storage path for the S&P 500 data

    def fetch_market_index_SP500(self, symbol: str):
        # If not in cache or local storage, check the in-memory cache
        if symbol in self.data_cache:
            return self.data_cache[symbol]

        # Check if data exists locally first
        if os.path.exists(self.cache_path):
            print(f"Loading data from local storage: {self.cache_path}")
            df = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
            df.index = df.index.date  # Convert index to date format

            # Check if the last available date in the CSV is the current date
            last_cached_date = df.index[-1]  # Get the last date in the CSV
            #previous day
            prev_day = date.today() - pd.Timedelta(days=1)

            if last_cached_date == prev_day:
                print(f"Data is up to date for {prev_day}. Using local cache.")
                self.data_cache[symbol] = df
                return df
            else:
                print(f"Data is outdated. Last available date: {last_cached_date}. Fetching new data from API...")

        # Fetch data from the API if local data is outdated or doesn't exist
        try:
            response = requests.get(
                f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={self.api_key}")
            response.raise_for_status()  # Raise an exception for non-200 responses

            data = response.json()
            if 'Time Series (Daily)' in data:
                # Convert the data to a pandas DataFrame
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index', dtype=float)

                # Convert index to datetime, then extract only the date (removing time)
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
        # Fetch the market index data for the S&P 500
        df = self.fetch_market_index_SP500(symbol)

        if df is not None:
            # Extract the target date
            target_date = timestamp.date()
            prev_day = date.today() - pd.Timedelta(days=1)
            prev_prev_day = date.today() - pd.Timedelta(days=2)

            # Ensure the index is in datetime format if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index).date

            # Check if the target date exists in the index
            if target_date in df.index:
                # Return the value in the '4. close' column for the matched date
                return df.loc[target_date, '4. close']
            elif prev_day:
                return df.loc[prev_day, '4. close']
            elif prev_prev_day:
                return df.loc[prev_prev_day, '4. close']
            else:
                print(f"No data found for {target_date}")
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
                elif entry_date == target_date - pd.Timedelta(days=1):
                    return float(entry['value'])
                elif entry_date == target_date - pd.Timedelta(days=2):
                    return float(entry['value'])

        return None  # Return None if no match is found