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
        """
        Return a DataFrame of daily bars for `symbol` (e.g. 'SPY').

        Logic
        -----
        1. Try to read the CSV at self.cache_path.
           • If successful, store in `self.data_cache` and return.
           • If the file is missing or cannot be parsed, go to step 2.
        2. Call Alpha Vantage, build a DataFrame, **overwrite** the CSV,
           cache it in memory, and return.
        """

        # ── 1. attempt to use local cache ────────────────────────────────────
        if os.path.exists(self.cache_path):
            try:
                df = pd.read_csv(self.cache_path, index_col=0, parse_dates=[0])
                df.sort_index(inplace=True)
                self.data_cache[symbol] = df
                return df
            except Exception as exc:
                print(f"[Cache] Could not read cache ({exc}); fetching fresh data.")

        # ── 2. fetch full history from Alpha Vantage ─────────────────────────
        url = (
            "https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY&symbol={symbol}"
            f"&outputsize=full&apikey={self.api_key}"
        )
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            payload = r.json()

            ts_key = "Time Series (Daily)"
            if ts_key not in payload:
                print("[Alpha Vantage] No 'Time Series (Daily)' key in response.")
                return None

            df = pd.DataFrame.from_dict(payload[ts_key], orient="index").astype(float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # overwrite / create cache file
            df.to_csv(self.cache_path)
            self.data_cache[symbol] = df
            return df

        except requests.RequestException as err:
            print(f"[Alpha Vantage] Request failed ({err}). No data available.")
            return None

    from datetime import timedelta
    import pandas as pd

    def get_SP500_index_at_time(self, symbol: str, timestamp: pd.Timestamp):
        """
        Return the S&P 500 (or any market-index ETF) close price nearest to `timestamp`.

        • First look for the exact trading day.
        • If the market was closed, step backwards until the most recent
          available bar.
        • Falls back to `None` (and a message) when no data exist before
          the requested date.

        Parameters
        ----------
        symbol : str
            Ticker, e.g. "SPY".
        timestamp : pd.Timestamp
            Datetime you want the index value for (timezone-naïve OK).

        Returns
        -------
        float | None
        """
        df = self.fetch_market_index_SP500(symbol)
        if df is None or df.empty:
            print(f"[{symbol}] no data available.")
            return None

        # Ensure a proper DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Normalise both sides to midnight → compare dates not intraday times
        target = pd.to_datetime(timestamp).normalize()

        if target in df.index:
            return df.at[target, "4. close"]

        # Step back until we hit the first date that exists in the DataFrame
        earlier = df.index[df.index < target]
        if len(earlier):
            return df.loc[earlier[-1], "4. close"]

        print(f"[{symbol}] no historical bars prior to {target.date()}.")
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
