import numpy as np
import pandas as pd
import ta  # Technical Analysis library


class TechnicalIndicators:
    def __init__(self, ohlc: pd.DataFrame):
        self.ohlc = ohlc.copy()
        self.calculate_all_indicators()

    def calculate_all_indicators(self):
        close = self.ohlc['close']
        high = self.ohlc['high']
        low = self.ohlc['low']
        volume = self.ohlc['volume']

        # Calculate RSI
        self.ohlc['RSI_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        # Calculate MACD
        macd_indicator = ta.trend.MACD(close)
        self.ohlc['MACD'] = macd_indicator.macd()
        self.ohlc['MACD_signal'] = macd_indicator.macd_signal()
        self.ohlc['MACD_hist'] = macd_indicator.macd_diff()
        # Calculate Stochastic Oscillator
        stoch_indicator = ta.momentum.StochasticOscillator(high, low, close)
        self.ohlc['stoch_k'] = stoch_indicator.stoch()
        self.ohlc['stoch_d'] = stoch_indicator.stoch_signal()
        # Calculate ADX
        adx_indicator = ta.trend.ADXIndicator(high, low, close)
        self.ohlc['ADX'] = adx_indicator.adx()
        # Calculate CCI
        cci_indicator = ta.trend.CCIIndicator(high, low, close)
        self.ohlc['CCI'] = cci_indicator.cci()
        # Calculate OBV
        self.ohlc['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        # Calculate EMA
        self.ohlc['EMA_14'] = close.ewm(span=14, adjust=False).mean()
        # Calculate Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(close)
        self.ohlc['BB_upper'] = bb_indicator.bollinger_hband()
        self.ohlc['BB_middle'] = bb_indicator.bollinger_mavg()
        self.ohlc['BB_lower'] = bb_indicator.bollinger_lband()
        self.ohlc['BB_bandwidth'] = bb_indicator.bollinger_wband()
        # Calculate ATR
        atr_indicator = ta.volatility.AverageTrueRange(high, low, close)
        self.ohlc['ATR'] = atr_indicator.average_true_range()
        # Calculate Parabolic SAR
        psar_indicator = ta.trend.PSARIndicator(high, low, close)
        self.ohlc['PSAR'] = psar_indicator.psar()
        # Calculate Volume MA
        self.ohlc['Volume_MA_20'] = volume.rolling(window=20).mean()
        # Calculate Volume Oscillator
        self.ohlc['Volume_MA_short'] = volume.rolling(window=5).mean()
        self.ohlc['Volume_MA_long'] = volume.rolling(window=20).mean()
        self.ohlc['Volume_Oscillator'] = ((self.ohlc['Volume_MA_short'] - self.ohlc['Volume_MA_long']) / self.ohlc[
            'Volume_MA_long']) * 100
        # Calculate Ichimoku Cloud components
        ichimoku_indicator = ta.trend.IchimokuIndicator(high, low)
        self.ohlc['Ichimoku_Tenkan_Sen'] = ichimoku_indicator.ichimoku_conversion_line()
        self.ohlc['Ichimoku_Kijun_Sen'] = ichimoku_indicator.ichimoku_base_line()
        self.ohlc['Ichimoku_Senkou_Span_A'] = ichimoku_indicator.ichimoku_a()
        self.ohlc['Ichimoku_Senkou_Span_B'] = ichimoku_indicator.ichimoku_b()
        self.ohlc['Ichimoku_Chikou_Span'] = close.shift(-26)

    def get_indicator(self, name, idx):
        return self.ohlc[name].iloc[idx]

    def calculate_RSI(self, idx, period=14):
        close = self.ohlc['close']
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=period - 1, adjust=False).mean()
        ema_down = down.ewm(com=period - 1, adjust=False).mean()
        rs = ema_up / ema_down
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[idx]

    def calculate_MACD(self, idx, fast_period=12, slow_period=26, signal_period=9):
        close = self.ohlc['close']
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd - signal
        return macd.iloc[idx], signal.iloc[idx], macd_hist.iloc[idx]

    def calculate_bollinger_band_width(self, idx, period=20, n_std=2):
        close = self.ohlc['close']
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper_band = sma + n_std * std
        lower_band = sma - n_std * std
        bandwidth = (upper_band - lower_band) / sma
        return bandwidth.iloc[idx]

    def calculate_bollinger_bands(self, idx: int, timeperiod: int = 20, nbdevup: float = 2,
                                  nbdevdn: float = 2):
        """
        Calculate the Bollinger Bands (upper, middle, and lower) for the given index.

        Args:
        - data: DataFrame containing OHLCV data.
        - idx: Index for which to calculate the Bollinger Bands.
        - timeperiod: Period for the moving average (default is 20).
        - nbdevup: Number of standard deviations for the upper band (default is 2).
        - nbdevdn: Number of standard deviations for the lower band (default is 2).

        Returns:
        - A tuple (upper_band, middle_band, lower_band) at the given index.
        """
        close_prices = self.ohlc['close'].values

        # Calculate all three bands using TA-Lib's BBANDS function
        upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=timeperiod,
                                                        nbdevup=nbdevup, nbdevdn=nbdevdn)

        # Return the values at the specified index
        return upperband[idx], middleband[idx], lowerband[idx]

    def calculate_angle(self, idx1, idx2):
        x_diff = (self.ohlc['open_time'][idx2] - self.ohlc['open_time'][idx1]).total_seconds()
        y_diff = self.ohlc['close'][idx2] - self.ohlc['close'][idx1]
        angle = np.degrees(np.arctan2(y_diff, x_diff))
        return angle

    def calculate_volume_oscillator(self, idx, short_period=5, long_period=20):
        volume = self.ohlc['volume']
        short_ma = volume.rolling(window=short_period).mean()
        long_ma = volume.rolling(window=long_period).mean()
        vol_osc = ((short_ma - long_ma) / long_ma) * 100
        return vol_osc.iloc[idx]

