import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from classes.XABCD_classes import ValidPatterns, XABCDPatternFound
from classes.request_manager_class import RequestManager
from functions.pattern_error_and_ratios import calculate_harmonic_ratios, calculate_error
from classes.technical_indicators_class import TechnicalIndicators


class PatternManager:

    def __init__(self, ohlc: pd.DataFrame, extremes: pd.DataFrame, err_thresh: float = 0.2,
                 api_key_alpha_vantage: str = None, symbol='BTCUSDT', interval='1d'):
        self.ohlc = ohlc
        self.extremes = extremes
        self.err_thresh = err_thresh
        self.symbol = symbol
        self.interval = interval
        # Define Patterns
        self.pattern_df = None
        self.pattern_list = []
        self.ALL_PATTERNS = ValidPatterns.ALL_PATTERNS
        self.request_extractor = RequestManager(api_key_alpha_vantage)
        self.tech_indicators = TechnicalIndicators(ohlc)

    def find_xabcd_patterns(self):

        # Calculate segment heights and retracement ratios for extremes
        self.extremes['seg_height'] = self.extremes['ext_p'].diff().abs()
        self.extremes['retrace_ratio'] = self.extremes['seg_height'] / self.extremes['seg_height'].shift(1)

        output = []
        extreme_i = 0
        first_conf_i = self.extremes.index[0]

        for i in range(first_conf_i, len(self.ohlc)):

            # Update current extreme if a new extreme is reached
            if self.extremes.index[extreme_i + 1] == i:
                extreme_i += 1

            # Stop if we reach the last extreme
            if extreme_i + 1 >= len(self.extremes):
                break

            # Continue only if we have at least 3 prior extremes (X, A, B, C)
            if extreme_i < 3:
                continue

            current_extreme = self.extremes.iloc[extreme_i]
            last_conf_i = self.extremes.index[extreme_i]
            ext_type = current_extreme['type']

            # Determine D_price based on whether we're checking for a bull or bear pattern
            D_price = self.ohlc.iloc[i]['low'] if ext_type > 0 else self.ohlc.iloc[i]['high']

            # Ensure D_price is valid
            ohlc_slice = self.ohlc.iloc[last_conf_i:i]
            if (ext_type > 0 and ohlc_slice['low'].min() < D_price) or (
                    ext_type <= 0 and ohlc_slice['high'].max() > D_price):
                continue

            # Calculate retracement ratios for D point
            prev_extreme_1 = self.extremes.iloc[extreme_i - 1]
            prev_extreme_2 = self.extremes.iloc[extreme_i - 2]
            dc_retrace = abs(D_price - current_extreme['ext_p']) / current_extreme['seg_height']
            xa_ad_retrace = abs(D_price - prev_extreme_2['ext_p']) / prev_extreme_2['seg_height']

            # Find the best pattern match
            best_err, best_pat = min(
                ((sum([
                    calculate_error(current_extreme['retrace_ratio'], pat.AB_BC),
                    calculate_error(prev_extreme_1['retrace_ratio'], pat.XA_AB),
                    calculate_error(dc_retrace, pat.BC_CD),
                    calculate_error(xa_ad_retrace, pat.XA_AD)
                ]), pat.name) for pat in self.ALL_PATTERNS),
                key=lambda x: x[0]
            )

            # If the error is within the threshold, add the identified pattern
            if best_err <= self.err_thresh:
                pattern_data = XABCDPatternFound(
                    int(self.extremes.iloc[extreme_i - 3]['ext_i']),
                    int(self.extremes.iloc[extreme_i - 2]['ext_i']),
                    int(self.extremes.iloc[extreme_i - 1]['ext_i']),
                    int(self.extremes.iloc[extreme_i]['ext_i']),
                    i,
                    best_err,
                    best_pat,
                    ext_type > 0
                )

                pattern_data.name = pattern_data.name
                pattern_data.bull = ext_type > 0

                output.append(pattern_data)
        self.pattern_list = output

        return self.pattern_list

    def plot_xabcd_patterns(self, window=20, save=False):
        if self.pattern_df is None:
            self.patterns_to_dataframe()

        for idx, pattern in self.pattern_df.iterrows():
            # Determine the start and end index for the window
            start_index = max(pattern['pattern_start_index'] - window, 0)
            end_index = min(pattern['pattern_end_index'] + window, len(self.ohlc) - 1)

            # Slice the data to focus on the relevant range
            data_segment = self.ohlc.iloc[start_index:end_index]
            data_for_plot = data_segment.set_index('open_time')

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(14, 7))

            # Plot the candlestick chart
            mpf.plot(data_for_plot, type='candle', ax=ax, style='charles', show_nontrading=True)

            # Overlay the XABCD pattern
            x_points = [
                pattern['X_time'],
                pattern['A_time'],
                pattern['B_time'],
                pattern['C_time'],
                pattern['D_time']
            ]

            y_points = [
                pattern['X_price'],
                pattern['A_price'],
                pattern['B_price'],
                pattern['C_price'],
                pattern['D_price']
            ]

            ax.plot(x_points, y_points, marker='o', linestyle='-', color='r')

            # Annotate the points
            ax.text(x_points[0], y_points[0], 'X', color='r', fontsize=12, weight='bold')
            ax.text(x_points[1], y_points[1], 'A', color='g', fontsize=12, weight='bold')
            ax.text(x_points[2], y_points[2], 'B', color='b', fontsize=12, weight='bold')
            ax.text(x_points[3], y_points[3], 'C', color='m', fontsize=12, weight='bold')
            ax.text(x_points[4], y_points[4], 'D', color='y', fontsize=12, weight='bold')

            # Prepare the text box content with ratios
            textstr = '\n'.join((
                f'AB/XA Ratio: {pattern["ratio_AB_XA"]:.2f}',
                f'BC/AB Ratio: {pattern["ratio_BC_AB"]:.2f}',
                f'CD/BC Ratio: {pattern["ratio_CD_BC"]:.2f}',
                f'AD/XA Ratio: {pattern["ratio_AD_XA"]:.2f}'
            ))

            # Add a text box with the ratios
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

            # Set the title
            plt.title(f"XABCD Pattern {pattern['pattern_name']}_{idx + 1} symbol {self.symbol}")

            # Format the dates to make them more readable
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)  # Rotate X-axis labels for better visibility

            # Optionally, set the frequency of date ticks
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            # Adjust tick parameters for better visibility
            ax.tick_params(axis='x', which='major', labelsize=10)

            # Save the plot if requested
            if save:
                plt.savefig(f"charts/XABCD_{pattern['pattern_name']}_{idx + 1}.png")
            plt.show()

    def patterns_to_dataframe(self) -> pd.DataFrame:

        if self.pattern_list is None:
            raise ValueError("No patterns found. Run find_xabcd_patterns() first.")

        pattern_dicts = []

        # Loop through each pattern and extract the information using the helper function
        for pattern in self.pattern_list:
            pattern_info = self.extract_pattern_info(pattern)
            pattern_dicts.append(pattern_info)

        # Convert the list of pattern dictionaries into a DataFrame
        self.pattern_df = pd.DataFrame(pattern_dicts)
        self.pattern_df['symbol'] = self.symbol
        self.pattern_df['interval'] = self.interval
        return self.pattern_df

    def extract_pattern_info(self, pattern: XABCDPatternFound) -> dict:
        try:
            X_idx, A_idx, B_idx, C_idx, D_idx = pattern.X, pattern.A, pattern.B, pattern.C, pattern.D

            # Get prices at points X, A, B, C, D based on pattern type
            X = self.ohlc['low'][X_idx] if pattern.bull else self.ohlc['high'][X_idx]
            A = self.ohlc['high'][A_idx] if pattern.bull else self.ohlc['low'][A_idx]
            B = self.ohlc['low'][B_idx] if pattern.bull else self.ohlc['high'][B_idx]
            C = self.ohlc['high'][C_idx] if pattern.bull else self.ohlc['low'][C_idx]
            D = self.ohlc['low'][D_idx] if pattern.bull else self.ohlc['high'][D_idx]

            # Calculate harmonic ratios
            ratio_AB_XA, ratio_BC_AB, ratio_CD_BC, ratio_AD_XA = calculate_harmonic_ratios(X, A, B, C, D)

            # Determine the pattern type
            pattern_type = "Bullish" if pattern.bull else "Bearish"

            # Extract technical indicators at points X, A, B, C, D
            indicators = ['RSI_14', 'MACD', 'MACD_signal', 'MACD_hist', 'stoch_k', 'stoch_d', 'ADX', 'CCI',
                          'OBV', 'EMA_14', 'BB_bandwidth', 'ATR', 'PSAR', 'Volume_MA_20', 'Volume_Oscillator',
                          'Ichimoku_Tenkan_Sen', 'Ichimoku_Kijun_Sen', 'Ichimoku_Senkou_Span_A',
                          'Ichimoku_Senkou_Span_B', 'Ichimoku_Chikou_Span']

            # Initialize dictionaries to store indicators at each point
            indicator_values = {indicator: {} for indicator in indicators}

            # Collect indicators at each point
            for idx_label, idx in zip(['X', 'A', 'B', 'C', 'D'], [X_idx, A_idx, B_idx, C_idx, D_idx]):
                for indicator in indicators:
                    indicator_value = self.tech_indicators.get_indicator(indicator, idx)
                    indicator_values[indicator][f'{indicator}_at_{idx_label}'] = indicator_value


            # Compute angles between points
            angle_XA = self.tech_indicators.calculate_angle(X_idx, A_idx)
            angle_AB = self.tech_indicators.calculate_angle(A_idx, B_idx)
            angle_BC = self.tech_indicators.calculate_angle(B_idx, C_idx)
            angle_CD = self.tech_indicators.calculate_angle(C_idx, D_idx)


            # Extract time and duration information
            pattern_info = {
                'pattern_name': pattern.name,
                'pattern_type': pattern_type,
                'pattern_start_time': self.ohlc['open_time'][X_idx],
                'pattern_end_time': self.ohlc['open_time'][D_idx],
                'X_time': self.ohlc['open_time'][X_idx],
                'X_price': X,
                'A_time': self.ohlc['open_time'][A_idx],
                'A_price': A,
                'B_time': self.ohlc['open_time'][B_idx],
                'B_price': B,
                'C_time': self.ohlc['open_time'][C_idx],
                'C_price': C,
                'D_time': self.ohlc['open_time'][D_idx],
                'D_price': D,
                'ratio_AB_XA': ratio_AB_XA,
                'ratio_BC_AB': ratio_BC_AB,
                'ratio_CD_BC': ratio_CD_BC,
                'ratio_AD_XA': ratio_AD_XA,
                'pattern_duration': (self.ohlc['close_time'][D_idx] - self.ohlc['open_time'][X_idx]).total_seconds() / 60,
                'duration_XA': (self.ohlc['open_time'][A_idx] - self.ohlc['open_time'][X_idx]).total_seconds() / 60,
                'duration_AB': (self.ohlc['open_time'][B_idx] - self.ohlc['open_time'][A_idx]).total_seconds() / 60,
                'duration_BC': (self.ohlc['open_time'][C_idx] - self.ohlc['open_time'][B_idx]).total_seconds() / 60,
                'duration_CD': (self.ohlc['open_time'][D_idx] - self.ohlc['open_time'][C_idx]).total_seconds() / 60,
                'open_price_at_D': self.ohlc['open'][D_idx],
                'close_price_at_D': self.ohlc['close'][D_idx],
                'volume_at_D': self.ohlc['volume'][D_idx],
                'quote_asset_volume_at_D': self.ohlc['quote_asset_volume'][D_idx],
                'number_of_trades_at_D': self.ohlc['number_of_trades'][D_idx],
                'taker_buy_base_asset_volume_at_D': self.ohlc['taker_buy_base_asset_volume'][D_idx],
                'taker_buy_quote_asset_volume_at_D': self.ohlc['taker_buy_quote_asset_volume'][D_idx],
                # Angles between points
                'angle_XA': angle_XA,
                'angle_AB': angle_AB,
                'angle_BC': angle_BC,
                'angle_CD': angle_CD,
                # Additional external data
                'fear_greed_index_at_D': self.request_extractor.get_fear_greed_index_at_time(
                    self.ohlc['close_time'][D_idx]),
                'SP500_at_D': self.request_extractor.get_SP500_index_at_time('SPY', self.ohlc['close_time'][D_idx]),
            }

            # Collect Bollinger Bands at each point
            for idx_label, idx in zip(['X', 'A', 'B', 'C', 'D'], [X_idx, A_idx, B_idx, C_idx, D_idx]):
                pattern_info[f'BB_upper_at_{idx_label}'] = self.tech_indicators.get_indicator('BB_upper', idx)
                pattern_info[f'BB_middle_at_{idx_label}'] = self.tech_indicators.get_indicator('BB_middle', idx)
                pattern_info[f'BB_lower_at_{idx_label}'] = self.tech_indicators.get_indicator('BB_lower', idx)


            # Merge indicator values into pattern_info
            for indicator in indicators:
                pattern_info.update(indicator_values[indicator])

            return pattern_info
        except Exception as e:
            print(e)
            return {}
