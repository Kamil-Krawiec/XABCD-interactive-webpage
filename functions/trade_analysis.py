# trade_analysis.py

import logging
import warnings
import pandas as pd
import numpy as np

from config import INTERVAL_PARAMETERS
from classes import TradeAnalyzer
from classes import TradeEvaluator

warnings.filterwarnings('ignore')

# Configure logging (optional, you can adjust or remove it)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def perform_trade_analysis( patterns_df, ohlc_data_dict):
    """
    Performs trade analysis on the given patterns DataFrame.

    Parameters:
    - client: Binance Client instance.
    - patterns_df: DataFrame containing detected patterns.
    - ohlc_data_dict: Dictionary containing OHLC data for symbols and intervals.

    Returns:
    - DataFrame with trade analysis results.
    """
    # Dictionary to store the results for each interval
    interval_results = []

    # Group by interval and process each group separately
    for interval, params in INTERVAL_PARAMETERS.items():
        # Filter the patterns for the current interval
        interval_df = patterns_df[patterns_df['interval'] == interval]

        # Check if there are patterns for the interval
        if interval_df.empty:
            logging.warning(f"No patterns found for interval: {interval}. Skipping.")
            continue

        logging.info(f"Processing interval: {interval} with parameters: {params}")

        # Use the passed ohlc_data_dict instead of fetching data
        ohlc_dict = {}
        symbols = interval_df['symbol'].unique()
        for symbol in symbols:
            key = (symbol, interval)
            if key in ohlc_data_dict:
                ohlc_data = ohlc_data_dict[key]
                ohlc_dict[key] = ohlc_data
                logging.info(f"Using stored OHLC data for symbol: {symbol}, interval: {interval}.")
            else:
                logging.warning(f"OHLC data for symbol: {symbol}, interval: {interval} not found. Skipping.")
                continue

        if not ohlc_dict:
            logging.warning(f"No OHLC data available for interval: {interval}. Skipping.")
            continue

        # Extract parameters
        atr_period = int(params[0])
        n_candles = int(params[1])
        cd_retrace = float(params[2])
        atr_multiplier = float(params[3])
        risk_reward = float(params[4])
        max_holding_period = int(params[5])

        # Initialize the TradeAnalyzer and TradeEvaluator with the interval-specific parameters
        try:
            trade_analyzer = TradeAnalyzer(
                atr_period=atr_period,
                cd_retrace=cd_retrace,
                atr_multiplier=atr_multiplier,
                risk_reward=risk_reward,
                transaction_cost=0.0011,  # Assuming fixed transaction cost
                n_candles=n_candles
            )
            logging.info(f"Initialized TradeAnalyzer for interval: {interval}.")
        except Exception as e:
            logging.error(f"Error initializing TradeAnalyzer for interval: {interval}: {e}")
            continue

        try:
            trade_evaluator = TradeEvaluator(
                max_holding_period=max_holding_period,
                transaction_cost=0.0011  # Assuming fixed transaction cost
            )
            logging.info(f"Initialized TradeEvaluator for interval: {interval}.")
        except Exception as e:
            logging.error(f"Error initializing TradeEvaluator for interval: {interval}: {e}")
            continue

        # Initialize 'Y' column and other trade detail columns
        interval_df = interval_df.copy()
        interval_df['Y'] = 0  # Default value
        interval_df['entry_time'] = pd.NaT
        interval_df['entry_price'] = np.nan
        interval_df['SL'] = np.nan
        interval_df['TP'] = np.nan
        interval_df['profit'] = 0.0  # Default profit
        interval_df['exit_reason'] = ''  # To store exit reason

        # Iterate over each pattern in interval_df
        for idx, pattern in interval_df.iterrows():
            try:
                symbol = pattern['symbol']
                key = (symbol, interval)
                if key not in ohlc_dict:
                    logging.warning(
                        f"OHLC data for symbol: {symbol}, interval: {interval} not found. Skipping pattern at index {idx}."
                    )
                    continue

                ohlc_data = ohlc_dict[key].copy()

                # Calculate ATR
                trade_analyzer.calculate_atr(ohlc_data)

                # Set trade levels using the original (rule-based) approach
                trade_levels = trade_analyzer.set_trade_levels(pattern, ohlc_data)

                if trade_levels is not None:
                    # Evaluate trade using TradeEvaluator
                    profit_percent, outcome, exit_reason = trade_evaluator.evaluate_trade(
                        trade_levels, ohlc_data, is_bullish=pattern['pattern_type'].lower() == 'bullish'
                    )

                    # Assign 'Y' based on profit
                    interval_df.at[idx, 'Y'] = 1 if profit_percent > 0 else 0

                    # Store trade details
                    interval_df.at[idx, 'profit'] = profit_percent
                    interval_df.at[idx, 'entry_time'] = trade_levels.get('entry_time', pd.NaT)
                    interval_df.at[idx, 'entry_price'] = trade_levels.get('entry_price', np.nan)
                    interval_df.at[idx, 'SL'] = trade_levels.get('SL', np.nan)
                    interval_df.at[idx, 'TP'] = trade_levels.get('TP', np.nan)
                    interval_df.at[idx, 'exit_reason'] = exit_reason
                else:
                    # No trade triggered, 'Y' remains 0 and other fields stay default
                    interval_df.at[idx, 'profit'] = 0.0
                    interval_df.at[idx, 'exit_reason'] = 'No trade triggered'

            except Exception as e:
                logging.error(f"Error processing pattern at index {idx}: {e}")
                continue

        # Append the evaluated patterns to the results list
        interval_results.append(interval_df)

        logging.info(f"Completed processing interval: {interval}.")

    if not interval_results:
        logging.error("No interval results to process.")
        return pd.DataFrame()

    # Combine all results into a single DataFrame
    combined_results = pd.concat(interval_results, ignore_index=True)
    logging.info("Combined all interval results into a single DataFrame.")

    return combined_results