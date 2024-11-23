import pandas as pd
import numpy as np
import logging


class TradeEvaluator:
    def __init__(
            self,
            max_holding_period=10,
            transaction_cost=0.0011,
            slippage=0.0005,  # 0.05% slippage
    ):
        self.max_holding_period = max_holding_period
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def evaluate_trade(self, trade_levels, ohlc, is_bullish):
        """
        Evaluates a trade to calculate profit based on TP, SL, slippage, and transaction costs.

        Args:
            trade_levels (dict): Dictionary containing trade details such as entry_time, entry_price, SL, TP.
            ohlc (pd.DataFrame): DataFrame containing OHLC data.
            is_bullish (bool): Indicates if the trade is bullish.

        Returns:
            tuple: (profit_percent (float), outcome (int), exit_reason (str))
        """
        entry_time = trade_levels['entry_time']
        entry_price = trade_levels['entry_price']
        SL = trade_levels['SL']
        TP = trade_levels['TP']

        ohlc_copy = ohlc.copy()
        ohlc_copy.set_index('open_time', inplace=True)

        # Get future ohlc data starting after entry_time
        future_ohlc = ohlc_copy.loc[ohlc_copy.index > entry_time].head(self.max_holding_period)

        if future_ohlc.empty:
            logging.warning(f"No future OHLC data available after {entry_time}.")
            # Exit due to timeout
            last_close = ohlc_copy['close'].iloc[-1]
            if is_bullish:
                exit_price = last_close * (1 - self.slippage) * (1 - self.transaction_cost)
                profit_percent = (exit_price - entry_price) / entry_price
            else:
                exit_price = last_close * (1 + self.slippage) * (1 + self.transaction_cost)
                profit_percent = (entry_price - exit_price) / entry_price
            outcome = 1 if profit_percent > 0 else 0
            exit_reason = 'Timeout'
            return profit_percent, outcome, exit_reason

        for time, row in future_ohlc.iterrows():
            current_high = row['high']
            current_low = row['low']

            if is_bullish:
                # Check SL first
                if current_low <= SL:
                    # Adjust for slippage and transaction cost
                    exit_price = SL * (1 - self.slippage)
                    exit_price *= (1 - self.transaction_cost)
                    profit_percent = (exit_price - entry_price) / entry_price
                    outcome = 1 if profit_percent > 0 else 0
                    exit_reason = 'SL'
                    return profit_percent, outcome, exit_reason
                # Check TP
                if current_high >= TP:
                    # Adjust for slippage and transaction cost
                    exit_price = TP * (1 + self.slippage)
                    exit_price *= (1 - self.transaction_cost)
                    profit_percent = (exit_price - entry_price) / entry_price
                    outcome = 1 if profit_percent > 0 else 0
                    exit_reason = 'TP'
                    return profit_percent, outcome, exit_reason
            else:
                # Check SL first
                if current_high >= SL:
                    # Adjust for slippage and transaction cost
                    exit_price = SL * (1 + self.slippage)
                    exit_price *= (1 + self.transaction_cost)
                    profit_percent = (entry_price - exit_price) / entry_price
                    outcome = 1 if profit_percent > 0 else 0
                    exit_reason = 'SL'
                    return profit_percent, outcome, exit_reason
                # Check TP
                if current_low <= TP:
                    # Adjust for slippage and transaction cost
                    exit_price = TP * (1 - self.slippage)
                    exit_price *= (1 + self.transaction_cost)
                    profit_percent = (entry_price - exit_price) / entry_price
                    outcome = 1 if profit_percent > 0 else 0
                    exit_reason = 'TP'
                    return profit_percent, outcome, exit_reason

        # If neither SL nor TP hit, exit at last close price with slippage and transaction cost
        last_close = future_ohlc.iloc[-1]['close']
        if is_bullish:
            exit_price = last_close * (1 - self.slippage) * (1 - self.transaction_cost)
            profit_percent = (exit_price - entry_price) / entry_price
        else:
            exit_price = last_close * (1 + self.slippage) * (1 + self.transaction_cost)
            profit_percent = (entry_price - exit_price) / entry_price

        outcome = 1 if profit_percent > 0 else 0
        exit_reason = 'Timeout'

        return profit_percent, outcome, exit_reason
