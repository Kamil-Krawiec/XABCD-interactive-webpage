import pandas as pd
import logging
import numpy as np


class TradeAnalyzer:
    def __init__(
            self,
            atr_period=14,
            cd_retrace=0.183,
            atr_multiplier=1.0,
            risk_reward=2.0,
            transaction_cost=0.0011,
            n_candles=10,
            max_holding_period=100,
            slippage=0.0005,
    ):
        self.atr_period = atr_period
        self.cd_retrace = cd_retrace

        self.atr_multiplier = atr_multiplier
        self.risk_reward = risk_reward
        self.slippage = slippage

        self.transaction_cost = transaction_cost
        self.n_candles = n_candles
        self.max_holding_period = max_holding_period

    def calculate_atr(self, ohlc):
        """
        Calculate the Average True Range (ATR) for the OHLC data.
        """
        if 'ATR' not in ohlc.columns:
            high_low = ohlc['high'] - ohlc['low']
            high_close = (ohlc['high'] - ohlc['close'].shift()).abs()
            low_close = (ohlc['low'] - ohlc['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            ohlc['ATR'] = true_range.rolling(window=self.atr_period).mean()

    def set_trade_levels(self, pattern_row, ohlc):
        """
        Set TP and SL based on the original, rule-based approach.
        """
        D_time = pattern_row['D_time']
        D_price = pattern_row['D_price']
        C_price = pattern_row['C_price']
        pattern_type = pattern_row['pattern_type']

        CD = abs(D_price - C_price)

        trade_entry_level = (
            D_price + self.cd_retrace * CD
            if pattern_type.lower() == "bullish"
            else D_price - self.cd_retrace * CD
        )
        ohlc_copy = ohlc.copy()
        ohlc_copy.set_index('open_time', inplace=True)
        # Include one extra candle for the next opening price
        future_ohlc = ohlc_copy.loc[ohlc_copy.index > D_time].head(self.n_candles + 1)

        future_times = future_ohlc.index.tolist()

        # Iterate to find when close price crosses trade_entry_level
        for idx in range(len(future_ohlc) - 1):  # Adjusted to prevent out-of-range
            row = future_ohlc.iloc[idx]

            if pattern_type.lower() == "bullish" and row['close'] >= trade_entry_level:
                # Proceed to the next candle
                next_idx = idx + 1
                if next_idx >= len(future_ohlc):
                    break  # Prevent out-of-range
                next_time = future_times[next_idx]
                next_row = future_ohlc.iloc[next_idx]
                entry_time = next_time
                entry_price = next_row['open']  # Use next opening price
                atr_at_entry = next_row['ATR']
                trade_levels = self.calculate_trade_levels(entry_price, atr_at_entry, True)
                trade_levels['entry_time'] = entry_time
                return trade_levels

            elif pattern_type.lower() == "bearish" and row['close'] <= trade_entry_level:
                # Proceed to the next candle
                next_idx = idx + 1
                if next_idx >= len(future_ohlc):
                    break  # Prevent out-of-range
                next_time = future_times[next_idx]
                next_row = future_ohlc.iloc[next_idx]
                entry_time = next_time
                entry_price = next_row['open']  # Use next opening price
                atr_at_entry = next_row['ATR']
                trade_levels = self.calculate_trade_levels(entry_price, atr_at_entry, False)
                trade_levels['entry_time'] = entry_time
                return trade_levels

        return None  # No trade triggered

    def calculate_trade_levels(self, entry_price, atr_at_entry, is_bullish):
        """
        Calculate TP and SL based on fixed multipliers (original approach).
        """
        # Adjust entry price for transaction cost
        entry_price_adjusted = entry_price * (1 + self.transaction_cost)

        # Set risk amount (SL amount)
        sl_amount = self.atr_multiplier * atr_at_entry

        # Ensure SL amount does not exceed 100% of entry price
        max_sl_amount = entry_price_adjusted  # Maximum SL amount is 100% of entry price
        sl_amount = min(sl_amount, max_sl_amount)

        if is_bullish:
            stop_loss_level = max(entry_price_adjusted - sl_amount, 0)
            risk_amount = entry_price_adjusted - stop_loss_level
            take_profit_level = entry_price_adjusted + self.risk_reward * risk_amount
        else:
            stop_loss_level = entry_price_adjusted + sl_amount
            risk_amount = stop_loss_level - entry_price_adjusted
            take_profit_level = max(entry_price_adjusted - self.risk_reward * risk_amount, 0)

        trade_levels = {
            'entry_price': entry_price_adjusted,
            'SL': stop_loss_level,
            'TP': take_profit_level,
        }

        return trade_levels
