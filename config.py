DEFAULT_THRESHOLDS = {
    '30m': 0.01,
    '1h': 0.012,
    '4h': 0.020,
    '1d': 0.030,
    '1w': 0.040,
}

DEFAULT_DELTAS = {
    '30m': 0.100,
    '1h': 0.130,
    '4h': 0.180,
    '1d': 0.100,
    '1w': 0.100,
}

INTERVAL_PARAMETERS = {
    # [atr_period, n_candles, cd_retrace, atr_multiplier, risk_reward, max_holding_period]
    "1w": [6, 1, 0.216, 0.4573332, 2.98, 3],
    "1d": [28, 2, 0.33, 1.54, 2.76, 13],
    "4h": [9, 2, 0.349, 1.1, 2.764, 60],
    "1h": [17, 29, 0.112, 1.303, 1.017, 28],
    "30m": [13, 19, 0.282, 1.400, 1.330, 23],
}

INTERESTING_COLUMNS = [
    'entry_price', 'exit_reason','SL', 'TP', 'profit', 'Y', 'pattern_name', 'pattern_type',
    'symbol', 'interval','X_time', 'A_time', 'B_time', 'C_time', 'D_time',
    'X_price', 'A_price', 'B_price', 'C_price', 'D_price','entry_time',
]


YAHOO_SYMBOL_MAPPING={
    "ETHBTC": "ETH-BTC",
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "BNBUSDT": "BNB-USD",
    "SOLUSDT": "SOL-USD",
}
