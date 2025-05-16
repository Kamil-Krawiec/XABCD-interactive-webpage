import pandas as pd

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
    "1w": [8, 1, 0.22875419, 0.45872564, 2.77202136, 1],
    "4h": [12, 5, 0.33349214, 1.08311109, 1.57182004, 25],
    "1d": [23, 2, 0.33739758, 3.06114478, 2.73937681, 13],
    "1h": [17, 30, 0.11269095, 1.30360779, 1.01760487, 28],
    "30m": [14, 19, 0.28214013, 1.40095309, 1.33092255, 23],
}

INTERESTING_COLUMNS = [
    'entry_price', 'exit_reason','SL', 'TP', 'profit', 'Y', 'pattern_name', 'pattern_type',
    'symbol', 'interval','X_time', 'A_time', 'B_time', 'C_time', 'D_time',
    'X_price', 'A_price', 'B_price', 'C_price', 'D_price','entry_time',
]



MAX_REQUESTS_DAYS={
    '30m': 60,
    '1h': 730,
}

TOP_40_FEATURES = pd.read_csv("data/feature_importance.csv").sort_values("Score", ascending=False).head(40)["Feature name"].tolist()

ALL_FEATURES = pd.read_csv("data/feature_importance.csv").sort_values("Score", ascending=False)["Feature name"].tolist()
