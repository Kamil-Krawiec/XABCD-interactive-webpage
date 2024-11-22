import numpy as np
import pandas as pd


def directional_change(close: np.ndarray, high: np.ndarray, low: np.ndarray, sigma: float):
    """
    Identify the tops and bottoms based on a directional change approach.

    Parameters:
    close (np.ndarray): Array of closing prices.
    high (np.ndarray): Array of high prices.
    low (np.ndarray): Array of low prices.
    sigma (float): The percentage change threshold for determining a directional change.

    Returns:
    tops (List[List[int, int, float]]): List of tops identified with each element containing
                                        [confirmation index, extreme index, extreme price].
    bottoms (List[List[int, int, float]]): List of bottoms identified with each element containing
                                           [confirmation index, extreme index, extreme price].
    """

    up_zig = True  # Last extreme is a bottom. The next expected extreme is a top.
    tmp_max = high[0]
    tmp_min = low[0]
    tmp_max_i = 0
    tmp_min_i = 0

    tops = []
    bottoms = []

    for i in range(len(close)):
        if up_zig:  # Last extreme is a bottom
            if high[i] > tmp_max:
                # New high, update temporary maximum
                tmp_max = high[i]
                tmp_max_i = i
            elif close[i] < tmp_max * (1 - sigma):
                # Price retraced by sigma %. Confirm the top
                tops.append([i, tmp_max_i, tmp_max])

                # Setup for next bottom
                up_zig = False
                tmp_min = low[i]
                tmp_min_i = i
        else:  # Last extreme is a top
            if low[i] < tmp_min:
                # New low, update temporary minimum
                tmp_min = low[i]
                tmp_min_i = i
            elif close[i] > tmp_min * (1 + sigma):
                # Price retraced by sigma %. Confirm the bottom
                bottoms.append([i, tmp_min_i, tmp_min])

                # Setup for next top
                up_zig = True
                tmp_max = high[i]
                tmp_max_i = i

    return tops, bottoms


def get_extremes(ohlc: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """
    Extract the extreme points (tops and bottoms) from the OHLC data using directional changes.

    Parameters:
    ohlc (pd.DataFrame): DataFrame containing 'close', 'high', and 'low' price series.
    sigma (float): The percentage change threshold for determining a directional change.

    Returns:
    extremes (pd.DataFrame): DataFrame of confirmed extremes with columns:
                             'ext_i' (extreme index),
                             'ext_p' (extreme price),
                             'type' (1 for top, -1 for bottom).
    """

    tops, bottoms = directional_change(ohlc['close'].to_numpy(), ohlc['high'].to_numpy(), ohlc['low'].to_numpy(), sigma)

    tops_df = pd.DataFrame(tops, columns=['conf_i', 'ext_i', 'ext_p'])
    bottoms_df = pd.DataFrame(bottoms, columns=['conf_i', 'ext_i', 'ext_p'])

    tops_df['type'] = 1  # Indicate tops
    bottoms_df['type'] = -1  # Indicate bottoms

    extremes = pd.concat([tops_df, bottoms_df])
    extremes.set_index('conf_i', inplace=True)
    extremes.sort_index(inplace=True)

    return extremes
