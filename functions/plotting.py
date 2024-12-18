import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd


def plot_xabcd_pattern(pattern, ohlc, save_plots=False, save_dir='charts', candles_left=50, candles_right=50):
    """
    Plots a single XABCD pattern on an OHLC candlestick chart without Stop-Loss (SL) and Take-Profit (TP) levels.

    Parameters:
    - pattern (dict): Dictionary containing pattern details.
    - ohlc (pd.DataFrame): OHLC DataFrame indexed by datetime.
    - save_plots (bool): If True, saves the plot as a JPEG file in the specified directory.
    - save_dir (str): Directory path to save the plot if save_plots is True.
    - candles_left (int): Number of candles to display before the pattern starts.
    - candles_right (int): Number of candles to display after the pattern ends.

    Returns:
    - matplotlib.figure.Figure: The generated plot figure.
    """
    # Create directory for saving plots if required
    if save_plots and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Helper function to find the nearest index
    def find_nearest_idx(dt_index, target_time):
        """Find the index of the nearest timestamp to target_time in dt_index."""
        dt_array = dt_index.to_numpy()
        target_np = np.datetime64(target_time)
        diff = np.abs(dt_array - target_np)
        nearest_idx = diff.argmin()
        return nearest_idx

    # Extract pattern details
    symbol = pattern['symbol']
    interval = pattern['interval']
    pattern_name = pattern['pattern_name']
    pattern_type = pattern.get('pattern_type', '')
    points = ['X', 'A', 'B', 'C', 'D']
    point_times = [pd.to_datetime(pattern[f'{point}_time']) for point in points]
    point_prices = [pattern[f'{point}_price'] for point in points]

    # Define the window: from pattern_start_time - candles_left to D_time + candles_right
    pattern_start_time = pd.to_datetime(pattern['pattern_start_time'])
    D_time = pd.to_datetime(pattern['D_time'])

    # Locate the start and end positions in the OHLC data
    try:
        pattern_start_idx = find_nearest_idx(ohlc.index, pattern_start_time)
    except Exception as e:
        print(f"Error finding pattern_start_time: {e}")
        return None
    try:
        D_idx = find_nearest_idx(ohlc.index, D_time)
    except Exception as e:
        print(f"Error finding D_time: {e}")
        return None

    # Calculate start and end indices with bounds checking
    start_idx = max(pattern_start_idx - candles_left, 0)
    end_idx = min(D_idx + candles_right, len(ohlc) - 1)

    # Slice the data to focus on the relevant range
    data_segment = ohlc.iloc[start_idx:end_idx + 1].copy()
    data_segment.sort_index(inplace=True)  # Ensure sorted by time

    # Prepare data for mplfinance
    data_for_plot = data_segment.copy()

    # Create the figure and axis using mplfinance
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the candlestick chart
    try:
        mpf.plot(data_for_plot, type='candle', ax=ax, style='charles', show_nontrading=True,
                 volume=False)
    except Exception as e:
        print(f"Error plotting candlestick: {e}")
        plt.close(fig)
        return None

    # Overlay the XABCD pattern
    try:
        ax.plot(point_times, point_prices, marker='o', linestyle='-', color='r', label='XABCD Pattern')
    except Exception as e:
        print(f"Error plotting XABCD pattern: {e}")

    # Annotate the points
    try:
        colors = ['r', 'g', 'b', 'm', 'y']
        for point, time, price, color in zip(points, point_times, point_prices, colors):
            ax.annotate(point, xy=(time, price), xytext=(0, 10),
                        textcoords='offset points', color=color, fontsize=12, weight='bold',
                        horizontalalignment='center', transform=ax.transAxes)
    except Exception as e:
        print(f"Error annotating points: {e}")

    # Set the title
    plt.title(f"XABCD Pattern {pattern_name} for {symbol} ({interval})")

    # Format the dates to make them more readable
    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)  # Rotate X-axis labels for better visibility
        # Optionally, set the frequency of date ticks
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        # Adjust tick parameters for better visibility
        ax.tick_params(axis='x', which='major', labelsize=10)
    except Exception as e:
        print(f"Error formatting date axis: {e}")

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the plot if requested
    if save_plots:
        try:
            filename = f"XABCD_{pattern_name}_{symbol}_{interval}.jpg"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, format='jpg')
            print(f"Saved plot to {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    # Return the figure without showing it
    return fig


def plot_xabcd_patterns_with_sl_tp(pattern, ohlc, save_plots=False, save_dir='charts', candles_left=50,
                                   candles_right=50):
    """
    Plots a single ABCD pattern with Stop-Loss (SL) and Take-Profit (TP) levels on an OHLC candlestick chart.

    Parameters:
    - pattern (dict): Dictionary containing pattern details.
    - ohlc (pd.DataFrame): OHLC DataFrame indexed by datetime.
    - save_plots (bool): If True, saves the plot as a JPEG file in the specified directory.
    - save_dir (str): Directory path to save the plot if save_plots is True.

    Returns:
    - matplotlib.figure.Figure: The generated plot figure.
    """
    # Create directory for saving plots if required
    if save_plots and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Helper function to find the nearest index
    def find_nearest_idx(dt_index, target_time):
        """Find the index of the nearest timestamp to target_time in dt_index."""
        idx = dt_index.get_indexer([target_time], method='nearest')[0]
        return idx

    # Extract pattern details
    symbol = pattern['symbol']
    interval = pattern['interval']
    pattern_name = pattern['pattern_name']
    profit = pattern['profit']
    points = ['X', 'A', 'B', 'C', 'D']
    point_times = [pd.to_datetime(pattern[f'{point}_time']) for point in points]
    point_prices = [pattern[f'{point}_price'] for point in points]

    # Ensure 'ohlc' index is sorted and in datetime format
    ohlc = ohlc.sort_index()
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        ohlc.index = pd.to_datetime(ohlc.index)

    # Extract pattern times and check if they are within ohlc index range
    pattern_start_time = pd.to_datetime(pattern['pattern_start_time'])
    D_time = pd.to_datetime(pattern['D_time'])

    if pattern_start_time < ohlc.index.min() or D_time > ohlc.index.max():
        print("Pattern times are outside the OHLC data range.")
        return None

    # Locate the start and end positions in the OHLC data
    try:
        pattern_start_idx = find_nearest_idx(ohlc.index, pattern_start_time)
    except Exception as e:
        print(f"Error finding pattern_start_time index: {e}")
        return None
    try:
        D_idx = find_nearest_idx(ohlc.index, D_time)
    except Exception as e:
        print(f"Error finding D_time index: {e}")
        return None

    # Calculate start and end indices with bounds checking
    start_idx = max(pattern_start_idx - candles_left, 0)
    end_idx = min(D_idx + candles_right, len(ohlc) - 1)

    # Slice the data to focus on the relevant range
    data_segment = ohlc.iloc[start_idx:end_idx + 1].copy()
    data_segment.sort_index(inplace=True)  # Ensure sorted by time

    # Limit data segment size
    MAX_DATA_POINTS = 1000  # Adjust as needed
    if len(data_segment) > MAX_DATA_POINTS:
        data_segment = data_segment.iloc[-MAX_DATA_POINTS:]
        print(f"Data segment truncated to last {MAX_DATA_POINTS} data points.")

    # Prepare data for mplfinance
    data_for_plot = data_segment.copy()

    # Create the figure and axis using mplfinance
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the candlestick chart
    try:
        mpf.plot(data_for_plot, type='candle', ax=ax, style='charles', show_nontrading=True,
                 axtitle=f"{symbol} - {interval} ABCD Pattern: {pattern_name}", volume=False)
    except Exception as e:
        print(f"Error plotting candlestick: {e}")
        plt.close(fig)
        return None

    # Overlay the XABCD pattern
    try:
        ax.plot(point_times, point_prices, marker='o', linestyle='-', color='r', label='ABCD Pattern')
    except Exception as e:
        print(f"Error plotting ABCD pattern: {e}")

    # Annotate the points (Removed transform=ax.transAxes)
    try:
        colors = ['r', 'g', 'b', 'm', 'y']
        for point, time, price, color in zip(points, point_times, point_prices, colors):
            ax.annotate(point, xy=(time, price), xytext=(5, 5), textcoords='offset points',
                        color=color, fontsize=12, weight='bold')
    except Exception as e:
        print(f"Error annotating points: {e}")

    # Plot and annotate Stop Loss (SL), Take Profit (TP), and entry price if they are valid
    stop_loss_level = pattern.get('SL', None)
    take_profit_level = pattern.get('TP', None)
    entry_price = pattern.get('entry_price', None)
    entry_time = pd.to_datetime(pattern['entry_time']) if pd.notna(pattern.get('entry_time')) else None

    # Ensure the values are finite before plotting them
    if pd.notna(stop_loss_level) and np.isfinite(stop_loss_level):
        try:
            ax.axhline(y=stop_loss_level, color='red', linestyle='--', label='Stop Loss')
            if entry_time:
                ax.text(entry_time, stop_loss_level, 'SL', color='red', fontsize=12, weight='bold',
                        verticalalignment='bottom')
        except Exception as e:
            print(f"Error plotting Stop Loss: {e}")

    if pd.notna(take_profit_level) and np.isfinite(take_profit_level):
        try:
            ax.axhline(y=take_profit_level, color='green', linestyle='--', label='Take Profit')
            if entry_time:
                ax.text(entry_time, take_profit_level, 'TP', color='green', fontsize=12, weight='bold',
                        verticalalignment='top')
        except Exception as e:
            print(f"Error plotting Take Profit: {e}")

    if pd.notna(entry_price) and np.isfinite(entry_price):
        try:
            ax.axhline(y=entry_price, color='orange', linestyle='-.', label='Entry Price')
            if entry_time:
                # Draw a more noticeable arrow to indicate the entry point
                ax.annotate(
                    'Entry',
                    xy=(entry_time, entry_price),
                    xytext=(entry_time, entry_price + (data_segment['high'].max() - data_segment['low'].min()) * 0.05),
                    arrowprops=dict(
                        arrowstyle='->',
                        color='orange',
                        lw=2,
                        connectionstyle='arc3,rad=0.2',
                    ),
                    color='orange',
                    fontsize=12,
                    weight='bold'
                )
        except Exception as e:
            print(f"Error plotting Entry Price: {e}")

    # Calculate profit percentage
    profit_percentage = profit * 100
    if profit_percentage > 0:
        result_text = f"Profit: {profit_percentage:.2f}%"
        text_color = 'green'
    elif profit_percentage < 0:
        result_text = f"Loss: {profit_percentage:.2f}%"
        text_color = 'red'
    else:
        result_text = "No trade entry or no profit/loss"
        text_color = 'gray'

    # Prepare the text box content with ratios and profit/loss information
    ratio_AB_XA = pattern.get('ratio_AB_XA', np.nan)
    ratio_BC_AB = pattern.get('ratio_BC_AB', np.nan)
    ratio_CD_BC = pattern.get('ratio_CD_BC', np.nan)
    ratio_AD_XA = pattern.get('ratio_AD_XA', np.nan)
    textstr = '\n'.join(
        (
            f"AB/XA Ratio: {ratio_AB_XA:.2f}" if pd.notna(ratio_AB_XA) else "AB/XA Ratio: N/A",
            f"BC/AB Ratio: {ratio_BC_AB:.2f}" if pd.notna(ratio_BC_AB) else "BC/AB Ratio: N/A",
            f"CD/BC Ratio: {ratio_CD_BC:.2f}" if pd.notna(ratio_CD_BC) else "CD/BC Ratio: N/A",
            f"AD/XA Ratio: {ratio_AD_XA:.2f}" if pd.notna(ratio_AD_XA) else "AD/XA Ratio: N/A",
            result_text,
            f"Exit reason: {pattern.get('exit_reason', 'N/A')}"
        )
    )

    # Add a text box with the ratios and result
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    try:
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props,
            color=text_color,
        )
    except Exception as e:
        print(f"Error adding text box: {e}")

    # Set the title
    plt.title(f"XABCD Pattern {pattern_name} for {symbol} ({interval})")

    # Format the dates to make them more readable
    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)  # Rotate X-axis labels for better visibility
        # Optionally, set the frequency of date ticks
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        # Adjust tick parameters for better visibility
        ax.tick_params(axis='x', which='major', labelsize=10)
    except Exception as e:
        print(f"Error formatting date axis: {e}")

    # Display the legend for SL, TP, and Entry lines
    try:
        ax.legend()
    except Exception as e:
        print(f"Error adding legend: {e}")

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the plot if requested
    if save_plots:
        try:
            filename = f"XABCD_{pattern_name}_{symbol}_{interval}.jpg"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, format='jpg')
            print(f"Saved plot to {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    return fig
