import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd


def plot_xabcd_patterns_with_sl_tp(pattern, ohlc, save_plots=False, save_dir='charts'):
    """
    Plots a single ABCD pattern with Stop-Loss (SL) and Take-Profit (TP) levels on an OHLC candlestick chart.

    Parameters:
    - pattern (pd.Series or dict): Series or dictionary containing pattern details.
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
        dt_array = dt_index.to_numpy()
        target_np = np.datetime64(target_time)
        diff = np.abs(dt_array - target_np)
        nearest_idx = diff.argmin()
        return nearest_idx

    # Extract pattern details
    symbol = pattern['symbol']
    interval = pattern['interval']
    pattern_name = pattern['pattern_name']
    pattern_type = pattern['pattern_type']
    profit = pattern['profit']
    points = ['X', 'A', 'B', 'C', 'D']
    point_times = [pd.to_datetime(pattern[f'{point}_time']) for point in points]
    point_prices = [pattern[f'{point}_price'] for point in points]

    # Define the window: from pattern_start_time - candles_left to D_time + candles_right
    candles_left = 10
    candles_right = 10
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
                 axtitle=f"{symbol} - {interval} ABCD Pattern: {pattern_name}")
    except Exception as e:
        print(f"Error plotting candlestick: {e}")
        plt.close(fig)
        return None

    # Overlay the XABCD pattern
    try:
        ax.plot(point_times, point_prices, marker='o', linestyle='-', color='r', label='ABCD Pattern')
    except Exception as e:
        print(f"Error plotting ABCD pattern: {e}")

    # Annotate the points
    try:
        colors = ['r', 'g', 'b', 'm', 'y']
        for point, time, price, color in zip(points, point_times, point_prices, colors):
            ax.annotate(point, xy=(time, price), xytext=(time, price),
                        textcoords='offset points', color=color, fontsize=12, weight='bold')
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
                    '',
                    xy=(entry_time, entry_price + (data_segment['high'].max() - data_segment['low'].min()) * 0.02),
                    xytext=(entry_time, entry_price),
                    arrowprops=dict(
                        arrowstyle='->',
                        color='orange',
                        lw=3.5,
                        connectionstyle='arc3,rad=0.2',
                        shrinkA=0,
                        shrinkB=5,
                    ),
                )
                ax.text(entry_time, entry_price, 'Entry', color='orange', fontsize=12, weight='bold',
                        verticalalignment='bottom')
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
            filename = f"XABCD_{pattern_name}_{idx + 1}_SL_TP.jpg"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, format='jpg')
            print(f"Saved plot to {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    return fig