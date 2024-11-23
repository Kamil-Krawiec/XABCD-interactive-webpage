# app.py

import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from classes.pattern_manager_class import PatternManager
from functions.binance_api import get_historical_data
from functions.extremas import get_extremes
from functions.plotting import plot_xabcd_patterns_with_sl_tp, plot_xabcd_pattern
from binance.client import Client

from dotenv import load_dotenv

load_dotenv()

# Default thresholds and deltas
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
    '1d': 0.10,
    '1w': 0.10,
}

def process_symbol_interval(symbol, interval, start_date, threshold, delta):
    """
    Processes a given symbol and interval to find XABCD patterns.

    Parameters:
    - symbol (str): Cryptocurrency symbol (e.g., 'BTCUSDT').
    - interval (str): Time interval (e.g., '1d', '4h').
    - start_date (str): Start date for historical data in 'YYYY-MM-DD' format.
    - threshold (float): Threshold value for extreme point detection.
    - delta (float): Delta value for pattern detection.

    Returns:
    - pd.DataFrame or None: DataFrame containing detected patterns or None if no patterns found.
    """
    try:
        # Initialize Binance client with API credentials
        client = Client(os.getenv('API_KEY'), os.getenv('SECRET_KEY'))
        st.write(f"Processing {symbol} on {interval} interval from {start_date}")

        # Step 1: Get historical data
        historic_data = get_historical_data(client, symbol, interval, start_date)

        # Check if historical data is available
        if historic_data is None or historic_data.empty:
            st.warning(f"No historical data for {symbol} on {interval}. Skipping.")
            return None
        else:
            st.write(f"Fetched {len(historic_data)} rows of historical data for {symbol} on {interval}.")

        # Step 2: Detect extreme points
        extremes = get_extremes(historic_data, threshold)

        # Check if extremes are found
        if extremes is None or extremes.empty:
            st.warning(f"No extremes found for {symbol} on {interval}. Skipping.")
            return None
        else:
            st.write(f"Found {len(extremes)} extremes for {symbol} on {interval}.")

        # Step 3: Initialize PatternManager
        pm = PatternManager(
            historic_data, extremes, delta,
            api_key_alpha_vantage=os.getenv('ALPHA_VANTAGE_KEY'),
            symbol=symbol,
            interval=interval
        )

        # Step 4: Find XABCD patterns
        pm.find_xabcd_patterns()

        # Check if patterns are found
        if not pm.pattern_list:
            st.info(f"No patterns found for {symbol} on {interval}.")
            return None
        else:
            st.write(f"Found {len(pm.pattern_list)} patterns for {symbol} on {interval}.")

        # Convert patterns to DataFrame
        pm.patterns_to_dataframe()

        # Add extra columns to track symbol and interval
        pm.pattern_df['symbol'] = symbol
        pm.pattern_df['interval'] = interval

        return pm.pattern_df

    except Exception as e:
        st.error(f"Error processing {symbol} on {interval}: {e}")
        return None

def create_candlestick_with_patterns(filtered_df, symbol_selected, interval_selected):
    """
    Creates and displays a Plotly Candlestick chart with overlaid XABCD patterns.

    Parameters:
    - filtered_df (pd.DataFrame): DataFrame containing patterns to overlay.
    - symbol_selected (str): Selected cryptocurrency symbol.
    - interval_selected (str): Selected time interval.
    """
    client = Client(os.getenv('API_KEY'), os.getenv('SECRET_KEY'))
    # Fetch the latest data to cover all patterns
    latest_pattern_start_time = pd.to_datetime(filtered_df['pattern_start_time']).max()
    start_date_plot = latest_pattern_start_time.strftime('%Y-%m-%d')
    historic_data = get_historical_data(client, symbol_selected, interval_selected, start_date_plot)
    if historic_data is None or historic_data.empty:
        st.error("Failed to retrieve OHLC data for plotting.")
        return

    # Ensure 'open_time' is datetime and set as index
    historic_data['open_time'] = pd.to_datetime(historic_data['open_time'])
    historic_data.set_index('open_time', inplace=True)

    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=historic_data.index,
        open=historic_data['open'],
        high=historic_data['high'],
        low=historic_data['low'],
        close=historic_data['close'],
        name='Candlestick'
    ))

    # Add each pattern with labels
    for idx, pattern in filtered_df.iterrows():
        points = ['X', 'A', 'B', 'C', 'D']
        x_times = [pattern[f'{point}_time'] for point in points]
        y_prices = [pattern[f'{point}_price'] for point in points]
        labels = points  # Labels for each point

        fig.add_trace(go.Scatter(
            x=x_times,
            y=y_prices,
            mode='lines+markers+text',
            name=f"Pattern {idx + 1}",
            marker=dict(size=8),
            line=dict(width=2),
            text=labels,
            textposition="top center"
        ))

    fig.update_layout(
        title=f"XABCD Patterns for {symbol_selected} on {interval_selected} Interval",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Cryptocurrency XABCD Pattern Analyzer", layout="wide")
    st.title("Cryptocurrency XABCD Pattern Analyzer")

    # Sidebar for user inputs
    st.sidebar.header("Configuration")

    # User selects symbols
    symbols = st.sidebar.multiselect(
        "Select Symbols",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ETHBTC", "SOLUSDT"],
        default=["BTCUSDT"]
    )

    # User selects intervals
    intervals = st.sidebar.multiselect(
        "Select Intervals",
        ["1w", "1d", "4h", "1h", "30m"],
        default=["1d"]
    )

    # Let users input thresholds and deltas
    thresholds = {}
    deltas = {}
    for interval in intervals:
        default_threshold = DEFAULT_THRESHOLDS.get(interval, 0.04)
        default_delta = DEFAULT_DELTAS.get(interval, 0.10)
        thresholds[interval] = st.sidebar.number_input(
            f"Price percent change for extremes {interval}", min_value=0.000, max_value=1.000, value=default_threshold, step=0.001, format="%.3f")
        deltas[interval] = st.sidebar.number_input(
            f"Formation error allowed for {interval}", min_value=0.000, max_value=1.000, value=default_delta, step=0.001, format="%.3f")

    # User selects start date
    start_date = st.sidebar.date_input(
        "Start Date",
        pd.to_datetime("2017-01-01")
    ).strftime("%Y-%m-%d")

    # Button to fetch data
    if st.sidebar.button("Fetch and Analyze"):
        if not symbols or not intervals:
            st.error("Please select at least one symbol and one interval.")
            st.stop()

        csv_output_path = "data/crypto_patterns.csv"
        all_patterns_df = pd.DataFrame()

        with st.spinner("Collecting and processing data..."):
            for interval in intervals:
                for symbol in symbols:
                    df = process_symbol_interval(symbol, interval, start_date, thresholds[interval], deltas[interval])
                    if df is not None and not df.empty:
                        all_patterns_df = pd.concat([all_patterns_df, df], ignore_index=True)

        if not all_patterns_df.empty:
            # Save the master DataFrame to CSV
            all_patterns_df.to_csv(csv_output_path, index=False)
            st.success(f"Data saved to `{csv_output_path}`")
            st.dataframe(all_patterns_df)

            # Store the DataFrame in session state
            st.session_state['all_patterns_df'] = all_patterns_df
        else:
            st.error("No patterns were found for the selected configuration.")

    # Check if 'all_patterns_df' is in session state
    if 'all_patterns_df' in st.session_state:
        all_patterns_df = st.session_state['all_patterns_df']

        # Visualization: Candlestick Chart with Patterns
        st.header("XABCD Patterns Visualization")
        symbol_selected = st.selectbox(
            "Select Symbol for Visualization",
            all_patterns_df['symbol'].unique(),
            key='symbol_select'
        )
        interval_selected = st.selectbox(
            "Select Interval for Visualization",
            all_patterns_df['interval'].unique(),
            key='interval_select'
        )

        def update_visualization():
            filtered_df = all_patterns_df[
                (all_patterns_df['symbol'] == symbol_selected) &
                (all_patterns_df['interval'] == interval_selected)
            ]

            if not filtered_df.empty:
                # Create Candlestick chart with overlaid patterns
                create_candlestick_with_patterns(filtered_df, symbol_selected, interval_selected)

                # Detailed View: Select a Pattern to View
                st.header("Detailed Pattern View")

                # Reset dataframe index to ensure it starts from 0 and is sequential
                filtered_df.reset_index(drop=True, inplace=True)
                pattern_options = [
                    f"Pattern {i + 1}: {row['pattern_name']} on {row['X_time']}"
                    for i, row in filtered_df.iterrows()
                ]

                # Use st.selectbox with a key to store the selection in session_state
                selected_option = st.selectbox(
                    "Select a Pattern to View",
                    options=pattern_options,
                    key='selected_pattern'
                )

                # Get the selected index from the session state
                selected_index = pattern_options.index(selected_option)

                # Add input fields for candles_left and candles_right
                candles_left = st.number_input("Candles to display before the pattern (Left)", min_value=1, max_value=100, value=10)
                candles_right = st.number_input("Candles to display after the pattern (Right)", min_value=1, max_value=100, value=10)

                # Create a placeholder for the plot
                plot_placeholder = st.empty()

                # Define a function to display the selected pattern
                def display_selected_pattern():
                    try:
                        # Retrieve the selected pattern using .iloc
                        pattern = filtered_df.iloc[selected_index].to_dict()

                        # Fetch the corresponding OHLC data with adjusted window
                        client = Client(os.getenv('API_KEY'), os.getenv('SECRET_KEY'))
                        symbol = pattern['symbol']
                        interval = pattern['interval']
                        pattern_start_time = pattern['pattern_start_time']
                        D_time = pattern['D_time']  # Assuming D_time is the end of the pattern

                        # Convert times to datetime
                        pattern_start_time_dt = pd.to_datetime(pattern_start_time)
                        D_time_dt = pd.to_datetime(D_time)

                        # Calculate the start and end dates for data fetching
                        unit_map = {'m': 'T', 'h': 'H', 'd': 'D', 'w': 'W'}
                        interval_value = int(interval[:-1])
                        interval_unit = unit_map.get(interval[-1], 'D')

                        start_date_plot = (pattern_start_time_dt - pd.Timedelta(candles_left * interval_value, unit=interval_unit)).strftime('%Y-%m-%d')
                        end_date_plot = (D_time_dt + pd.Timedelta(candles_right * interval_value, unit=interval_unit)).strftime('%Y-%m-%d')

                        historic_data = get_historical_data(client, symbol, interval, start_date_plot, end_date_plot)
                        if historic_data is None or historic_data.empty:
                            st.error("Failed to retrieve OHLC data for plotting.")
                            st.stop()

                        # Ensure 'open_time' is datetime and set as index
                        historic_data['open_time'] = pd.to_datetime(historic_data['open_time'])
                        historic_data.set_index('open_time', inplace=True)

                        # Plot the pattern
                        fig_plot = plot_xabcd_pattern(
                            pattern=pattern,
                            ohlc=historic_data,
                            candles_left=candles_left,
                            candles_right=candles_right,
                            save_plots=False
                        )
                        plot_placeholder.pyplot(fig_plot)
                    except Exception as e:
                        st.error(f"Error viewing selected pattern: {e}")

                # Use st.button with on_click to ensure the callback has access to the current session state
                st.button("View Selected Pattern", on_click=display_selected_pattern)

            else:
                st.warning("No data available for the selected symbol and interval.")

        # Call the function to update visualization
        update_visualization()
    else:
        st.info("Please fetch data to visualize patterns.")

if __name__ == "__main__":
    main()