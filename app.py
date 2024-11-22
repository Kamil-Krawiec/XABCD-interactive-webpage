# app.py

import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from classes.pattern_manager_class import PatternManager
from functions.binance_api import get_historical_data
from functions.extremas import get_extremes
from functions.plotting import plot_xabcd_patterns_with_sl_tp
from binance.client import Client

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define thresholds and deltas for each interval
thresholds = {
    '30m': 0.01,
    '1h': 0.012,
    '4h': 0.02,
    '1d': 0.03,
    '1w': 0.04,
}

deltas = {
    '30m': 0.1,
    '1h': 0.13,
    '4h': 0.18,
    '1d': 0.1,
    '1w': 0.1,
}


# Function to process data
def process_symbol_interval(symbol, interval, start_date):
    try:
        # Initialize Binance client
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

        # Adjust threshold based on interval
        threshold = thresholds.get(interval, 0.04)  # Default to 0.04 if interval not in thresholds

        # Step 2: Detect extreme points
        extremes = get_extremes(historic_data, threshold)

        # Check if extremes are found
        if extremes is None or extremes.empty:
            st.warning(f"No extremes found for {symbol} on {interval}. Skipping.")
            return None
        else:
            st.write(f"Found {len(extremes)} extremes for {symbol} on {interval}.")

        # Adjust delta based on interval
        delta = deltas.get(interval, 0.4)  # Default to 0.4 if interval not in deltas

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


# Function to create Candlestick chart with overlaid patterns
def create_candlestick_with_patterns(filtered_df, symbol_selected, interval_selected):
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

    # Add each pattern
    for idx, pattern in filtered_df.iterrows():
        points = ['X', 'A', 'B', 'C', 'D']
        x_times = [pattern[f'{point}_time'] for point in points]
        y_prices = [pattern[f'{point}_price'] for point in points]
        fig.add_trace(go.Scatter(
            x=x_times,
            y=y_prices,
            mode='lines+markers',
            name=f"Pattern {idx + 1}",
            marker=dict(size=8),
            line=dict(width=2)
        ))

    fig.update_layout(
        title=f"XABCD Patterns for {symbol_selected} on {interval_selected} Interval",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)


# Streamlit App
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
                    df = process_symbol_interval(symbol, interval, start_date)
                    if df is not None and not df.empty:
                        all_patterns_df = pd.concat([all_patterns_df, df], ignore_index=True)

        if not all_patterns_df.empty:
            # Save the master DataFrame to CSV
            all_patterns_df.to_csv(csv_output_path, index=False)
            st.success(f"Data saved to `{csv_output_path}`")
            st.dataframe(all_patterns_df)

            # Visualization: Candlestick Chart with Patterns
            st.header("XABCD Patterns Visualization")
            symbol_selected = st.selectbox("Select Symbol for Visualization", all_patterns_df['symbol'].unique())
            interval_selected = st.selectbox("Select Interval for Visualization", all_patterns_df['interval'].unique())

            filtered_df = all_patterns_df[
                (all_patterns_df['symbol'] == symbol_selected) &
                (all_patterns_df['interval'] == interval_selected)
                ]

            if not filtered_df.empty:
                create_candlestick_with_patterns(filtered_df, symbol_selected, interval_selected)

                # Detailed View: Select a Pattern to View
                st.header("Detailed Pattern View")

                # Create options for selectbox
                pattern_options = [
                    f"Pattern {i + 1}: {row['pattern_name']} on {row['X_time']}"
                    for i, row in filtered_df.iterrows()
                ]

                selected_option = st.selectbox(
                    "Select a Pattern to View",
                    options=pattern_options
                )

                # Map the selected option back to the dataframe index
                selected_index = pattern_options.index(selected_option)

                if st.button("View Selected Pattern"):
                    try:
                        # Retrieve the selected pattern using .iloc
                        pattern = filtered_df.iloc[selected_index].to_dict()

                        # Fetch the corresponding OHLC data
                        client = Client(os.getenv('API_KEY'), os.getenv('SECRET_KEY'))
                        symbol = pattern['symbol']
                        interval = pattern['interval']
                        pattern_start_time = pattern['pattern_start_time']
                        start_date_plot = pattern_start_time.split(' ')[0]  # Extract date part

                        historic_data = get_historical_data(client, symbol, interval, start_date_plot)
                        if historic_data is None or historic_data.empty:
                            st.error("Failed to retrieve OHLC data for plotting.")
                            st.stop()

                        # Ensure 'open_time' is datetime and set as index
                        historic_data['open_time'] = pd.to_datetime(historic_data['open_time'])
                        historic_data.set_index('open_time', inplace=True)

                        # Plot the pattern
                        fig_plot = plot_xabcd_patterns_with_sl_tp(
                            pattern=pattern,
                            ohlc=historic_data,
                            save_plots=False
                        )
                        st.pyplot(fig_plot)
                    except Exception as e:
                        st.error(f"Error viewing selected pattern: {e}")
            else:
                st.warning("No data available for the selected symbol and interval.")
        else:
            st.error("No patterns were found for the selected configuration.")


if __name__ == "__main__":
    main()
