import os
import pandas as pd
import streamlit as st
import plotly.express as px

from classes.pattern_manager_class import PatternManager
from functions.binance_api import get_historical_data
from functions.extremas import get_extremes
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

# Function to process data (as defined in your main script)
def process_symbol_interval(symbol, interval, start_date):
    try:
        # Initialize Binance client inside the process to avoid issues with multiprocessing
        client = Client(os.getenv('API_KEY'), os.getenv('SECRET_KEY'))
        print(f"Processing {symbol} on {interval} interval from {start_date}")

        # Step 1: Get historical data
        historic_data = get_historical_data(client, symbol, interval, start_date)

        # Check if historical data is available
        if historic_data is None or historic_data.empty:
            print(f"No historical data for {symbol} on {interval}. Skipping.")
            return None
        else:
            print(f"Fetched {len(historic_data)} rows of historical data for {symbol} on {interval}.")

        # Adjust threshold based on interval
        threshold = thresholds.get(interval, 0.04)  # Default to 0.04 if interval not in thresholds

        # Step 2: Detect extreme points
        extremes = get_extremes(historic_data, threshold)

        # Check if extremes are found
        if extremes is None or extremes.empty:
            print(f"No extremes found for {symbol} on {interval}. Skipping.")
            return None
        else:
            print(f"Found {len(extremes)} extremes for {symbol} on {interval}.")

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
            print(f"No patterns found for {symbol} on {interval}.")
            return None
        else:
            print(f"Found {len(pm.pattern_list)} patterns for {symbol} on {interval}.")

        # Convert patterns to DataFrame
        pm.patterns_to_dataframe()

        # Add extra columns to track symbol and interval
        pm.pattern_df['symbol'] = symbol
        pm.pattern_df['interval'] = interval

        return pm.pattern_df

    except Exception as e:
        print(f"Error processing {symbol} on {interval}: {e}")
        return None

# Streamlit App
def main():
    st.title("Cryptocurrency XABCD Pattern Analyzer")

    # Sidebar for user inputs
    st.sidebar.header("Configuration")

    # User selects symbols
    symbols = st.sidebar.multiselect(
        "Select Symbols",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ETHBTC", "SOLUSDT"],
    )

    # User selects intervals
    intervals = st.sidebar.multiselect(
        "Select Intervals",
        ["1w", "1d", "4h", "1h", "30m"],
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
            return

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
            st.success(f"Data saved to {csv_output_path}")
            st.dataframe(all_patterns_df)

            # Visualization
            st.header("XABCD Patterns Visualization")
            symbol_selected = st.selectbox("Select Symbol for Visualization", all_patterns_df['symbol'].unique())
            interval_selected = st.selectbox("Select Interval for Visualization", all_patterns_df['interval'].unique())

            filtered_df = all_patterns_df[(all_patterns_df['symbol'] == symbol_selected) &
                                          (all_patterns_df['interval'] == interval_selected)]
            if not filtered_df.empty:
                fig = px.scatter(
                    filtered_df,
                    x='X_time',  # Ensure these columns exist in your DataFrame
                    y='X_price',  # Adjust based on your DataFrame
                    title=f"XABCD Patterns for {symbol_selected} on {interval_selected} Interval",
                    hover_data=['pattern_name', 'ratio_AB_XA', 'ratio_BC_AB', 'ratio_CD_BC', 'ratio_AD_XA']
                )
                st.plotly_chart(fig)
            else:
                st.warning("No data available for the selected symbol and interval.")
        else:
            st.error("No patterns were found for the selected configuration.")

if __name__ == "__main__":
    main()