from datetime import datetime, timedelta

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import numpy as np
from functions import perform_trade_analysis
from config import (DEFAULT_THRESHOLDS, DEFAULT_DELTAS, INTERESTING_COLUMNS, INTERVAL_PARAMETERS,
                    TOP_40_FEATURES, ALL_FEATURES)
from classes import PatternManager
from functions import get_historical_data
from functions import get_extremes
from functions import plot_xabcd_pattern, plot_xabcd_patterns_with_sl_tp

# Load API key and models
ALPHA_VANTAGE_KEY = st.secrets["ALPHA_VANTAGE_KEY"]
clf_top40 = joblib.load("./data/clf_top40.pkl")
clf_all = joblib.load("./data/clf_topAll.pkl")
clf_top60 = joblib.load("./data/clf_top60.pkl")
reg_xgb = joblib.load("./data/reg_xgb_all.pkl")
scaler = joblib.load("./data/scaler_all.pkl")


def process_symbol_interval(symbol, interval, start_date, threshold, delta):
    """
    Processes a given symbol and interval to find XABCD patterns.

    Parameters:
    - symbol (str): Cryptocurrency symbol (e.g., 'BTCUSDT').
    - interval (str): Time interval (e.g., '1d', '4h').
    - start_date (str): Start date for historical data in 'YYYY-MM-DD' format.
    - threshold (float): Minimum relative price change (decimal) to identify an extreme high or low (e.g., 0.05 = 5%). Higher values filter out smaller swings.
    - delta (float): Maximum allowed deviation (decimal) from ideal Fibonacci ratios when validating XABCD pattern legs. Smaller values enforce stricter pattern fit.

    Returns:
    - pd.DataFrame or None: DataFrame containing detected patterns or None if no patterns found.
    """
    try:
        st.write(f"Processing {symbol} on {interval} interval from {start_date}")

        # Step 1: Get historical data
        historic_data = get_historical_data(symbol, interval, start_date)

        # Check if historical data is available
        if historic_data is None or historic_data.empty:
            st.warning(f"No historical data for {symbol} on {interval}. Skipping.")
            return None
        else:
            st.write(f"Fetched {len(historic_data)} rows of historical data for {symbol} on {interval}.")

        # Store OHLC data in session state
        ohlc_data_key = (symbol, interval)
        if 'ohlc_data_dict' not in st.session_state:
            st.session_state['ohlc_data_dict'] = {}

        st.session_state['ohlc_data_dict'][ohlc_data_key] = historic_data

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
            api_key_alpha_vantage=ALPHA_VANTAGE_KEY,
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
    # Fetch the earliest pattern start time to cover all patterns
    earliest_pattern_start_time = pd.to_datetime(filtered_df['pattern_start_time']).min()
    start_date_plot = earliest_pattern_start_time.strftime('%Y-%m-%d')
    historic_data = get_historical_data(symbol_selected, interval_selected, start_date_plot)
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


def plot_selected_pattern(pattern, candles_left, candles_right):
    symbol = pattern['symbol']
    interval = pattern['interval']
    pattern_start_time = pattern['pattern_start_time']
    D_time = pattern['D_time']

    # Convert times to datetime
    pattern_start_time_dt = pd.to_datetime(pattern_start_time)
    D_time_dt = pd.to_datetime(D_time)

    # Calculate the start and end dates for data fetching
    unit_map = {'m': 'T', 'h': 'H', 'd': 'D', 'w': 'W'}
    interval_value = int(interval[:-1])
    interval_unit = unit_map.get(interval[-1], 'D')

    start_date_plot = (pattern_start_time_dt - pd.Timedelta(candles_left * interval_value, unit=interval_unit))
    end_date_plot = (D_time_dt + pd.Timedelta(candles_right * interval_value, unit=interval_unit))

    historic_data = get_historical_data(symbol, interval, start_date_plot.strftime('%Y-%m-%d'),
                                        end_date_plot.strftime('%Y-%m-%d'))
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

    return fig_plot


def render_prediction_section(filtered_df: pd.DataFrame):
    """
    Renders the Pattern Prediction section in the Streamlit app.

    Parameters:
    - filtered_df: DataFrame of detected XABCD patterns for the current symbol/interval.
    - symbol: the cryptocurrency symbol being analyzed.
    - interval: the time interval of the patterns.
    """
    st.header("Pattern Prediction")

    # 1) Bail early if no patterns
    if filtered_df.empty:
        st.info("No patterns available for prediction.")
        return

    # 2) Build descriptive labels
    df = filtered_df.reset_index(drop=True)
    options = [f"{i + 1}: {row['pattern_name']} @ {row['X_time']} - {row.Y}-{row.exit_reason}" for i, row in df.iterrows()]

    # 3) User selects pattern
    choice = st.selectbox("Select Pattern to Predict", options, key="pattern_pred_select")
    idx = int(choice.split(":")[0]) - 1
    row = df.iloc[[idx]]

    st.subheader(f"Ad-Hoc Prediction for Pattern {idx + 1}: {row['pattern_name'].iloc[0]}")

    # 4) Feature engineering
    df_dummy = pd.get_dummies(row, drop_first=True)
    Xc_top40 = df_dummy.reindex(columns=clf_top40.get_booster().feature_names, fill_value=0)
    Xc_top60 = df_dummy.reindex(columns=clf_top60.get_booster().feature_names, fill_value=0)
    Xc_all  = df_dummy.reindex(columns=clf_all.get_booster().feature_names, fill_value=0)
    Xr = df_dummy.reindex(columns=ALL_FEATURES, fill_value=0)

    # 5) Make predictions
    prob_success_top40 = clf_top40.predict_proba(Xc_top40)[0, 1] * 100
    prob_success_top60 = clf_top60.predict_proba(Xc_top60)[0, 1] * 100
    prob_success_all = clf_all.predict_proba(Xc_all)[0, 1] * 100

    profit_xgb = reg_xgb.predict(scaler.transform(Xr))[0] * 100

    # 6) Display metrics and recommendation side by side
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏆 Success Prob. Top40", f"{prob_success_top40:.2f}%")
    c2.metric("🏆 Success Prob. Top60", f"{prob_success_top60:.2f}%")
    c3.metric("🏆 Success Prob. All-Features", f"{prob_success_all:.2f}%")
    c4.metric("💰 Predicted Profit", f"{profit_xgb:.2f}%")

    # Determine best model and build recommendation
    probs = {
        "Top40": prob_success_top40,
        "Top60": prob_success_top60,
        "All-Features": prob_success_all
    }
    best_model, best_prob = max(probs.items(), key=lambda x: x[1])

    # Customize your thresholds to taste
    HIGH_CONFIDENCE = 50.0  # %
    MIN_PROFIT = 0.0  # %

    if best_prob < 50:
        recommendation = (
            f"⚠️ Low confidence ({best_model} @ {best_prob:.1f}%) "
            "or non-positive profit forecast – consider skipping this setup."
        )
    elif best_prob >= HIGH_CONFIDENCE and profit_xgb >= MIN_PROFIT:
        recommendation = (
            f"✅ Strong signal from {best_model} model ({best_prob:.1f}% success) "
            f"and healthy profit expectation ({profit_xgb:.2f}%) – "
            "you may consider taking this trade with appropriate sizing."
        )
    else:
        recommendation = (
            f"ℹ️ Moderate outlook: {best_model} at {best_prob:.1f}% success, "
            f"{profit_xgb:.2f}% profit projected – "
            "you might wait for additional confirmation or scale in cautiously."
        )

    # 5th column: show recommendation
    c5.markdown(f"### 💡 Recommendation\n\n{recommendation}")

    # 7) Plot prediction chart with SL/TP overlay
    with st.expander("Show Prediction Chart", expanded=True):
        # Extract symbol and interval from the selected pattern row
        pat = row.iloc[0]
        sym = pat['symbol']
        iv = pat['interval']
        # Retrieve stored OHLC for this symbol/interval
        ohlc = st.session_state['ohlc_data_dict'].get((sym, iv))
        if ohlc is None or ohlc.empty:
            st.error("OHLC data unavailable for this pattern.")
        else:
            # Prepare OHLC index
            df_ohlc = ohlc.copy()
            df_ohlc['open_time'] = pd.to_datetime(df_ohlc['open_time'])
            df_ohlc.set_index('open_time', inplace=True)
            # Determine candle window using INTERVAL_PARAMETERS
            left = 10
            right = 10
            # Plot using your SL/TP overlay helper
            fig = plot_xabcd_patterns_with_sl_tp(
                pattern=pat,
                ohlc=df_ohlc,
                candles_left=left,
                candles_right=right
            )
            st.pyplot(fig)


def main():
    st.set_page_config(page_title="Cryptocurrency XABCD Pattern Analyzer", layout="wide")
    st.title("Cryptocurrency XABCD Pattern Analyzer")

    # --- Session state init ---
    st.session_state.setdefault('ohlc_data_dict', {})
    st.session_state.setdefault('patterns_df', pd.DataFrame())
    st.session_state.setdefault('trade_results', pd.DataFrame())

    # --- Sidebar config ---
    st.sidebar.header("Configuration")
    symbols = st.sidebar.multiselect(
        "Symbols", ["BTCUSDT", "ETHUSDT", "ETHBTC", "SOLUSDT"], default=["BTCUSDT"]
    )
    intervals = st.sidebar.multiselect(
        "Intervals", ["1w", "1d", "4h","1h", "30m","15m"], default=["1d"]
    )

    thresholds, deltas, start_dates = {}, {}, {}
    for iv in intervals:
        st.sidebar.markdown(f"### {iv} Settings")
        earliest = datetime.utcnow() - timedelta(7*365)
        date_input = st.sidebar.date_input(
            "Start Date", value=earliest.date(), min_value=earliest.date(), max_value=datetime.utcnow().date(),
            key=f"start_{iv}"
        )
        start_dates[iv] = date_input.strftime('%Y-%m-%d')
        thresholds[iv] = st.sidebar.number_input(
            "DELTA-Minimum relative price change (decimal) to identify an extreme high or low (e.g., 0.05 = 5%). Higher values filter out smaller swings. ",
            0.0, 1.0, DEFAULT_THRESHOLDS.get(iv, 0.04), 0.0001, key=f"th_{iv}"
        )
        deltas[iv] = st.sidebar.number_input(
            "Threshold-Maximum allowed deviation (decimal) from ideal Fibonacci ratios when validating XABCD pattern legs. Smaller values enforce stricter pattern fit.",
            0.0, 1.0, DEFAULT_DELTAS.get(iv, 0.1), 0.0001, key=f"dl_{iv}"
        )

    # --- Fetch & analyze patterns ---
    if st.sidebar.button("Fetch & Analyze"):
        all_patterns = []
        with st.spinner("Detecting patterns..."):
            for iv in intervals:
                for sym in symbols:
                    df = process_symbol_interval(sym, iv, start_dates[iv], thresholds[iv], deltas[iv])
                    if df is not None and not df.empty:
                        all_patterns.append(df)
        if all_patterns:
            st.success("Patterns detected successfully.")
            st.session_state['patterns_df'] = pd.concat(all_patterns, ignore_index=True)
        else:
            st.error("No patterns found.")

    # --- Display patterns ---
    df_patterns = st.session_state['patterns_df']
    if df_patterns.empty:
        st.info("Click 'Fetch & Analyze' to detect patterns.")
        return

    st.header("Overall Patterns")
    symbol = st.selectbox("Select Symbol", df_patterns['symbol'].unique())
    interval = st.selectbox("Select Interval", df_patterns['interval'].unique())
    df_sel = df_patterns[(df_patterns.symbol == symbol) & (df_patterns.interval == interval)]
    if df_sel.empty:
        st.warning("No patterns for this selection.")
    else:
        st.dataframe(df_sel)
        create_candlestick_with_patterns(df_sel, symbol, interval)
    # --- Trade Analysis Trigger ---
    if st.button("Run Trade Analysis"):
        trades = perform_trade_analysis(df_patterns, st.session_state['ohlc_data_dict'])
        if trades is not None and not trades.empty:
            st.success("Trade analysis complete.")
            st.session_state['trade_results'] = trades
        else:
            st.warning("No trade outcomes.")

    # --- Trade Filters & Visualization (always shown once analysis has run) ---
    df_trades = st.session_state.get('trade_results', pd.DataFrame())
    if not df_trades.empty:
        st.subheader("Filter Trades")
        syms = df_trades.symbol.unique().tolist()
        exrs = df_trades.exit_reason.unique().tolist()
        f_sym = st.multiselect("Symbols", syms, default=syms)
        f_exr = st.multiselect("Exit Reasons", exrs, default=exrs)
        prof = st.checkbox("Profitable only")
        df_f = df_trades[
            df_trades.symbol.isin(f_sym) &
            df_trades.exit_reason.isin(f_exr)
            ]
        if prof:
            df_f = df_f[df_f.profit > 0]

        st.dataframe(df_f[INTERESTING_COLUMNS])

        # Visualize a trade
        sltp = df_f.dropna(subset=['SL', 'TP'])
        if not sltp.empty:
            st.subheader("Visualize Trade")
            opts = [
                f"{i + 1}: {r.symbol}@{r.D_time} ({r.profit:.2f}%)"
                for i, r in sltp.reset_index().iterrows()
            ]

            choice = st.selectbox("Select Trade", opts, key="trade_to_plot")
            idx = int(choice.split(":")[0]) - 1
            trade = sltp.reset_index().iloc[idx]

            ohlc = st.session_state['ohlc_data_dict'][(trade.symbol, trade.interval)].copy()
            ohlc['open_time'] = pd.to_datetime(ohlc['open_time'])
            ohlc.set_index('open_time', inplace=True)
            left, right = 2, INTERVAL_PARAMETERS[trade.interval][5] + 1
            fig = plot_xabcd_patterns_with_sl_tp(pattern=trade, ohlc=ohlc, candles_left=left, candles_right=right)
            st.pyplot(fig)

        # --- Pattern Prediction (always shown too) ---
        render_prediction_section(sltp)


if __name__ == "__main__":
    main()
