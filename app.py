import time
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go
import requests

# Increase timeout and add retry mechanism
yf.set_tz_cache_location(path=None)
yf.set_options(timeout=30, retry=3)

def safe_option_chain_fetch(ticker, exp_date):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            opt_chain = ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Combine and filter options
            options = pd.concat([calls, puts])
            options = options[
                (options['bid'] > 0) & 
                (options['ask'] > 0) & 
                (options['openInterest'] > 10)
            ]
            
            return options, calls, puts
        except Exception as e:
            st.warning(f'Attempt {attempt + 1} failed: {e}')
            time.sleep(2)  # Wait before retry
    
    st.error('Failed to fetch option chain after multiple attempts')
    return None, None, None

def main():
    st.set_page_config(page_title="Implied Volatility Surface", layout="wide")
    st.title('🌟 Implied Volatility Surface')

    # [Previous functions remain the same: bs_call_price, implied_volatility, etc.]

    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.header('Parameters')
        
        ticker_symbol = st.text_input('Ticker Symbol', value='SPY', max_chars=10).upper()
        risk_free_rate = st.slider('Risk-Free Rate', min_value=0.0, max_value=0.1, value=0.015, step=0.001)
        dividend_yield = st.slider('Dividend Yield', min_value=0.0, max_value=0.1, value=0.013, step=0.001)
        min_strike_pct = st.slider('Minimum Strike Price (% of spot)', min_value=50.0, max_value=199.0, value=80.0, step=1.0)
        max_strike_pct = st.slider('Maximum Strike Price (% of spot)', min_value=51.0, max_value=200.0, value=120.0, step=1.0)
        y_axis_option = st.radio('Y-Axis:', ('Strike Price ($)', 'Moneyness'), horizontal=True)

    with right_column:
        st.header('Implied Volatility Surface')
        
        # Validation and data retrieval
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Fetch spot price with error handling
            try:
                spot_price = ticker.history(period='5d')['Close'].iloc[-1]
            except Exception:
                st.error(f'Cannot retrieve spot price for {ticker_symbol}')
                return

            # Get expiration dates
            try:
                expirations = ticker.options
            except Exception as e:
                st.error(f'Error fetching options: {e}')
                return

            # Filter and process expiration dates
            today = pd.Timestamp('today').normalize()
            exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=3)]

            if not exp_dates:
                st.error(f'No options found for {ticker_symbol}')
                return

            # Collect option data
            option_data = []
            for exp_date in exp_dates[:5]:  # Limit to first 5 expirations to prevent overload
                options, calls, puts = safe_option_chain_fetch(ticker, exp_date)
                
                if options is None or options.empty:
                    continue

                for _, row in options.iterrows():
                    option_data.append({
                        'expirationDate': exp_date,
                        'strike': row['strike'],
                        'bid': row['bid'],
                        'ask': row['ask'],
                        'mid': (row['bid'] + row['ask']) / 2,
                        'type': 'call' if row in calls.itertuples() else 'put'
                    })

            if not option_data:
                st.error('No valid option data found')
                return

            options_df = pd.DataFrame(option_data)
            
            # Rest of the processing remains similar to previous implementation
            options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
            options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

            options_df = options_df[
                (options_df['strike'] >= spot_price * (min_strike_pct / 100)) &
                (options_df['strike'] <= spot_price * (max_strike_pct / 100))
            ]

            # Implied volatility calculation
            options_df['impliedVolatility'] = options_df.apply(
                lambda row: implied_volatility(
                    price=row['mid'],
                    S=spot_price,
                    K=row['strike'],
                    T=row['timeToExpiration'],
                    r=risk_free_rate,
                    q=dividend_yield
                ), axis=1
            )

            options_df.dropna(subset=['impliedVolatility'], inplace=True)
            options_df['impliedVolatility'] *= 100

            if len(options_df) < 10:
                st.error('Insufficient option data')
                return

            # Visualization code remains the same as previous implementation
            # [Rest of the surface plot generation code]

        except Exception as e:
            st.error(f'Unexpected error: {e}')

if __name__ == "__main__":
    main()
