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

def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan

    def objective_function(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price

    try:
        implied_vol = brentq(objective_function, 1e-6, 5)
    except (ValueError, RuntimeError):
        implied_vol = np.nan

    return implied_vol

def safe_option_chain_fetch(ticker, exp_date):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            opt_chain = ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            options = pd.concat([calls, puts])
            options = options[
                (options['bid'] > 0) & 
                (options['ask'] > 0) & 
                (options['openInterest'] > 10)
            ]
            
            return options, calls, puts
        except Exception as e:
            st.warning(f'Attempt {attempt + 1} failed: {e}')
            time.sleep(2)
    
    st.error('Failed to fetch option chain')
    return None, None, None

def main():
    st.set_page_config(page_title="Implied Volatility Surface", layout="wide")
    st.title('🌟 Implied Volatility Surface')

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
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            try:
                spot_price = ticker.history(period='5d')['Close'].iloc[-1]
            except Exception:
                st.error(f'Cannot retrieve spot price for {ticker_symbol}')
                return

            try:
                expirations = ticker.options
            except Exception as e:
                st.error(f'Error fetching options: {e}')
                return

            today = pd.Timestamp('today').normalize()
            exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=3)]

            if not exp_dates:
                st.error(f'No options found for {ticker_symbol}')
                return

            option_data = []
            for exp_date in exp_dates[:5]:
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
            
            options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
            options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

            options_df = options_df[
                (options_df['strike'] >= spot_price * (min_strike_pct / 100)) &
                (options_df['strike'] <= spot_price * (max_strike_pct / 100))
            ]

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

            Y = options_df['strike'].values if y_axis_option == 'Strike Price ($)' else options_df['strike'].values / spot_price
            y_label = 'Strike Price ($)' if y_axis_option == 'Strike Price ($)' else 'Moneyness (Strike / Spot)'

            X = options_df['timeToExpiration'].values
            Z = options_df['impliedVolatility'].values

            ti = np.linspace(X.min(), X.max(), 50)
            ki = np.linspace(Y.min(), Y.max(), 50)
            T, K = np.meshgrid(ti, ki)

            Zi = griddata((X, Y), Z, (T, K), method='linear')
            Zi = np.ma.array(Zi, mask=np.isnan(Zi))

            fig = go.Figure(data=[go.Surface(
                x=T, y=K, z=Zi,
                colorscale='Viridis',
                colorbar_title='Implied Volatility (%)'
            )])

            fig.update_layout(
                title=f'Implied Volatility Surface for {ticker_symbol} Options',
                scene=dict(
                    xaxis_title='Time to Expiration (years)',
                    yaxis_title=y_label,
                    zaxis_title='Implied Volatility (%)'
                ),
                width=1000,
                height=800,
                margin=dict(l=65, r=50, b=65, t=90)
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f'Unexpected error: {e}')

if __name__ == "__main__":
    main()
