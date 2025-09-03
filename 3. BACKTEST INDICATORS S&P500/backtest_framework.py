import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import warnings

# Ignore non-critical warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Creation of the folder to store S&P500 companies data
db_folder = 'sp500_database'
price_data_folder = os.path.join(db_folder, 'price_data')
features_folder = os.path.join(db_folder, 'features')
sp500_list_file = os.path.join(db_folder, 'S&P500_companies.csv')
os.makedirs(price_data_folder, exist_ok=True)
os.makedirs(features_folder, exist_ok=True)

# Market regimes (Periods and Market regimes)
market_regimes = [
    {"name": "2000-08-01 - 2002-09-03", "label": "Bear Market", "start": "2000-08-01", "end": "2002-09-03"},
    {"name": "2002-09-03 - 2007-10-01", "label": "Bull Market", "start": "2002-09-03", "end": "2007-10-01"},
    {"name": "2007-10-01 - 2009-02-02", "label": "Bear Market", "start": "2007-10-01", "end": "2009-02-02"},
    {"name": "2009-02-02 - 2019-12-02", "label": "Bull Market", "start": "2009-02-02", "end": "2019-12-02"},
    {"name": "2019-12-02 - 2020-08-03", "label": "COVID V-Shape", "start": "2019-12-02", "end": "2020-08-03"},
    {"name": "2020-08-03 - 2021-12-01", "label": "Bull Market", "start": "2020-08-03", "end": "2021-12-01"},
    {"name": "2021-12-01 - 2022-09-01", "label": "Bear Market", "start": "2021-12-01", "end": "2022-09-01"}
]

# Strategies to run (Strategies and Parameters)
strategies_to_run = [
    # --- Moving Average Strategies ---
    {'name': 'Price-SMA Crossover Strategy', 'func_name': 'backtest_sma_price_crossover', 'params': {'period': 9}},
    {'name': 'Dual SMA Crossover Strategy', 'func_name': 'backtest_dual_sma_crossover', 'params': {'fast': 50, 'slow': 200}},
    {'name': 'Price-EMA Crossover Strategy', 'func_name': 'backtest_ema_price_crossover', 'params': {'period': 9}},
    {'name': 'Dual EMA Crossover Strategy', 'func_name': 'backtest_dual_ema_crossover', 'params': {'fast': 12, 'slow': 26}},
    {'name': 'Price-ALMA Crossover Strategy', 'func_name': 'backtest_alma_price_crossover', 'params': {'window': 9, 'sigma': 6, 'offset': 0.85}},
    {'name': 'Dual ALMA Crossover Strategy', 'func_name': 'backtest_dual_alma_crossover', 'params': {'fast': 12, 'slow': 26, 'sigma': 6, 'offset': 0.85}},
    # --- RSIs ---
    {'name': 'RSI Mean Reversion Strategy', 'func_name': 'backtest_rsi_oversold', 'params': {'period': 14, 'upper': 70, 'lower': 30}},
    {'name': 'RSI Signal Line Crossover Strategy', 'func_name': 'backtest_rsi_sma_signal', 'params': {'rsi_period': 14, 'signal_period': 14}},
    # --- MACDs ---
    {'name': 'MACD Signal Line Crossover Strategy', 'func_name': 'backtest_macd_crossover', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
    {'name': 'Zero-Lag MACD Signal Line Crossover Strategy', 'func_name': 'backtest_zlmacd_crossover', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
    {'name': 'Adaptive MACD Signal Line Crossover Strategy', 'func_name': 'backtest_adaptive_macd_crossover', 'params': {'fast': 12, 'slow': 26, 'signal': 9, 'atr': 14}},
    # --- Channels and Bands ---
    {'name': 'Bollinger Bands Mean Reversion Strategy', 'func_name': 'backtest_bbands_mean_reversion', 'params': {'length': 20, 'std': 2.0}},
    {'name': 'Bollinger Bands Breakout Strategy', 'func_name': 'backtest_bbands_breakout', 'params': {'length': 20, 'std': 2.0}},
    {'name': 'Donchian Channel Breakout Strategy', 'func_name': 'backtest_donchian_breakout', 'params': {'period': 20}},
    {'name': 'Keltner Channel Breakout Strategy', 'func_name': 'backtest_keltner_breakout', 'params': {'length': 20, 'multiplier': 2.0}},
]

# Benchmark and Trading account parameters 
benchmark_ticker = '^GSPC'  # ticker symbol for S&P 500 index
initial_capital = 10000.0
trade_size = 1000.0  # commissions omitted due to purpose of analysis

# --- FINANCIAL DATA PREPARATION ---

# Retrieve S&P 500 companies from Wikipedia
## Optimized: downloads data only if the file is not found locally.
def get_sp500_tickers():
    if os.path.exists(sp500_list_file):
        df = pd.read_csv(sp500_list_file, header=None)
        df.columns = ['Security', 'Symbol']
    else:
        print("Downloading S&P 500's companies list from Wikipedia...")
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        df_full = pd.read_html(url, header=0)[0]
        df_full['Symbol'] = df_full['Symbol'].str.replace('.', '-', regex=False)
        df = df_full[['Security', 'Symbol']].copy()
        df.to_csv(sp500_list_file, index=False, header=False)
    return dict(zip(df['Security'], df['Symbol']))

# Retrieve historical stock prices (from 2000 to present)
## Optimized: downloads data only if the file is not found locally.
def get_price_data(ticker):
    filepath = os.path.join(price_data_folder, f"{ticker}.csv")
    df = None
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        except Exception:
            try: os.remove(filepath)
            except OSError: pass
            return get_price_data(ticker)
    else:
        print(f"  Downloading price data for {ticker}...")
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start="2000-01-01", end=datetime.now(), auto_adjust=True)
            if df.empty: return None
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.to_csv(filepath, index=True)
        except Exception: return None
    
    if df is not None and not df.empty:
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df.index.name = 'Date'
    return df

# Calculate stock features for each period (Beta Coefficient (β), Historical Volatility, Average Daily Dollar Volume (in $))
## Optimized: downloads data only if the file is not found locally.
def calculate_company_features(ticker, all_price_data, benchmark_data):
    filepath = os.path.join(features_folder, f"{ticker}_features.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"  Calculating features for {ticker}...")
        results = []
        if benchmark_data is None: return None
        for regime in market_regimes:
            regime_data = all_price_data.loc[regime['start']:regime['end']]
            if regime_data.empty or len(regime_data) < 20: continue
            volatility = regime_data['Close'].pct_change().std() * np.sqrt(252)
            beta_start = (datetime.strptime(regime['start'], '%Y-%m-%d') - relativedelta(years=1)).strftime('%Y-%m-%d')
            beta_end = regime['start']
            beta_data_asset = all_price_data.loc[beta_start:beta_end]
            beta_data_bench = benchmark_data.loc[beta_start:beta_end]
            beta = np.nan
            if not beta_data_asset.empty and not beta_data_bench.empty:
                returns = pd.concat([beta_data_asset['Close'].pct_change(), beta_data_bench['Close'].pct_change()], axis=1).dropna()
                returns.columns = ['asset', 'benchmark']
                if len(returns) > 60:
                    cov_matrix = returns.cov()
                    beta = cov_matrix.loc['asset', 'benchmark'] / cov_matrix.loc['benchmark', 'benchmark']
            dollar_volume = (regime_data['Close'] * regime_data['Volume']).mean()
            results.append({"Period": regime['name'], "Market Phase": regime['label'], "Beta": f"{beta:.2f}", "Historical Volatility": f"{volatility*100:.2f}%", "Average Volume ($)": f"${dollar_volume/1e6:,.2f}M"})
        if not results: return None
        features_df = pd.DataFrame(results)
        features_df.to_csv(filepath, index=False)
        return features_df

# --- BACKTESTING FUNCTIONS ---

def _prepare_data(df):
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_copy = df.copy()
    for col in cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    return df_copy.dropna(subset=['Close'])

def _finalize_backtest(trades, capital):
    summary = { 'P&L %': ((capital - initial_capital) / initial_capital) * 100, 'Final Capital': capital, 'Num. Trades': len(trades) }
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.insert(0, 'Trade Number', range(1, 1 + len(trades_df)))
    return summary, trades_df

# Function to backtest: Price-SMA Crossover Strategy
def backtest_sma_price_crossover(price_data, params):
    if price_data is None or len(price_data) < params['period']: return None, None
    data = _prepare_data(price_data)
    data['SMA'] = data['Close'].rolling(window=params['period']).mean()
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    close = data['Close'].to_numpy(); sma = data['SMA'].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (close[i-1] <= sma[i-1]) and (close[i] > sma[i])
        is_exit = (close[i-1] >= sma[i-1]) and (close[i] < sma[i])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: Dual SMA Crossover Strategy
def backtest_dual_sma_crossover(price_data, params):
    if price_data is None or len(price_data) < params['slow']: return None, None
    data = _prepare_data(price_data)
    data['Fast'] = data['Close'].rolling(window=params['fast']).mean()
    data['Slow'] = data['Close'].rolling(window=params['slow']).mean()
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    fast = data['Fast'].to_numpy(); slow = data['Slow'].to_numpy(); close = data['Close'].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (fast[i-1] <= slow[i-1]) and (fast[i] > slow[i])
        is_exit = (fast[i-1] >= slow[i-1]) and (fast[i] < slow[i])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: Price-EMA Crossover Strategy
def backtest_ema_price_crossover(price_data, params):
    if price_data is None or len(price_data) < params['period']: return None, None
    data = _prepare_data(price_data.copy())
    data['EMA'] = data['Close'].ewm(span=params['period'], adjust=False).mean()
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    close = data['Close'].to_numpy(); ema = data['EMA'].to_numpy(); dates = data.index 
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (close[i-1] <= ema[i-1]) and (close[i] > ema[i])
        is_exit = (close[i-1] >= ema[i-1]) and (close[i] < ema[i])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: Dual EMA Crossover Strategy
def backtest_dual_ema_crossover(price_data, params):
    if price_data is None or len(price_data) < params['slow']: return None, None
    data = _prepare_data(price_data.copy())
    data['Fast'] = data['Close'].ewm(span=params['fast'], adjust=False).mean()
    data['Slow'] = data['Close'].ewm(span=params['slow'], adjust=False).mean()
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    fast = data['Fast'].to_numpy(); slow = data['Slow'].to_numpy(); close = data['Close'].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (fast[i-1] <= slow[i-1]) and (fast[i] > slow[i])
        is_exit = (fast[i-1] >= slow[i-1]) and (fast[i] < slow[i])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: Price-ALMA Crossover Strategy
def backtest_alma_price_crossover(price_data, params):
    if price_data is None or len(price_data) < params['window']: return None, None
    data = _prepare_data(price_data.copy())
    def alma(series, window, sigma, offset):
        m = offset * (window - 1); s = window / sigma
        weights = np.exp(-((np.arange(window) - m) ** 2) / (2 * s ** 2))
        return series.rolling(window).apply(lambda x: np.dot(x, weights / weights.sum()), raw=True)
    data['ALMA'] = alma(data['Close'], params['window'], params['sigma'], params['offset'])
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    close = data['Close'].to_numpy(); alma_values = data['ALMA'].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (close[i-1] <= alma_values[i-1]) and (close[i] > alma_values[i])
        is_exit = (close[i-1] >= alma_values[i-1]) and (close[i] < alma_values[i])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: Dual ALMA Crossover Strategy
def backtest_dual_alma_crossover(price_data, params):
    if price_data is None or len(price_data) < params['slow']: return None, None
    data = _prepare_data(price_data.copy())
    def alma(series, window, sigma, offset):
        m = offset * (window - 1); s = window / sigma
        weights = np.exp(-((np.arange(window) - m) ** 2) / (2 * s ** 2))
        return series.rolling(window).apply(lambda x: np.dot(x, weights / weights.sum()), raw=True)
    data['Fast'] = alma(data['Close'], params['fast'], params['sigma'], params['offset'])
    data['Slow'] = alma(data['Close'], params['slow'], params['sigma'], params['offset'])
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    fast = data['Fast'].to_numpy(); slow = data['Slow'].to_numpy(); close = data['Close'].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (fast[i-1] <= slow[i-1]) and (fast[i] > slow[i])
        is_exit = (fast[i-1] >= slow[i-1]) and (fast[i] < slow[i])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: RSI Mean Reversion Strategy
def backtest_rsi_oversold(price_data, params):
    if price_data is None or len(price_data) < params['period']: return None, None
    data = _prepare_data(price_data.copy())
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=params['period'] - 1, min_periods=params['period']).mean()
    avg_loss = loss.ewm(com=params['period'] - 1, min_periods=params['period']).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    close = data['Close'].to_numpy(); rsi = data['RSI'].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    in_oversold = False; in_overbought = False
    for i in range(len(data)):
        is_long = (rsi[i-1] <= params['lower']) and (rsi[i] > params['lower'])
        is_exit = (rsi[i-1] >= params['upper']) and (rsi[i] < params['upper'])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False; in_overbought = False
    return _finalize_backtest(trades, capital)

# Function to backtest: RSI Signal Line Crossover Strategy
def backtest_rsi_sma_signal(price_data, params):
    if price_data is None or len(price_data) < params['rsi_period'] + params['signal_period']: return None, None
    data = _prepare_data(price_data.copy())
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=params['rsi_period'] - 1, min_periods=params['rsi_period']).mean()
    avg_loss = loss.ewm(com=params['rsi_period'] - 1, min_periods=params['rsi_period']).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['Signal'] = data['RSI'].rolling(window=params['signal_period']).mean()
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    close = data['Close'].to_numpy(); rsi = data['RSI'].to_numpy(); signal = data['Signal'].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (rsi[i-1] <= signal[i-1]) and (rsi[i] > signal[i])
        is_exit = (rsi[i-1] >= signal[i-1]) and (rsi[i] < signal[i])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: MACD Signal Line Crossover Strategy
def backtest_macd_crossover(price_data, params):
    if price_data is None or len(price_data) < params['slow']: return None, None
    data = _prepare_data(price_data.copy())
    macd_cols = data.ta.macd(fast=params['fast'], slow=params['slow'], signal=params['signal'], append=True)
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    close = data['Close'].to_numpy(); macd = data[macd_cols.columns[0]].to_numpy(); signal = data[macd_cols.columns[2]].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (macd[i-1] <= signal[i-1]) and (macd[i] > signal[i])
        is_exit = (macd[i-1] >= signal[i-1]) and (macd[i] < signal[i])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: Zero-Lag MACD Signal Line Crossover Strategy
def backtest_zlmacd_crossover(price_data, params):
    if price_data is None or len(price_data) < params['slow']: return None, None
    data = _prepare_data(price_data.copy())
    data['ZLMACD'] = ta.zlma(data['Close'], length=params['fast']) - ta.zlma(data['Close'], length=params['slow'])
    data['Signal'] = data['ZLMACD'].ewm(span=params['signal'], adjust=False).mean()
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    close = data['Close'].to_numpy(); macd = data['ZLMACD'].to_numpy(); signal = data['Signal'].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (macd[i-1] <= signal[i-1]) and (macd[i] > signal[i])
        is_exit = (macd[i-1] >= signal[i-1]) and (macd[i] < signal[i])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: Adaptive MACD Signal Line Crossover Strategy
def backtest_adaptive_macd_crossover(price_data, params):
    if price_data is None or len(price_data) < params['slow'] + params['atr']: return None, None
    data = _prepare_data(price_data.copy())
    tr = ta.true_range(data['High'], data['Low'], data['Close'])
    atr = ta.sma(tr, length=params['atr'])
    atr_scaled = (atr / atr.rolling(window=params['atr']).mean()).clip(lower=0.5, upper=1.5)
    fast_p = (params['fast'] * atr_scaled).round().fillna(params['fast'])
    slow_p = (params['slow'] * atr_scaled).round().fillna(params['slow'])
    def adaptive_ema(series, periods):
        emas = np.full(len(series), np.nan)
        alphas = 2 / (periods + 1)
        if len(series) > 0: emas[0] = series.iloc[0]
        for i in range(1, len(series)):
            emas[i] = series.iloc[i] * alphas.iloc[i] + emas[i-1] * (1 - alphas.iloc[i]) if pd.notna(emas[i-1]) else series.iloc[i]
        return pd.Series(emas, index=series.index)
    data['MACD'] = adaptive_ema(data['Close'], fast_p) - adaptive_ema(data['Close'], slow_p)
    data['Signal'] = data['MACD'].ewm(span=params['signal'], adjust=False).mean()
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    close = data['Close'].to_numpy(); macd = data['MACD'].to_numpy(); signal = data['Signal'].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (macd[i-1] <= signal[i-1]) and (macd[i] > signal[i])
        is_exit = (macd[i-1] >= signal[i-1]) and (macd[i] < signal[i])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: Bollinger Bands Mean Reversion Strategy
def backtest_bbands_mean_reversion(price_data, params):
    if price_data is None or len(price_data) < params['length']: return None, None
    data = _prepare_data(price_data.copy())
    bbands = data.ta.bbands(length=params['length'], std=params['std'], append=True)
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    close = data['Close'].to_numpy(); lower = data[f'BBL_{params["length"]}_{params["std"]}'].to_numpy(); middle = data[f'BBM_{params["length"]}_{params["std"]}'].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (close[i-1] > lower[i-1]) and (close[i] <= lower[i])
        is_exit = (close[i-1] < middle[i-1]) and (close[i] >= middle[i])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: Bollinger Bands Breakout Strategy
def backtest_bbands_breakout(price_data, params):
    if price_data is None or len(price_data) < params['length']: return None, None
    data = _prepare_data(price_data.copy())
    bbands = data.ta.bbands(length=params['length'], std=params['std'], append=True)
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    close = data['Close'].to_numpy(); upper = data[f'BBU_{params["length"]}_{params["std"]}'].to_numpy(); middle = data[f'BBM_{params["length"]}_{params["std"]}'].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (close[i-1] < upper[i-1]) and (close[i] >= upper[i-1])
        is_exit = (close[i-1] > middle[i-1]) and (close[i] <= middle[i-1])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: Donchian Channel Breakout Strategy
def backtest_donchian_breakout(price_data, params):
    if price_data is None or len(price_data) < params['period']: return None, None
    data = _prepare_data(price_data.copy())
    data.ta.donchian(lower_length=params['period'], upper_length=params['period'], append=True)
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    close = data['Close'].to_numpy(); upper = data[f'DCU_{params["period"]}_{params["period"]}'].to_numpy(); middle = data[f'DCM_{params["period"]}_{params["period"]}'].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (close[i-1] < upper[i-1]) and (close[i] >= upper[i-1])
        is_exit = (close[i-1] > middle[i-1]) and (close[i] <= middle[i-1])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# Function to backtest: Keltner Channel Breakout Strategy
def backtest_keltner_breakout(price_data, params):
    if price_data is None or len(price_data) < params['length']: return None, None
    data = _prepare_data(price_data.copy())
    kc = data.ta.kc(length=params['length'], scalar=params['multiplier'], append=True)
    upper_col, middle_col = kc.columns[-1], kc.columns[-2]
    data.dropna(inplace=True)
    if len(data) < 2: return None, None
    close = data['Close'].to_numpy(); upper = data[upper_col].to_numpy(); middle = data[middle_col].to_numpy(); dates = data.index
    trades, capital, position_open, entry_price = [], initial_capital, False, 0.0
    for i in range(1, len(data)):
        is_long = (close[i-1] < upper[i-1]) and (close[i] >= upper[i-1])
        is_exit = (close[i-1] > middle[i-1]) and (close[i] <= middle[i-1])
        if not position_open and is_long:
            position_open, entry_price, entry_date, asset_quantity = True, float(close[i]), dates[i], trade_size / float(close[i])
        elif position_open and (is_exit or i == len(data) - 1):
            exit_price = float(close[i]); pnl = (exit_price - entry_price) * asset_quantity; capital += pnl
            trades.append({'Open Date': entry_date.strftime('%Y-%m-%d'), 'Open Price': round(entry_price, 2), 'Close Date': dates[i].strftime('%Y-%m-%d'), 'Close Price': round(exit_price, 2), 'Trade Type': 'LONG', 'P&L ($)': round(pnl, 2), 'P&L (%)': round((pnl / trade_size) * 100, 2), 'Current Capital': round(capital, 2)})
            position_open = False
    return _finalize_backtest(trades, capital)

# --- 4. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    
    print("Starting s&p 500 analysis...")
    companies = get_sp500_tickers()
    total_companies = len(companies)
    print(f"Found {total_companies} companies.")
    
    print("\nPreparing benchmark data (^GSPC)...")
    benchmark_data = get_price_data(benchmark_ticker)
    if benchmark_data is None:
        print("CRITICAL ERROR: unable to get benchmark data.")
        exit()
    print("Benchmark data ready.")
    
    strategy_functions = {
        'backtest_sma_price_crossover': backtest_sma_price_crossover,
        'backtest_dual_sma_crossover': backtest_dual_sma_crossover,
        'backtest_ema_price_crossover': backtest_ema_price_crossover,
        'backtest_dual_ema_crossover': backtest_dual_ema_crossover,
        'backtest_alma_price_crossover': backtest_alma_price_crossover,
        'backtest_dual_alma_crossover': backtest_dual_alma_crossover,
        'backtest_rsi_oversold': backtest_rsi_oversold,
        'backtest_rsi_sma_signal': backtest_rsi_sma_signal,
        'backtest_macd_crossover': backtest_macd_crossover,
        'backtest_zlmacd_crossover': backtest_zlmacd_crossover,
        'backtest_adaptive_macd_crossover': backtest_adaptive_macd_crossover,
        'backtest_bbands_mean_reversion': backtest_bbands_mean_reversion,
        'backtest_bbands_breakout': backtest_bbands_breakout,
        'backtest_donchian_breakout': backtest_donchian_breakout,
        'backtest_keltner_breakout': backtest_keltner_breakout,
    }
    
    with open('complete_report.txt', 'w', encoding='utf-8') as report_file:
        # to test the code: limit the execution to 5 assets by changing the following line to
        # for i, (name, ticker) in enumerate(list(companies.items())[:5]):
        # to run the full code: use
        # for i, (name, ticker) in enumerate(companies.items()):
        for i, (name, ticker) in enumerate(companies.items()):
            print(f"[{i+1}/{total_companies}] Analyzing: {name} ({ticker})")
            
            price_data = get_price_data(ticker)
            if price_data is None or price_data.empty:
                print(f"  > Data not available for {ticker}. Skip.")
                continue
            
            features_df = calculate_company_features(ticker, price_data, benchmark_data)
            
            report_file.write(f"\n======================================================================\n## {name} ({ticker})\n======================================================================\n\n")

            if features_df is not None and not features_df.empty:
                report_file.write("### Historical Asset Features\n" + features_df.to_markdown(index=False) + "\n\n")
            
            report_file.write("### Strategy Performance Summary\n\n")
            
            for strategy in strategies_to_run:
                strategy_func = strategy_functions.get(strategy['func_name'])
                if not callable(strategy_func): 
                    print(f"  > WARNING: Function not found for strategy '{strategy['name']}'. Skip.")
                    continue
                
                strategy_results = []
                for regime in market_regimes:
                    regime_price_data = price_data.loc[regime['start']:regime['end']]
                    summary, _ = strategy_func(regime_price_data, strategy['params'])
                    
                    if summary:
                        strategy_results.append({
                            'Period': regime['name'], 'Market Phase': regime['label'],
                            'P&L %': f"{summary.get('P&L %', 0):.2f}%",
                            'Final Capital': f"${summary.get('Final Capital', initial_capital):,.2f}",
                            'Num. Trades': summary.get('Num. Trades', 0)
                        })
                
                report_file.write(f"#### Strategy: {strategy['name']}\n")
                if strategy_results:
                    summary_df = pd.DataFrame(strategy_results)
                    report_file.write(summary_df.to_markdown(index=False) + "\n\n")

            report_file.write("\n### Trade Details\n\n")
            
            for strategy in strategies_to_run:
                strategy_func = strategy_functions.get(strategy['func_name'])
                if not callable(strategy_func): continue

                report_file.write(f"--- \n#### Trade Details for: {strategy['name']}\n")
                for regime in market_regimes:
                    regime_price_data = price_data.loc[regime['start']:regime['end']]
                    _, trades_df = strategy_func(regime_price_data, strategy['params'])
                    
                    if trades_df is not None and not trades_df.empty:
                        report_file.write(f"*Period {regime['name']} ({regime['label']})*\n")
                        report_file.write(trades_df.to_markdown(index=False) + "\n\n")

    print("\nAnalysis complete. The file 'complete_report.txt' has been created")
