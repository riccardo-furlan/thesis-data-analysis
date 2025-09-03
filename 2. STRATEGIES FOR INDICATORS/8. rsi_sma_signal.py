import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

ticker_symbol = 'AAPL'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
rsi_period = 5
signal_period = 5
initial_capital = 10000.0
trade_size = 1000.0 # Fixed amount for each trade

data = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    print("No data found. Check ticker or date range.")
    exit()

# Calculate RSI
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)

avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Calculate SMA of RSI (Signal Line)
data['RSI_SMA'] = data['RSI'].rolling(window=signal_period).mean()
data.dropna(inplace=True) # Remove rows with NaN values due to calculation

# BACKTEST EXECUTION

close_prices = data['Close'].to_numpy()
rsi_values = data['RSI'].to_numpy()
signal_values = data['RSI_SMA'].to_numpy()
dates = data.index

trades = []
position_open = False
capital = initial_capital
capital_series = [initial_capital]
entry_price = 0.0

# Start from 1 to compare each day with the previous one (i-1)
for i in range(1, len(close_prices)):
    current_close = close_prices[i]
    current_rsi = rsi_values[i]
    prev_rsi = rsi_values[i-1]
    current_signal = signal_values[i]
    prev_signal = signal_values[i-1]

    # Check crossover conditions
    is_long_signal = (prev_rsi <= prev_signal) and (current_rsi > current_signal) # bullish crossover
    is_exit_signal = (prev_rsi >= prev_signal) and (current_rsi < current_signal) # bearish crossover

    # Trade Opening (LONG ONLY)
    if not position_open:
        # If there's a LONG signal, open a buy position
        if is_long_signal:
            position_open = True
            position_type = 'LONG'
            entry_price = float(current_close)
            entry_date = dates[i]
            asset_quantity = trade_size / entry_price
        
    # Trade Closing
    elif position_open:
        # A LONG position is only closed by a SHORT signal or at the end of the data
        if position_type == 'LONG' and (is_exit_signal or i == len(close_prices) - 1):
            exit_price = float(current_close)
            exit_date = dates[i]
            
            pnl_dollars = (exit_price - entry_price) * asset_quantity
            pnl_percent = (pnl_dollars / trade_size) * 100
            capital += pnl_dollars
            
            trades.append({
                'Open Date': entry_date.strftime('%Y-%m-%d'),
                'Open Price': round(entry_price, 2),
                'Close Date': exit_date.strftime('%Y-%m-%d'),
                'Close Price': round(exit_price, 2),
                'Trade Type': position_type,
                'P&L ($)': round(pnl_dollars, 2),
                'P&L (%)': round(pnl_percent, 2),
                'Current Capital': round(capital, 2)
            })
            position_open = False
    
    capital_series.append(capital)

# CALCULATE FINAL STATISTICS

final_capital = capital
pnl_total_dollars = final_capital - initial_capital
pnl_total_percent = (pnl_total_dollars / initial_capital) * 100 if initial_capital > 0 else 0
max_capital = max(capital_series) if capital_series else initial_capital
min_capital = min(capital_series) if capital_series else initial_capital

# CREATE REPORT CSV FILE

start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')
filename = f"report_{ticker_symbol}_RSI_SMA_Signal_{rsi_period}-{signal_period}_LongOnly.csv"

try:
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        f.write("--- Backtest Summary ---\n")
        f.write(f"Indicator Used: RSI({rsi_period}) with SMA Signal Line({signal_period})\n")
        f.write("Strategy Implemented: RSI/Signal Line Crossover (Long Only)\n")
        f.write(f"Asset Traded: {ticker_symbol}\n")
        f.write(f"Period: {start_str} to {end_str}\n")
        f.write("\n")
        f.write("--- Performance ---\n")
        f.write(f"Initial Capital: ${initial_capital:,.2f}\n")
        f.write(f"Final Capital: ${final_capital:,.2f}\n")
        f.write(f"Total P&L: ${pnl_total_dollars:,.2f} ({pnl_total_percent:.2f}%)\n")
        f.write(f"Maximum Capital Reached: ${max_capital:,.2f}\n")
        f.write(f"Minimum Capital Reached: ${min_capital:,.2f}\n")
        f.write("\n")
        f.write("--- Trade Details ---\n")
        
        # Create and write the trades table
        if trades:
            df_trades = pd.DataFrame(trades)
            df_trades.insert(0, 'Trade Number', range(1, 1 + len(df_trades)))
            df_trades.to_csv(f, index=False, header=True)
        else:
            f.write("No trades executed during the period.\n")

    # PRINT PERFORMANCE TO TERMINAL

    print("\n--- Backtest Performance Summary ---")
    print(f"Asset: {ticker_symbol} | Strategy: RSI with SMA Signal Line (Long Only)")
    print(f"Analysis Period: {start_str} to {end_str}")
    print("-" * 40)
    print(f"{'Initial Capital:':<25} ${initial_capital:,.2f}")
    print(f"{'Final Capital:':<25} ${final_capital:,.2f}")
    print(f"{'Total P&L:':<25} ${pnl_total_dollars:,.2f} ({pnl_total_percent:.2f}%)")
    print("-" * 40)
    print(f"{'Maximum Capital Reached:':<25} ${max_capital:,.2f}")
    print(f"{'Minimum Capital Reached:':<25} ${min_capital:,.2f}")
    print(f"{'Number of Trades:':<25} {len(trades)}")
    print("-" * 40)

    print(f"\nDetailed CSV report saved as: {filename}")

except IOError as e:
    print(f"\nError writing file: {e}")
