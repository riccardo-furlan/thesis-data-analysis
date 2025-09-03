import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ticker_symbol = 'AAPL'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
fast_period = 12
slow_period = 26
signal_period = 9

ticker_obj = yf.Ticker(ticker_symbol)
data = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    print("No data found. Check ticker or date range.")
    exit()

# Calculate MACD
data.ta.macd(fast=fast_period, slow=slow_period, signal=signal_period, append=True)

macd_line_col = f'MACD_{fast_period}_{slow_period}_{signal_period}'
macd_histogram_col = f'MACDh_{fast_period}_{slow_period}_{signal_period}'
macd_signal_col = f'MACDs_{fast_period}_{slow_period}_{signal_period}'

print(data[['Close', macd_line_col, macd_signal_col, macd_histogram_col]].dropna().tail(10))

# Format date strings for file naming
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Plotting with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(f"{ticker_symbol} Price and MACD\nFrom {start_str} to {end_str}", fontsize=16)

# Plotting Price (top subplot)
ax1.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=1.5)
ax1.set_ylabel("Price (USD)")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Plotting MACD (bottom subplot)
ax2.plot(data.index, data[macd_line_col], label='MACD Line', color='blue', linewidth=1)
ax2.plot(data.index, data[macd_signal_col], label='Signal Line', color='red', linestyle='--', linewidth=1)
colors = ['green' if val >= 0 else 'red' for val in data[macd_histogram_col]]
ax2.bar(data.index, data[macd_histogram_col], label='Histogram', color=colors, width=0.7, alpha=0.6)
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_ylabel("MACD")
ax2.set_xlabel("Date")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Adjust layout for better appearance
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save plot
filename = f"{ticker_symbol}_MACD_from_{start_str}_to_{end_str}.png"
plt.savefig(filename, dpi=300)
plt.close()

print(f"\nPlot saved successfully as {filename}")
