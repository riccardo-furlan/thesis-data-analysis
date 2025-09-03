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

# Calculate Zero-Lag MACD
# Calculate ZLMA (Zero-Lag Moving Average)
fast_zlma = data.ta.zlma(length=fast_period)
slow_zlma = data.ta.zlma(length=slow_period)

# Calculate the ZL MACD line (difference between the two ZLMAs)
data['ZLMACD'] = fast_zlma - slow_zlma

# Calculate signal line (EMA of the ZL MACD line)
data['ZLMACD_signal'] = data['ZLMACD'].ewm(span=signal_period, adjust=False).mean()

# Calculate histogram
data['ZLMACD_histogram'] = data['ZLMACD'] - data['ZLMACD_signal']

print(data[['Close', 'ZLMACD', 'ZLMACD_signal', 'ZLMACD_histogram']].dropna().tail(10))

# Format date strings for file naming
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Plotting with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(f"{ticker_symbol} Price and Zero-Lag MACD\nFrom {start_str} to {end_str}", fontsize=16)

# Plotting Price (top subplot)
ax1.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=1.5)
ax1.set_ylabel("Price (USD)")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Plotting Zero-Lag MACD (bottom subplot)
# ZL MACD line and signal line
ax2.plot(data.index, data['ZLMACD'], label='ZL MACD Line', color='blue', linewidth=1)
ax2.plot(data.index, data['ZLMACD_signal'], label='Signal Line', color='red', linestyle='--', linewidth=1)
# Histogram (colored bars)
colors = ['green' if val >= 0 else 'red' for val in data['ZLMACD_histogram']]
ax2.bar(data.index, data['ZLMACD_histogram'], label='Histogram', color=colors, width=0.7, alpha=0.6)
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_ylabel("ZL MACD")
ax2.set_xlabel("Date")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Adjust layout for better appearance
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save plot
filename = f"{ticker_symbol}_ZLMACD_from_{start_str}_to_{end_str}.png"
plt.savefig(filename, dpi=300)
plt.close()

print(f"\nPlot saved successfully as {filename}")