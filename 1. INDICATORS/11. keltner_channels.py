import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ticker_symbol = 'AAPL'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
kc_length = 20
kc_multiplier = 2.0

ticker_obj = yf.Ticker(ticker_symbol)
data = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    print("No data found. Check ticker or date range.")
    exit()

# Calculate Keltner Channels
data.ta.kc(length=kc_length, scalar=kc_multiplier, append=True)

lower_channel_col = data.columns[-3]
middle_channel_col = data.columns[-2]
upper_channel_col = data.columns[-1]
print(data[['Close', lower_channel_col, middle_channel_col, upper_channel_col]].dropna().tail(10))

# Format date strings for file naming
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Plotting
plt.figure(figsize=(12, 6))
# Price
plt.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.9, linewidth=1)
# Keltner Channels
plt.plot(data.index, data[upper_channel_col], label='Upper Channel', color='red', linestyle='--', linewidth=1)
plt.plot(data.index, data[middle_channel_col], label='Middle Line (EMA)', color='black', linestyle='-.', linewidth=1, alpha=0.7)
plt.plot(data.index, data[lower_channel_col], label='Lower Channel', color='green', linestyle='--', linewidth=1)
plt.fill_between(data.index, data[lower_channel_col], data[upper_channel_col], color='gray', alpha=0.1)

plt.title(f"{ticker_symbol} Price and Keltner Channels \nFrom {start_str} to {end_str}", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save plot
filename = f"{ticker_symbol}_KeltnerChannels_from_{start_str}_to_{end_str}.png"
plt.savefig(filename, dpi=300)
plt.close()

print(f"\nPlot saved successfully as {filename}")
