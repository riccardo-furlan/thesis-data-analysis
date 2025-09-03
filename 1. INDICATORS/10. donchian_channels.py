import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ticker_symbol = 'AAPL'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
dc_period = 20

ticker_obj = yf.Ticker(ticker_symbol)
data = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    print("No data found. Check ticker or date range.")
    exit()

# Calculate Donchian Channels
data.ta.donchian(lower_length=dc_period, upper_length=dc_period, append=True)

lower_channel_col = f'DCL_{dc_period}_{dc_period}'
middle_channel_col = f'DCM_{dc_period}_{dc_period}'
upper_channel_col = f'DCU_{dc_period}_{dc_period}'
print(data[['Close', lower_channel_col, middle_channel_col, upper_channel_col]].dropna().tail(10))

# Format date strings for file naming
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Plotting
plt.figure(figsize=(12, 6))
# Price
plt.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.9, linewidth=1)
# Donchian channels
plt.plot(data.index, data[upper_channel_col], label='Upper Channel', color='red', linestyle='--', linewidth=1)
plt.plot(data.index, data[middle_channel_col], label='Middle Line', color='black', linestyle='-.', linewidth=1, alpha=0.7)
plt.plot(data.index, data[lower_channel_col], label='Lower Channel', color='green', linestyle='--', linewidth=1)
plt.fill_between(data.index, data[lower_channel_col], data[upper_channel_col], color='gray', alpha=0.1)

plt.title(f"{ticker_symbol} Price and Donchian Channels \nFrom {start_str} to {end_str}", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save plot
filename = f"{ticker_symbol}_DonchianChannels_from_{start_str}_to_{end_str}.png"
plt.savefig(filename, dpi=300)
plt.close()

print(f"\nPlot saved successfully as {filename}")
