import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ticker_symbol = 'AAPL'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
bb_length = 20  # Period for the moving average and standard deviation
bb_std = 2.0    # Number of standard deviations

ticker_obj = yf.Ticker(ticker_symbol)
data = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    print("No data found. Check ticker or date range.")
    exit()

# Calculate Bollinger Bands
data.ta.bbands(length=bb_length, std=bb_std, append=True)

lower_band_col = f'BBL_{bb_length}_{bb_std}'
middle_band_col = f'BBM_{bb_length}_{bb_std}'
upper_band_col = f'BBU_{bb_length}_{bb_std}'
print(data[['Close', lower_band_col, middle_band_col, upper_band_col]].dropna().tail(10))

# Format date strings for file naming
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Plotting
plt.figure(figsize=(12, 6))
# Price
plt.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.9, linewidth=1)
# Bollinger bands
plt.plot(data.index, data[upper_band_col], label='Upper Band', color='red', linestyle='--', linewidth=1)
plt.plot(data.index, data[middle_band_col], label='Middle Band (SMA)', color='black', linestyle='-.', linewidth=1, alpha=0.7)
plt.plot(data.index, data[lower_band_col], label='Lower Band', color='green', linestyle='--', linewidth=1)
plt.fill_between(data.index, data[lower_band_col], data[upper_band_col], color='gray', alpha=0.1)

plt.title(f"{ticker_symbol} Price and Bollinger Bands \nFrom {start_str} to {end_str}", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save plot
filename = f"{ticker_symbol}_BollingerBands_from_{start_str}_to_{end_str}.png"
plt.savefig(filename, dpi=300)
plt.close()

print(f"\nPlot saved successfully as {filename}")
