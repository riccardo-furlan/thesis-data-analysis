import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

ticker_symbol = 'AAPL'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

alma_window = 9    # time period (length of the moving average)
alma_sigma = 6      # controls the smoothness of the moving average by adjusting the spread of the Gaussian weighting function.
alma_offset = 0.85  # balance between reactivity and stability (determines the position of the Gaussian peak within the window)

data = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    print("No data found. Check ticker or date range.")
    exit()

# ALMA funciotion
def alma(series, window, sigma, offset):
    m = offset * (window - 1)
    s = window / sigma
    weights = np.exp(-((np.arange(window) - m) ** 2) / (2 * s ** 2))
    weights /= weights.sum()
    return series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)

# Calculate ALMA
data['ALMA'] = alma(data['Close'], alma_window, alma_sigma, alma_offset)

print(data[['Close', 'ALMA']].tail(10))

# Format date strings for file naming
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price', color='blue')
plt.plot(data.index, data['ALMA'], label=f'ALMA ({alma_window}, {alma_sigma}, {alma_offset})', color='orange')
plt.title(f"{ticker_symbol} ALMA\nFrom {start_str} to {end_str}")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

# Save plot
filename = f"{ticker_symbol}_ALMA{alma_window}_from_{start_str}_to_{end_str}.png"
plt.savefig(filename, dpi=300)
plt.close()

print(f"\nPlot saved successfully as: {filename}")