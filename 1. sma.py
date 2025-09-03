import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ticker_symbol = 'AAPL'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
sma_period = 9

data = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    print("No data found. Check ticker or date range.")
    exit()

# Calculate SMA
data['SMA'] = data['Close'].rolling(window=sma_period).mean()

print(data[['Close', 'SMA']].tail(10))

# Format date strings for file naming
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price', color='blue')
plt.plot(data.index, data['SMA'], label=f'{sma_period}-Day SMA', color='orange')
plt.title(f"{ticker_symbol} {sma_period}-Day SMA\nFrom {start_str} to {end_str}")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

# Save plot
filename = f"{ticker_symbol}_SMA{sma_period}_from_{start_str}_to_{end_str}.png"
plt.savefig(filename, dpi=300)
plt.close()

print(f"\nPlot saved successfully as: {filename}")