import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ticker_symbol = 'AAPL'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
ema_period = 9

data = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    print("No data found. Check ticker or date range.")
    exit()

# Calculate EMA
data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

print(data[['Close', 'EMA']].tail(10))

# Format date strings for file naming
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price', color='blue')
plt.plot(data.index, data['EMA'], label=f'{ema_period}-Day EMA', color='orange')
plt.title(f"{ticker_symbol} {ema_period}-Day EMA\nFrom {start_str} to {end_str}")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

# Save plot
filename = f"{ticker_symbol}_EMA{ema_period}_from_{start_str}_to_{end_str}.png"
plt.savefig(filename, dpi=300)
plt.close()

print(f"\nPlot saved successfully as: {filename}")