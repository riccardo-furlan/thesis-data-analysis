import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ticker_symbol = 'AAPL'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
rsi_period = 14
ma_period = 14

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

# Calculate SMA
data['RSI_SMA'] = data['RSI'].rolling(window=ma_period).mean()

print(data[['Close', 'RSI', 'RSI_SMA']].dropna().tail(10))

# Format date strings for file naming
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Plotting with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(f"{ticker_symbol} Price and RSI ({rsi_period}) with SMA ({ma_period})\nFrom {start_str} to {end_str}", fontsize=16)

# Plotting Price (top subplot)
ax1.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=1.5)
ax1.set_ylabel("Price (USD)")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Plotting RSI with SMA (bottom subplot)
ax2.plot(data.index, data['RSI'], label=f'RSI ({rsi_period})', color='purple', linewidth=1.5)
ax2.plot(data.index, data['RSI_SMA'], label=f'SMA ({ma_period})', color='orange', linestyle='--', linewidth=1.5)
ax2.axhline(70, color='gray', linestyle='--', linewidth=1)
ax2.axhline(50, color='gray', linestyle='--', linewidth=1)
ax2.axhline(30, color='gray', linestyle='--', linewidth=1)
ax2.set_ylabel("RSI")
ax2.set_xlabel("Date")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Adjust layout for better appearance
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save plot
filename = f"{ticker_symbol}_RSI{rsi_period}_SMA{ma_period}_from_{start_str}_to_{end_str}.png"
plt.savefig(filename, dpi=300)
plt.close()

print(f"\nPlot saved successfully as {filename}")
