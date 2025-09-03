import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ticker_symbol = 'AAPL'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
base_fast = 12
base_slow = 26
signal_period = 9
atr_period = 14

data = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    print("No data found. Check the ticker symbol or date range.")
    exit()

# Calculate ATR
high_low = data['High'] - data['Low']
high_close = np.abs(data['High'] - data['Close'].shift())
low_close = np.abs(data['Low'] - data['Close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
atr = tr.rolling(atr_period).mean()

# Adaptivity based on Normalized ATR
atr_scaled = atr / atr.rolling(window=atr_period).mean()
atr_scaled = atr_scaled.clip(lower=0.5, upper=1.5)

# Calculate Adaptive Periods
adaptive_fast = (base_fast * atr_scaled).round().fillna(base_fast)
adaptive_slow = (base_slow * atr_scaled).round().fillna(base_slow)

def adaptive_ema(data_series, periods_series):
    ema_values = np.full(len(data_series), np.nan)
    alpha_series = 2 / (periods_series + 1)

    first_valid_label = periods_series.first_valid_index()
    if first_valid_label is None:
        return pd.Series(ema_values, index=data_series.index)
    
    start_pos = data_series.index.get_loc(first_valid_label)
    
    ema_values[start_pos] = data_series.iloc[start_pos]

    for i in range(start_pos + 1, len(data_series)):
        close_price = data_series.iloc[i]
        prev_ema = ema_values[i-1]
        alpha = alpha_series.iloc[i]
        
        ema_values[i] = (close_price * alpha) + (prev_ema * (1 - alpha))
            
    return pd.Series(ema_values, index=data_series.index)

# Calculate Adaptive MACD
fast_ema = adaptive_ema(data['Close'], adaptive_fast)
slow_ema = adaptive_ema(data['Close'], adaptive_slow)

data['MACD'] = fast_ema - slow_ema
data['Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
data['Histogram'] = data['MACD'] - data['Signal']

print(data[['Close', 'MACD', 'Signal', 'Histogram']].dropna().tail(10))

# Format date strings for file naming
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Plotting with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(f"{ticker_symbol} Price and Adaptive MACD\nFrom {start_str} to {end_str}", fontsize=16)

# Plotting Price (top subplot)
ax1.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=1.5)
ax1.set_ylabel("Price (USD)")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Plotting Adaptive MACD (bottom subplot)
ax2.plot(data.index, data['MACD'], label='Adaptive MACD', color='blue', linewidth=1)
ax2.plot(data.index, data['Signal'], label='Signal Line', color='red', linestyle='--', linewidth=1)
colors = ['green' if val >= 0 else 'red' for val in data['Histogram']]
ax2.bar(data.index, data['Histogram'], label='Histogram', color=colors, width=0.7, alpha=0.6)
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_ylabel("Adaptive MACD")
ax2.set_xlabel("Date")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Adjust layout for better appearance
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save plot
filename = f"{ticker_symbol}_AdaptiveMACD_from_{start_str}_to_{end_str}.png"
plt.savefig(filename, dpi=300)
plt.close()

print(f"\nPlot saved successfully as {filename}")
