import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('data/EURUSD_1day_oanda.csv')

# Calculate ATR (14-period)
df['tr'] = df[['high', 'low', 'close']].apply(
    lambda row: max(
        row['high'] - row['low'],
        abs(row['high'] - df['close'].shift(1).loc[row.name]) if pd.notna(df['close'].shift(1).loc[row.name]) else row['high'] - row['low'],
        abs(row['low'] - df['close'].shift(1).loc[row.name]) if pd.notna(df['close'].shift(1).loc[row.name]) else row['high'] - row['low']
    ),
    axis=1
)

df['atr'] = df['tr'].rolling(window=14).mean()

# Calculate day d+1's high and low relative to day d's close
df['next_high'] = df['high'].shift(-1)
df['next_low'] = df['low'].shift(-1)
df['close_to_next_high'] = df['next_high'] - df['close']
df['close_to_next_low'] = df['next_low'] - df['close']

# Remove rows with NaN values
df = df.dropna().copy()

# Create the plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Next day's high relative to today's close vs ATR
axes[0].scatter(df['atr'], df['close_to_next_high'], alpha=0.5, s=10)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1)
axes[0].set_xlabel('Day d ATR (14-period)', fontsize=12)
axes[0].set_ylabel('Day d+1 High - Day d Close', fontsize=12)
axes[0].set_title('Next Day High (relative to today\'s close) vs ATR', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df['atr'], df['close_to_next_high'], 1)
p = np.poly1d(z)
axes[0].plot(df['atr'], p(df['atr']), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
axes[0].legend()

# Plot 2: Next day's low relative to today's close vs ATR
axes[1].scatter(df['atr'], df['close_to_next_low'], alpha=0.5, s=10, color='orange')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
axes[1].set_xlabel('Day d ATR (14-period)', fontsize=12)
axes[1].set_ylabel('Day d+1 Low - Day d Close', fontsize=12)
axes[1].set_title('Next Day Low (relative to today\'s close) vs ATR', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Add trend line
z2 = np.polyfit(df['atr'], df['close_to_next_low'], 1)
p2 = np.poly1d(z2)
axes[1].plot(df['atr'], p2(df['atr']), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z2[0]:.4f}x+{z2[1]:.4f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('atr_vs_next_day_prices.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'atr_vs_next_day_prices.png'")

# Print some statistics
print("\nStatistics:")
print(f"Correlation between ATR and next day's high (rel. to close): {df['atr'].corr(df['close_to_next_high']):.4f}")
print(f"Correlation between ATR and next day's low (rel. to close): {df['atr'].corr(df['close_to_next_low']):.4f}")
print(f"\nAverage next day high above close: {df['close_to_next_high'].mean():.5f}")
print(f"Average next day low below close: {df['close_to_next_low'].mean():.5f}")
print(f"\nAverage ATR: {df['atr'].mean():.5f}")

plt.show()
