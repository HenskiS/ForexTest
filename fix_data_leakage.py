"""
Fix data leakage: Shift features forward by 1 day.

Current (WRONG): Use day T's features (including close) to predict day T's return
Fixed (RIGHT): Use day T's features to predict day T+1's return

This way, we're using today's technical indicators to predict tomorrow's return,
which is how you'd actually trade in real life.
"""

import pandas as pd
import numpy as np

print("Loading data...")
df = pd.read_csv('data/EURUSD_1day_with_features.csv', index_col='date', parse_dates=True)

print(f"Original data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Feature columns (all technical indicators)
FEATURE_COLS = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]

print(f"\nFeatures to shift: {len(FEATURE_COLS)}")

# Create a copy
df_fixed = df.copy()

# Shift all features FORWARD by 1 day
# This means: day T's features will now align with day T+1's return
for col in FEATURE_COLS:
    df_fixed[col] = df_fixed[col].shift(1)

print("\nShifted features forward by 1 day")
print("Now: Day T features -> Day T+1 return")

# Drop the first row (has NaN features after shift)
df_fixed = df_fixed.dropna()

print(f"\nAfter dropping NaNs: {df_fixed.shape}")
print(f"Rows dropped: {df.shape[0] - df_fixed.shape[0]}")

# Save fixed data
output_file = 'data/EURUSD_1day_with_features_FIXED.csv'
df_fixed.to_csv(output_file)

print(f"\nFixed data saved to: {output_file}")
print(f"Date range: {df_fixed.index.min()} to {df_fixed.index.max()}")

# Verify the fix with an example
print("\n" + "="*60)
print("VERIFICATION (showing day T features -> day T+1 return)")
print("="*60)

# Show a few rows
sample = df_fixed[['open', 'close', 'simple_return', 'ema_10', 'ema_20', 'rsi']].iloc[100:105]
print(sample)

print("\n" + "="*60)
print("EXPLANATION:")
print("="*60)
print("On 2000-XX-XX (row 100):")
print("  - Features: EMA, RSI calculated from data UP TO this day")
print("  - Target: simple_return is NEXT day's return")
print("  - Trading: You see today's indicators, predict tomorrow, trade at tomorrow's open")
print("\nThis is the correct way to prevent data leakage!")
