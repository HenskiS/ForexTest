"""
Create alternative prediction targets for forex trading.

Current problem: Continuous return prediction leads to:
- Very few trades (~3/year)
- Sticky predictions (hold for 48+ days)
- Model predicts only 12% of actual volatility

Alternative targets to try:
1. Binary classification (up/down)
2. Multi-day returns (5-day, 10-day forward)
3. Volatility-adjusted returns
"""

import pandas as pd
import numpy as np

print("Loading FIXED data...")
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED.csv', index_col='date', parse_dates=True)

print(f"Original shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# ============================================================================
# TARGET 1: Binary Classification (Up/Down)
# ============================================================================
print("\n" + "="*70)
print("TARGET 1: Binary Classification")
print("="*70)

# Current day's return (already shifted, so this is T+1 return)
df['target_binary'] = (df['simple_return'] > 0).astype(int)

up_pct = df['target_binary'].sum() / len(df) * 100
print(f"  Up days: {df['target_binary'].sum()} ({up_pct:.1f}%)")
print(f"  Down days: {len(df) - df['target_binary'].sum()} ({100-up_pct:.1f}%)")
print(f"  Balanced: {'Yes' if 48 <= up_pct <= 52 else 'No'}")

# ============================================================================
# TARGET 2: Multi-day Returns (5-day, 10-day forward)
# ============================================================================
print("\n" + "="*70)
print("TARGET 2: Multi-day Forward Returns")
print("="*70)

# Calculate forward returns (must shift backward to maintain no-leakage)
# We want: day T features -> predict T to T+5 return
# Current simple_return is already T+1 return due to feature shift
# So we need cumulative return over next N days

# Get raw price data
df_temp = pd.read_csv('data/EURUSD_1day_with_features.csv', index_col='date', parse_dates=True)
close_prices = df_temp['close']

# Calculate forward returns
for days in [5, 10, 20]:
    # Forward return: (close[t+days] - close[t]) / close[t]
    forward_return = close_prices.pct_change(days).shift(-days)

    # Align with our FIXED data (features are already shifted)
    df[f'target_{days}day_return'] = forward_return

    # Also create binary version
    df[f'target_{days}day_binary'] = (forward_return > 0).astype(int)

    # Stats
    valid_returns = df[f'target_{days}day_return'].dropna()
    print(f"\n{days}-day forward return:")
    print(f"  Mean: {valid_returns.mean():.6f}")
    print(f"  Std:  {valid_returns.std():.6f}")
    print(f"  Min:  {valid_returns.min():.6f}")
    print(f"  Max:  {valid_returns.max():.6f}")

    valid_binary = df[f'target_{days}day_binary'].dropna()
    up_pct = valid_binary.sum() / len(valid_binary) * 100
    print(f"  Positive: {valid_binary.sum()} ({up_pct:.1f}%)")

# ============================================================================
# TARGET 3: Volatility-Adjusted Returns (Sharpe-like)
# ============================================================================
print("\n" + "="*70)
print("TARGET 3: Volatility-Adjusted Returns")
print("="*70)

# Calculate rolling 20-day volatility
rolling_vol = df['simple_return'].rolling(20).std()

# Volatility-adjusted return = return / volatility
df['target_vol_adjusted'] = df['simple_return'] / rolling_vol

valid = df['target_vol_adjusted'].dropna()
print(f"  Mean: {valid.mean():.6f}")
print(f"  Std:  {valid.std():.6f}")
print(f"  Min:  {valid.min():.6f}")
print(f"  Max:  {valid.max():.6f}")

# ============================================================================
# Save New Dataset
# ============================================================================
print("\n" + "="*70)
print("Saving data with alternative targets...")
print("="*70)

# Show summary of all targets
print("\nTarget columns created:")
print("  1. target_binary              - Binary (up=1, down=0)")
print("  2. target_5day_return         - 5-day forward return")
print("  3. target_5day_binary         - 5-day binary")
print("  4. target_10day_return        - 10-day forward return")
print("  5. target_10day_binary        - 10-day binary")
print("  6. target_20day_return        - 20-day forward return")
print("  7. target_20day_binary        - 20-day binary")
print("  8. target_vol_adjusted        - Volatility-adjusted return")

# Drop rows with NaN in any target (due to forward calculations)
df_out = df.dropna()

print(f"\nOriginal rows: {len(df)}")
print(f"After dropping NaNs: {len(df_out)}")
print(f"Rows dropped: {len(df) - len(df_out)}")

output_file = 'data/EURUSD_1day_with_features_FIXED_multitarget.csv'
df_out.to_csv(output_file)

print(f"\nSaved to: {output_file}")
print(f"Date range: {df_out.index.min()} to {df_out.index.max()}")

# ============================================================================
# Display Sample
# ============================================================================
print("\n" + "="*70)
print("Sample data (showing original + new targets):")
print("="*70)

sample = df_out[['simple_return', 'target_binary', 'target_5day_return',
                  'target_5day_binary', 'target_10day_return']].iloc[100:105]
print(sample.to_string())

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Train XGBoost with target_binary (classification)")
print("2. Train XGBoost with target_5day_return (longer horizon)")
print("3. Train XGBoost with target_10day_return (even longer)")
print("4. Compare trading frequency and profitability")
print("="*70)
