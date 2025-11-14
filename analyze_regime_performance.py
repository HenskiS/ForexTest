"""
Analyze which market conditions favor selective vs unselective thresholds.
"""

import pandas as pd
import numpy as np

# Load EURUSD data
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

# Focus on years with divergent threshold performance
years_of_interest = [2007, 2008, 2010, 2020]

for year in years_of_interest:
    year_data = df[df.index.year == year]

    if len(year_data) == 0:
        continue

    # Calculate regime characteristics
    volatility = year_data['atr'].mean()
    price_range = (year_data['close'].max() - year_data['close'].min()) / year_data['close'].mean()

    # Trend strength: how directional was the year?
    returns = year_data['close'].pct_change()
    abs_return = abs(year_data['close'].iloc[-1] / year_data['close'].iloc[0] - 1)

    # Choppiness: ratio of cumulative abs returns to net return
    cumulative_abs_moves = returns.abs().sum()
    net_move = abs_return
    choppiness = cumulative_abs_moves / net_move if net_move > 0 else 999

    # Volatility regime
    high_vol_days = (year_data['atr'] > year_data['atr'].median()).sum()
    vol_regime = high_vol_days / len(year_data)

    print(f"\n{'='*60}")
    print(f"YEAR {year}")
    print(f"{'='*60}")
    print(f"Average ATR (volatility):     {volatility:.6f}")
    print(f"Price range (% of mean):      {price_range*100:.2f}%")
    print(f"Net move:                     {abs_return*100:.2f}%")
    print(f"Choppiness ratio:             {choppiness:.2f}")
    print(f"High vol regime (%):          {vol_regime*100:.1f}%")

    # Interpret
    print(f"\nInterpretation:")
    if choppiness > 15:
        print(f"  - VERY CHOPPY market (ratio: {choppiness:.1f})")
        print(f"  - Many false breakouts, whipsaws")
        print(f"  - Extreme signals likely noisy -> LESS selective wins")
    elif choppiness < 10:
        print(f"  - TRENDING market (ratio: {choppiness:.1f})")
        print(f"  - Clean directional moves")
        print(f"  - Extreme signals capture big moves -> MORE selective wins")
    else:
        print(f"  - MIXED market (ratio: {choppiness:.1f})")

    if volatility > 0.010:
        print(f"  - HIGH volatility (ATR: {volatility:.6f})")
    else:
        print(f"  - NORMAL volatility (ATR: {volatility:.6f})")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print("""
2008: Financial crisis - expect high choppiness, extreme signals noisy
      -> Less selective (40/60) should win

2020: COVID crash/recovery - expect strong trend, extreme signals valid
      -> More selective (5/95) should win

2010: Post-crisis QE - expect trending, extreme signals capture moves
      -> More selective (5/95) should win

2007: Pre-crisis buildup - trending market
      -> More selective should win
""")
