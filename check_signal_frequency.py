import pandas as pd

pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
days = 1000

print("Signal Frequency Analysis")
print("=" * 60)

for pair in pairs:
    df = pd.read_csv(f'{pair}_backtest_trades_{days}days.csv')
    signal_rate = len(df) / days * 100
    print(f'{pair}: {len(df):>4} trades over {days} days = {signal_rate:>5.1f}% signal rate')

print("\n" + "=" * 60)
print("Expected Simultaneous Positions:")
print("=" * 60)

# If each pair trades ~98% of days independently
# Probability of exactly N pairs having signals on same day
avg_rate = 0.98  # Average ~98% signal rate

import math
from scipy.stats import binom

# Assuming independence (conservative - they're actually correlated)
n_pairs = 4
print(f"\nAssuming {avg_rate*100:.0f}% independent signal rate per pair:")
for k in range(n_pairs + 1):
    prob = binom.pmf(k, n_pairs, avg_rate)
    print(f"  {k} pairs with signals: {prob*100:>5.1f}%")

print("\n" + "=" * 60)
print("Reality Check:")
print("=" * 60)
print("With 98% signal rate per pair, you'll have signals on nearly")
print("ALL pairs almost every day. The 48/52 percentile thresholds")
print("generate signals ~96% of the time.")
print("\nSo you're right - you'll typically have 3-4 positions, not 1-2!")
print("\nRevised position sizing recommendation:")
print("  Use 2:1 leverage ($250 per pair) for safety with 4 positions")
print("  Or use 2.5:1 leverage ($312.50 per pair) for moderate risk")
