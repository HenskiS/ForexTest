"""
Fetch historical data with 9 AM EST alignment for all pairs.

This refetches data with daily candles aligned to 9 AM EST instead of 5 PM,
which allows us to trade during London/NY overlap with better spreads while
keeping predictions matched to execution timing.
"""
from oanda_data_fetcher import OandaDataFetcher
import os

PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
COUNT = 5000  # Fetch maximum history

print("="*80)
print("FETCHING 9 AM ALIGNED DATA")
print("="*80)
print(f"Pairs: {', '.join(PAIRS)}")
print(f"Alignment: 9 AM EST (better spreads)")
print(f"Candles: {COUNT}")
print("="*80)

fetcher = OandaDataFetcher(practice=False)  # Use same account as live trading

for pair in PAIRS:
    print(f"\n{pair}:")
    print("-" * 80)

    # Fetch with 9 AM alignment
    df = fetcher.get_historical_data(pair, count=COUNT, granularity='D', daily_alignment=9)

    if df.empty:
        print(f"  ERROR: No data fetched for {pair}")
        continue

    # Save to new file (don't overwrite old data yet)
    output_file = f'data/{pair}_1day_oanda_9am.csv'

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    df.to_csv(output_file, index=False)
    print(f"  Saved {len(df)} candles to {output_file}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

print("\n" + "="*80)
print("DATA FETCH COMPLETE")
print("="*80)
print("\nNext steps:")
print("1. Compare new 9 AM data with old 5 PM data")
print("2. Retrain models with 9 AM data")
print("3. Run backtest with 9 AM data + better spreads (0.013%)")
print("4. If results are good, replace old data and update production")
print("="*80)
