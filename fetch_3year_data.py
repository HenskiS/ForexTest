"""
Fetch 3 years of data for multiple pairs and timeframes
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_fetcher import FMPDataFetcher

fetcher = FMPDataFetcher()

pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
timeframes = ['1hour', '4hour', '1day']

print("="*80)
print("Fetching 3 years of data for all pairs and timeframes")
print("This may take a few minutes due to API rate limits...")
print("="*80)

for pair in pairs:
    for tf in timeframes:
        print(f"\nFetching {pair} {tf}...")
        data = fetcher.get_historical_data(pair, timeframe=tf)

        if not data.empty:
            filename = f"{pair}_{tf}.csv"
            fetcher.save_data(data, filename)
            print(f"  [OK] Saved {len(data)} candles ({data['date'].min()} to {data['date'].max()})")
        else:
            print(f"  [FAIL] Failed to fetch data")

print("\n" + "="*80)
print("Data fetching complete! All files saved to data/ folder")
print("="*80)
