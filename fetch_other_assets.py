"""
Fetch 9 AM EST aligned historical data for other asset classes.
Fetches Sugar, Coffee, S&P 500, DAX, and Oil for expansion testing.
"""
import sys
import os
import io

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add trading directory to path
sys.path.insert(0, 'trading')

from oanda_data_fetcher import OandaDataFetcher

# Assets to test - using OANDA naming conventions
OTHER_ASSETS = [
    'SUGARUSD',      # Sugar
    'COFUSD',        # Coffee (or might be COFFEEUSD)
    'SPX500USD',     # S&P 500
    'DE30EUR',       # DAX (Germany 30)
    'BCOUSD'         # Brent Crude Oil (or WTICOUSD for WTI)
]

def fetch_asset_data(pair, count=5000):
    """Fetch historical data for a single asset"""
    print(f"\n{'='*80}")
    print(f"Fetching {pair}")
    print(f"{'='*80}")

    try:
        fetcher = OandaDataFetcher(practice=False)

        # Fetch data aligned to 9 AM EST (data fetcher already does this)
        df = fetcher.get_historical_data(
            pair=pair,
            count=count,
            granularity='D'
        )

        if df is not None and len(df) > 0:
            # Save to data directory
            output_file = f'data/{pair}_1day_oanda.csv'
            df.to_csv(output_file)

            print(f"✓ Successfully fetched {len(df)} candles")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Saved to: {output_file}")

            # Show sample of recent data
            print(f"\nRecent data sample:")
            print(df.tail(3))

            return True
        else:
            print(f"✗ Failed to fetch data for {pair}")
            return False

    except Exception as e:
        print(f"✗ Error fetching {pair}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("FETCHING OTHER ASSET CLASSES FOR BACKTEST")
    print("="*80)
    print(f"\nAssets to fetch: {', '.join(OTHER_ASSETS)}")
    print(f"Candles per asset: 5000")
    print(f"Alignment: 9 AM EST (14:00 UTC)")
    print()

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    results = {}
    for pair in OTHER_ASSETS:
        results[pair] = fetch_asset_data(pair)

    # Summary
    print(f"\n{'='*80}")
    print("FETCH SUMMARY")
    print(f"{'='*80}")

    successful = sum(results.values())
    total = len(OTHER_ASSETS)

    for pair, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {pair}")

    print(f"\nTotal: {successful}/{total} assets fetched successfully")

    if successful > 0:
        print(f"\n✓ {successful} assets ready for backtesting!")
        print("\nNext steps:")
        for pair, success in results.items():
            if success:
                # Use appropriate spread for each asset type
                if 'USD' in pair and ('SUGAR' in pair or 'COF' in pair):
                    spread = '0.05'  # Agriculture typically has wider spreads
                elif 'SPX' in pair or 'DE30' in pair:
                    spread = '0.01'  # Indices have tight spreads
                elif 'BCO' in pair or 'WTI' in pair:
                    spread = '0.03'  # Oil moderate spreads
                else:
                    spread = '0.015'  # Default
                print(f"  python backtest_oanda_data.py --pair {pair} --test-days 750 --spread-pct {spread}")
    else:
        print("\n⚠ All assets failed to fetch. Check errors above.")

    print("="*80)

if __name__ == "__main__":
    main()
