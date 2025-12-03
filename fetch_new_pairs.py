"""
Fetch 9 AM EST aligned historical data for new currency pairs.
Fetches NZDUSD, USDCAD, USDCHF, EURGBP for expansion testing.
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

# New pairs to test
NEW_PAIRS = ['NZDUSD', 'USDCAD', 'USDCHF', 'EURGBP']

def fetch_pair_data(pair, count=5000):
    """Fetch historical data for a single pair"""
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
    print("FETCHING NEW CURRENCY PAIRS FOR BACKTEST")
    print("="*80)
    print(f"\nPairs to fetch: {', '.join(NEW_PAIRS)}")
    print(f"Candles per pair: 5000")
    print(f"Alignment: 9 AM EST (14:00 UTC)")
    print()

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    results = {}
    for pair in NEW_PAIRS:
        results[pair] = fetch_pair_data(pair)

    # Summary
    print(f"\n{'='*80}")
    print("FETCH SUMMARY")
    print(f"{'='*80}")

    successful = sum(results.values())
    total = len(NEW_PAIRS)

    for pair, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {pair}")

    print(f"\nTotal: {successful}/{total} pairs fetched successfully")

    if successful == total:
        print("\n✓ All pairs ready for backtesting!")
        print("\nNext steps:")
        print("  python backtest_oanda_data.py --pair NZDUSD --leverage 2.0 --days 3125")
        print("  python backtest_oanda_data.py --pair USDCAD --leverage 2.0 --days 3125")
        print("  python backtest_oanda_data.py --pair USDCHF --leverage 2.0 --days 3125")
        print("  python backtest_oanda_data.py --pair EURGBP --leverage 2.0 --days 3125")
    else:
        print("\n⚠ Some pairs failed to fetch. Check errors above.")

    print("="*80)

if __name__ == "__main__":
    main()
