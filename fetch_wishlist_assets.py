"""
Fetch the 4 wishlist assets: Sugar, S&P 500, DAX, and WTI Oil.
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

# Wishlist assets with their correct OANDA names
WISHLIST_ASSETS = {
    'SUGARUSD': 'SUGAR_USD',
    'SPX500USD': 'SPX500_USD',
    'DE30EUR': 'DE30_EUR',
    'WTICOUSD': 'WTICO_USD'
}

NAMES = {
    'SUGARUSD': 'Sugar',
    'SPX500USD': 'S&P 500',
    'DE30EUR': 'DAX (Germany 30)',
    'WTICOUSD': 'WTI Crude Oil'
}

def fetch_asset_data(pair, oanda_name, count=5000):
    """Fetch historical data for a single asset"""
    print(f"\n{'='*80}")
    print(f"Fetching {NAMES[pair]}")
    print(f"{'='*80}")

    try:
        fetcher = OandaDataFetcher(practice=False)

        # Override the instrument name mapping
        fetcher.INSTRUMENT_MAP[pair] = oanda_name

        # Fetch data aligned to 9 AM EST
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
    print("FETCHING WISHLIST ASSETS FOR BACKTEST")
    print("="*80)
    print(f"\nAssets to fetch:")
    for pair, oanda_name in WISHLIST_ASSETS.items():
        print(f"  {NAMES[pair]:20} ({oanda_name})")
    print(f"\nCandles per asset: 5000")
    print(f"Alignment: 9 AM EST (14:00 UTC)")
    print()

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    results = {}
    for pair, oanda_name in WISHLIST_ASSETS.items():
        results[pair] = fetch_asset_data(pair, oanda_name)

    # Summary
    print(f"\n{'='*80}")
    print("FETCH SUMMARY")
    print(f"{'='*80}")

    successful = sum(results.values())
    total = len(WISHLIST_ASSETS)

    for pair, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {NAMES[pair]}")

    print(f"\nTotal: {successful}/{total} assets fetched successfully")

    if successful > 0:
        print(f"\n✓ {successful} assets ready for backtesting!")
        print("\nNext steps:")

        # Suggested spreads for each asset type
        spreads = {
            'SUGARUSD': '0.05',      # Agricultural - wider spreads
            'SPX500USD': '0.01',     # Major index - tight spreads
            'DE30EUR': '0.01',       # Major index - tight spreads
            'WTICOUSD': '0.03'       # Oil - moderate spreads
        }

        for pair, success in results.items():
            if success:
                spread = spreads.get(pair, '0.015')
                print(f"  python backtest_oanda_data.py --pair {pair} --test-days 750 --spread-pct {spread}")
    else:
        print("\n⚠ All assets failed to fetch. Check errors above.")

    print("="*80)

if __name__ == "__main__":
    main()
