"""
Fetch historical spread data for backtesting.

This script fetches bid/ask data from OANDA and calculates spreads for
all trading pairs. The spread data is required for accurate backtesting.

Safe to run multiple times (idempotent) - will overwrite existing files.
"""
import os
import sys
import pandas as pd
from oanda_data_fetcher import OandaDataFetcher
from trading import TradingConfig


def fetch_spread_data_for_pair(fetcher, pair, count=5000):
    """
    Fetch spread data for a single pair.

    Args:
        fetcher: OandaDataFetcher instance
        pair: Currency pair
        count: Number of candles to fetch

    Returns:
        tuple: (DataFrame with spreads, error_message)
    """
    instrument = fetcher.get_instrument_name(pair)
    url = f"{fetcher.base_url}/instruments/{instrument}/candles"

    params = {
        'count': count,
        'granularity': 'D',
        'price': 'BA',  # Both Bid and Ask prices
        'alignmentTimezone': 'America/New_York',
        'dailyAlignment': 17
    }

    try:
        import requests
        response = requests.get(url, headers=fetcher.headers, params=params)
        response.raise_for_status()
        data = response.json()

        if 'candles' not in data or not data['candles']:
            return None, "No candles returned"

        rows = []
        for candle in data['candles']:
            if not candle['complete']:
                continue

            # Extract bid and ask data
            bid_data = candle['bid']
            ask_data = candle['ask']

            bid_close = float(bid_data['c'])
            ask_close = float(ask_data['c'])
            mid_close = (bid_close + ask_close) / 2
            spread = ask_close - bid_close
            spread_pips = spread * 10000
            spread_pct = spread / mid_close

            rows.append({
                'date': candle['time'],
                'open': (float(bid_data['o']) + float(ask_data['o'])) / 2,
                'high': (float(bid_data['h']) + float(ask_data['h'])) / 2,
                'low': (float(bid_data['l']) + float(ask_data['l'])) / 2,
                'close': mid_close,
                'volume': int(candle['volume']),
                'bid_close': bid_close,
                'ask_close': ask_close,
                'spread': spread,
                'spread_pips': spread_pips,
                'spread_pct': spread_pct
            })

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        return df, None

    except Exception as e:
        return None, str(e)


def fetch_all_spreads(practice=True, count=5000):
    """
    Fetch spread data for all pairs.

    Args:
        practice: Use practice account (default: True)
        count: Number of candles to fetch (default: 5000)

    Returns:
        dict: Results for each pair {pair: {'success': bool, 'error': str}}
    """
    pairs = TradingConfig.DEFAULT_PAIRS

    print("="*100)
    print("FETCHING SPREAD DATA FOR BACKTESTING")
    print("="*100)
    print(f"\nPairs: {', '.join(pairs)}")
    print(f"Candles: {count} days per pair")
    print(f"Mode: {'PRACTICE' if practice else 'LIVE'} account")
    print(f"\nNote: Spread data includes bid/ask prices for accurate backtest simulation")
    print()

    # Initialize fetcher
    try:
        fetcher = OandaDataFetcher(practice=practice)
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nTo fix this:")
        print("1. Sign up for OANDA account (fxTrade Practice for testing)")
        print("2. Generate API key from account settings")
        print("3. Add to .env file:")
        print("   OANDA_API_KEY=your_api_key_here")
        print("   OANDA_ACCOUNT_ID=your_account_id_here")
        return None

    # Create data directory
    os.makedirs('data', exist_ok=True)

    results = {}

    for i, pair in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {pair}")
        print("-" * 100)

        df, error = fetch_spread_data_for_pair(fetcher, pair, count)

        if df is None:
            results[pair] = {'success': False, 'error': error}
            print(f"  ✗ FAILED: {error}")
            continue

        # Save to CSV
        filename = f'{pair}_1day_with_spreads.csv'
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)

        # Calculate average spread
        avg_spread_pct = df['spread_pct'].mean() * 100
        avg_spread_pips = df['spread_pips'].mean()

        results[pair] = {'success': True, 'error': None}
        print(f"  ✓ SUCCESS: {len(df)} days saved to {filepath}")
        print(f"  Average spread: {avg_spread_pips:.2f} pips ({avg_spread_pct:.4f}%)")

    # Summary
    print("\n" + "="*100)
    print("SPREAD DATA FETCHING SUMMARY")
    print("="*100)

    successful = sum(1 for r in results.values() if r['success'])
    failed = len(pairs) - successful

    print(f"\nSuccessful: {successful}/{len(pairs)}")
    print(f"Failed: {failed}/{len(pairs)}")

    if successful > 0:
        print("\n✓ Successfully fetched:")
        for pair, result in results.items():
            if result['success']:
                print(f"  - {pair}")

    if failed > 0:
        print("\n✗ Failed to fetch:")
        for pair, result in results.items():
            if not result['success']:
                print(f"  - {pair}: {result['error']}")

    print()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fetch spread data for backtesting')
    parser.add_argument('--live', action='store_true', help='Use LIVE account instead of practice')
    parser.add_argument('--count', type=int, default=5000,
                       help='Number of candles to fetch (default: 5000)')
    args = parser.parse_args()

    results = fetch_all_spreads(practice=not args.live, count=args.count)

    # Exit with error code if any failed
    if results is None:
        sys.exit(1)

    failed_count = sum(1 for r in results.values() if not r['success'])
    if failed_count > 0:
        sys.exit(1)

    sys.exit(0)
