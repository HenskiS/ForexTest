"""
OANDA Data Fetcher for Forex Markets
Fetches historical data from OANDA API to ensure consistency with live trading.

IMPORTANT: Use OANDA data for training to match production execution prices.
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

load_dotenv()


class OandaDataFetcher:
    """Fetches forex data from OANDA v20 REST API"""

    # OANDA instrument names (different from standard format)
    INSTRUMENT_MAP = {
        'EURUSD': 'EUR_USD',
        'GBPUSD': 'GBP_USD',
        'USDJPY': 'USD_JPY',
        'AUDUSD': 'AUD_USD',
        'USDCAD': 'USD_CAD',
        'NZDUSD': 'NZD_USD',
        'USDCHF': 'USD_CHF',
    }

    def __init__(self, practice=True):
        """
        Initialize OANDA data fetcher.

        Args:
            practice: If True, use practice (demo) account. If False, use live account.
        """
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')

        if not self.api_key:
            raise ValueError("OANDA_API_KEY not found in .env file. Get your API key from OANDA account settings.")

        if not self.account_id:
            raise ValueError("OANDA_ACCOUNT_ID not found in .env file")

        # Use practice or live endpoints
        if practice:
            self.base_url = "https://api-fxpractice.oanda.com/v3"
        else:
            self.base_url = "https://api-fxtrade.oanda.com/v3"

        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        self.practice = practice
        print(f"OANDA API initialized ({'PRACTICE' if practice else 'LIVE'} mode)")

    def get_instrument_name(self, pair):
        """Convert standard pair name (EURUSD) to OANDA format (EUR_USD)"""
        if pair in self.INSTRUMENT_MAP:
            return self.INSTRUMENT_MAP[pair]

        # Try auto-conversion (insert underscore in middle)
        if len(pair) == 6:
            return f"{pair[:3]}_{pair[3:]}"

        raise ValueError(f"Unknown pair: {pair}. Use standard format like 'EURUSD'")

    def get_historical_data(self, pair, count=756, granularity='D'):
        """
        Fetch historical candle data from OANDA.

        Args:
            pair: Forex pair (e.g., 'EURUSD', 'GBPUSD')
            count: Number of candles to fetch (max 5000 per request)
            granularity: Candle timeframe
                'D' = Daily
                'H1' = 1 hour
                'H4' = 4 hours
                'M15' = 15 minutes
                etc.

        Returns:
            DataFrame with OHLCV data
        """
        instrument = self.get_instrument_name(pair)

        # OANDA limits to 5000 candles per request
        if count > 5000:
            print(f"Warning: Requested {count} candles, but OANDA max is 5000. Fetching 5000.")
            count = 5000

        url = f"{self.base_url}/instruments/{instrument}/candles"
        params = {
            'count': count,
            'granularity': granularity,
            'price': 'M',  # Midpoint prices (average of bid/ask)
        }

        try:
            print(f"Fetching {pair} ({instrument}) data: {count} {granularity} candles...")
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            if 'candles' not in data:
                print(f"Unexpected response format: {data}")
                return pd.DataFrame()

            candles = data['candles']

            if not candles:
                print(f"No candles returned for {pair}")
                return pd.DataFrame()

            # Parse candles into DataFrame
            rows = []
            for candle in candles:
                if not candle['complete']:
                    continue  # Skip incomplete candles

                rows.append({
                    'date': candle['time'],
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })

            df = pd.DataFrame(rows)

            if df.empty:
                print(f"No complete candles found for {pair}")
                return df

            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            print(f"Successfully fetched {len(df)} candles")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from OANDA: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return pd.DataFrame()

    def get_historical_data_range(self, pair, from_date, to_date=None, granularity='D'):
        """
        Fetch historical data for a specific date range.

        Args:
            pair: Forex pair (e.g., 'EURUSD')
            from_date: Start date (datetime or string 'YYYY-MM-DD')
            to_date: End date (datetime or string 'YYYY-MM-DD'). Defaults to today.
            granularity: Candle timeframe ('D', 'H1', 'H4', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        instrument = self.get_instrument_name(pair)

        # Convert dates to RFC3339 format (required by OANDA)
        if isinstance(from_date, str):
            from_date = datetime.strptime(from_date, '%Y-%m-%d')

        if to_date is None:
            to_date = datetime.now()
        elif isinstance(to_date, str):
            to_date = datetime.strptime(to_date, '%Y-%m-%d')

        from_str = from_date.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
        to_str = to_date.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')

        url = f"{self.base_url}/instruments/{instrument}/candles"
        params = {
            'from': from_str,
            'to': to_str,
            'granularity': granularity,
            'price': 'M',  # Midpoint prices
        }

        try:
            print(f"Fetching {pair} data from {from_date.date()} to {to_date.date()} ({granularity})...")
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            if 'candles' not in data:
                print(f"Unexpected response format: {data}")
                return pd.DataFrame()

            candles = data['candles']

            # Parse candles
            rows = []
            for candle in candles:
                if not candle['complete']:
                    continue

                rows.append({
                    'date': candle['time'],
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })

            df = pd.DataFrame(rows)

            if df.empty:
                print(f"No complete candles found")
                return df

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            print(f"Successfully fetched {len(df)} candles")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return pd.DataFrame()

    def get_current_price(self, pair):
        """
        Get current bid/ask prices for a pair.

        Args:
            pair: Forex pair (e.g., 'EURUSD')

        Returns:
            dict with 'bid', 'ask', 'mid', 'spread', 'time'
        """
        instrument = self.get_instrument_name(pair)

        url = f"{self.base_url}/accounts/{self.account_id}/pricing"
        params = {'instruments': instrument}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            if 'prices' not in data or not data['prices']:
                print(f"No pricing data returned for {pair}")
                return None

            price_data = data['prices'][0]

            bid = float(price_data['bids'][0]['price'])
            ask = float(price_data['asks'][0]['price'])
            mid = (bid + ask) / 2
            spread = ask - bid

            return {
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'spread': spread,
                'spread_pips': spread * 10000,  # Convert to pips
                'time': price_data['time']
            }

        except requests.exceptions.RequestException as e:
            print(f"Error fetching current price: {e}")
            return None

    def save_data(self, df, filename):
        """Save DataFrame to CSV in data/ folder"""
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def load_data(self, filename):
        """Load DataFrame from CSV in data/ folder"""
        filepath = os.path.join('data', filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            print(f"File {filepath} not found")
            return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='OANDA Data Fetcher')
    parser.add_argument('--live', action='store_true', help='Use LIVE account instead of practice')
    parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair (e.g., EURUSD, GBPUSD)')
    parser.add_argument('--count', type=int, default=5000, help='Number of candles to fetch (max 5000)')
    args = parser.parse_args()

    PAIR = args.pair.upper()

    # Fetch OANDA data
    print(f"OANDA Data Fetcher - {PAIR}")
    print("=" * 70)

    try:
        # Initialize fetcher (practice mode by default, unless --live flag)
        fetcher = OandaDataFetcher(practice=not args.live)

        if args.live:
            print("Using LIVE account (read-only, fetching data only)")
            print()

        # Test 1: Get current price
        print(f"\nCurrent {PAIR} Price")
        print("-" * 70)
        price = fetcher.get_current_price(PAIR)
        if price:
            print(f"Bid: {price['bid']:.5f}")
            print(f"Ask: {price['ask']:.5f}")
            print(f"Mid: {price['mid']:.5f}")
            print(f"Spread: {price['spread_pips']:.1f} pips")
            print(f"Time: {price['time']}")

        # Test 2: Get historical daily data
        print(f"\n\nHistorical Daily Data (last {args.count} days)")
        print("-" * 70)
        df = fetcher.get_historical_data(PAIR, count=args.count, granularity='D')

        if not df.empty:
            print(f"\nFirst 5 rows:")
            print(df.head())
            print(f"\nLast 5 rows:")
            print(df.tail())
            print(f"\nData shape: {df.shape}")

            # Save to CSV
            filename = f'{PAIR}_1day_oanda.csv'
            fetcher.save_data(df, filename)

    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nTo fix this:")
        print("1. Sign up for OANDA account (fxTrade Practice for testing)")
        print("2. Generate API key from account settings")
        print("3. Add to .env file:")
        print("   OANDA_API_KEY=your_api_key_here")
        print("   OANDA_ACCOUNT_ID=your_account_id_here")
