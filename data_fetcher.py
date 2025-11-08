"""
FMP Data Fetcher for Forex Markets
Fetches historical forex data from Financial Modeling Prep API
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


class FMPDataFetcher:
    """Fetches forex data from FMP API"""

    def __init__(self):
        self.api_key = os.getenv('FMP_API_KEY')
        if not self.api_key:
            raise ValueError("FMP_API_KEY not found in environment variables")
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_forex_pairs(self):
        """Get available forex pairs"""
        url = f"{self.base_url}/symbol/available-forex-currency-pairs"
        params = {'apikey': self.api_key}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forex pairs: {e}")
            return []

    def get_historical_data(self, symbol, from_date=None, to_date=None, timeframe='15min'):
        """
        Fetch historical forex data

        Args:
            symbol: Forex pair (e.g., 'EURUSD', 'GBPUSD')
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            timeframe: '1min', '5min', '15min', '30min', '1hour', '4hour', '1day'

        Returns:
            DataFrame with OHLCV data
        """
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')

        if not from_date:
            # Default to 3 years of data for better backtesting
            from_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')

        # FMP uses different endpoint for intraday vs daily
        if timeframe == '1day':
            url = f"{self.base_url}/historical-price-full/{symbol}"
            params = {
                'apikey': self.api_key,
                'from': from_date,
                'to': to_date
            }
        else:
            url = f"{self.base_url}/historical-chart/{timeframe}/{symbol}"
            params = {
                'apikey': self.api_key,
                'from': from_date,
                'to': to_date
            }

        try:
            print(f"Fetching {symbol} data from {from_date} to {to_date} ({timeframe})...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Handle different response formats
            if isinstance(data, dict) and 'historical' in data:
                df = pd.DataFrame(data['historical'])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                print(f"Unexpected response format: {data}")
                return pd.DataFrame()

            if df.empty:
                print(f"No data returned for {symbol}")
                return df

            # Standardize column names
            df.columns = df.columns.str.lower()

            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # Ensure we have required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Warning: Missing column {col}")

            print(f"Successfully fetched {len(df)} candles")
            return df[required_cols] if all(col in df.columns for col in required_cols) else df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def save_data(self, df, filename):
        """Save DataFrame to CSV"""
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def load_data(self, filename):
        """Load DataFrame from CSV"""
        filepath = os.path.join('data', filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            print(f"File {filepath} not found")
            return pd.DataFrame()


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = FMPDataFetcher()

    # Fetch EUR/USD data
    df = fetcher.get_historical_data('EURUSD', timeframe='1hour')
    if not df.empty:
        print(df.head())
        print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
        fetcher.save_data(df, 'EURUSD_1hour.csv')