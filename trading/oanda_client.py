"""
OANDA API Client

Handles all communication with OANDA API including:
- Order placement (market orders with stop loss and take profit)
- Position management (close positions)
- Account information (balance, open positions)
- Market data fetching (via OandaDataFetcher)
"""
import os
import requests
from dotenv import load_dotenv
from oanda_data_fetcher import OandaDataFetcher

load_dotenv()


class OandaClient:
    """OANDA API client for trading operations"""

    def __init__(self, pair, practice=True):
        """
        Initialize OANDA client.

        Args:
            pair: Forex pair to trade (e.g., 'EURUSD')
            practice: If True, use practice account. If False, use live account.
        """
        self.pair = pair.upper()
        self.practice = practice

        # Initialize data fetcher
        self.fetcher = OandaDataFetcher(practice=practice)

        # API credentials
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')

        if not self.api_key or not self.account_id:
            raise ValueError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set in .env")

        # Set API endpoint
        if practice:
            self.base_url = "https://api-fxpractice.oanda.com/v3"
        else:
            self.base_url = "https://api-fxtrade.oanda.com/v3"

        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def get_account_balance(self):
        """
        Get available account balance from OANDA.

        Returns:
            float: Account balance in dollars, or None if error
        """
        url = f"{self.base_url}/accounts/{self.account_id}/summary"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            balance = float(result['account']['balance'])
            print(f"Account balance: ${balance:.2f}")
            return balance

        except requests.exceptions.RequestException as e:
            print(f"Error fetching account balance: {e}")
            return None

    def get_open_positions(self):
        """
        Get all open positions from OANDA for this pair.

        Returns:
            dict: Position info (instrument, units, P&L) or None if no position
        """
        instrument = self.fetcher.get_instrument_name(self.pair)
        url = f"{self.base_url}/accounts/{self.account_id}/openPositions"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            # Check if there's an open position for this instrument
            for position in result.get('positions', []):
                if position['instrument'] == instrument:
                    long_units = float(position['long']['units'])
                    short_units = float(position['short']['units'])

                    if long_units != 0 or short_units != 0:
                        return {
                            'instrument': instrument,
                            'long_units': long_units,
                            'short_units': short_units,
                            'unrealized_pl': float(position.get('unrealizedPL', 0))
                        }

            return None  # No open position for this instrument

        except requests.exceptions.RequestException as e:
            print(f"Error fetching open positions: {e}")
            return None

    def place_order(self, signal, current_price, position_size_dollars, stop_loss_pct, take_profit_pct):
        """
        Place market order with stop loss and take profit.

        Args:
            signal: 1 for long, -1 for short
            current_price: Current market price
            position_size_dollars: Position size in dollars
            stop_loss_pct: Stop loss as decimal (e.g., 0.0018 for 0.18%)
            take_profit_pct: Take profit as decimal (e.g., 0.0200 for 2.00%)

        Returns:
            dict: Order fill info with keys 'success', 'entry_price', 'trade_id', or None if failed
        """
        instrument = self.fetcher.get_instrument_name(self.pair)

        # SAFETY CHECK: Verify no existing position
        existing_position = self.get_open_positions()
        if existing_position:
            print(f"\nWARNING: Open position already exists at OANDA!")
            print(f"  Long units: {existing_position['long_units']}")
            print(f"  Short units: {existing_position['short_units']}")
            print(f"  Unrealized P&L: ${existing_position['unrealized_pl']:.2f}")
            print(f"  Skipping new order to prevent duplicate trade")
            return None

        # Calculate units based on position size
        # For forex pairs, units represent base currency amount
        # EUR/USD: 1 unit = 1 EUR (~$1.05), so units = dollars / price
        # USD/JPY: 1 unit = 1 USD ($1.00), so units = dollars directly
        # GBP/USD: 1 unit = 1 GBP (~$1.27), so units = dollars / price

        if instrument.startswith('USD_'):
            # USD is base currency (USD/JPY, USD/CHF, USD/CAD)
            # 1 unit = $1, so units = desired dollar exposure
            units = int(position_size_dollars)
        else:
            # USD is quote currency (EUR/USD, GBP/USD, AUD/USD, NZD/USD)
            # 1 unit = 1 base currency, so units = dollars / price
            units = int(position_size_dollars / current_price)

        # Determine price precision based on instrument
        # JPY pairs use 3 decimals, others use 5
        if 'JPY' in instrument:
            price_precision = 3
        else:
            price_precision = 5

        # Calculate stop/target prices
        if signal == 1:  # Long
            stop_price = current_price * (1 - stop_loss_pct)
            target_price = current_price * (1 + take_profit_pct)
            order_units = abs(units)
        else:  # Short
            stop_price = current_price * (1 + stop_loss_pct)
            target_price = current_price * (1 - take_profit_pct)
            order_units = -abs(units)

        # Prepare order with correct precision
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(order_units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price": f"{stop_price:.{price_precision}f}"
                },
                "takeProfitOnFill": {
                    "price": f"{target_price:.{price_precision}f}"
                }
            }
        }

        url = f"{self.base_url}/accounts/{self.account_id}/orders"

        try:
            print(f"\nPlacing {'LONG' if signal == 1 else 'SHORT'} order:")
            print(f"  Units: {order_units}")
            print(f"  Entry: {current_price:.5f}")
            print(f"  Stop Loss: {stop_price:.5f} ({stop_loss_pct*100:.2f}%)")
            print(f"  Take Profit: {target_price:.5f} ({take_profit_pct*100:.2f}%)")

            response = requests.post(url, headers=self.headers, json=order_data)
            response.raise_for_status()
            result = response.json()

            if 'orderFillTransaction' in result:
                fill = result['orderFillTransaction']
                entry_price = float(fill['price'])
                trade_id = fill['id']

                print(f"Order filled at {entry_price:.5f}")
                print(f"Position size: ${position_size_dollars:.2f}")
                print(f"Trade ID: {trade_id}")

                return {
                    'success': True,
                    'entry_price': entry_price,
                    'trade_id': trade_id,
                    'stop_price': stop_price,
                    'target_price': target_price
                }
            else:
                print(f"Order not filled: {result}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error placing order: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None

    def close_position(self, position_direction):
        """
        Close current position via OANDA API.

        Args:
            position_direction: 1 for long, -1 for short

        Returns:
            bool: True if successful, False otherwise
        """
        instrument = self.fetcher.get_instrument_name(self.pair)

        # Close all positions for this instrument
        url = f"{self.base_url}/accounts/{self.account_id}/positions/{instrument}/close"

        data = {
            "longUnits": "ALL" if position_direction == 1 else "NONE",
            "shortUnits": "ALL" if position_direction == -1 else "NONE"
        }

        try:
            response = requests.put(url, headers=self.headers, json=data)
            response.raise_for_status()

            print(f"Position closed successfully")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Error closing position: {e}")

            # If 404, the position was already closed at OANDA (likely hit stop/TP)
            if hasattr(e.response, 'status_code') and e.response.status_code == 404:
                print("Position not found at OANDA - likely already closed by stop/TP")
                return True  # State should be synced

            return False

    def fetch_latest_data(self, count=500):
        """
        Fetch latest OHLC data for this pair.

        Args:
            count: Number of candles to fetch

        Returns:
            DataFrame: OHLC data with date index
        """
        return self.fetcher.get_historical_data(
            self.pair,
            count=count,
            granularity='D',
            daily_alignment=9  # 9 AM EST (14:00 UTC) - aligns with trading schedule
        )
