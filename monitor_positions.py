"""
Position Monitor - Checks if positions have closed and logs/notifies
Runs hourly to detect when positions are closed by stop-loss or take-profit
"""
import os
import sys
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from notification_service import NotificationService

# Load environment variables
load_dotenv()

class PositionMonitor:
    def __init__(self, pair='EURUSD', practice=True):
        self.pair = pair
        self.practice = practice

        # Load API credentials
        if practice:
            self.api_key = os.getenv('OANDA_API_KEY_PRACTICE')
            self.account_id = os.getenv('OANDA_ACCOUNT_ID_PRACTICE')
            self.base_url = "https://api-fxpractice.oanda.com/v3"
        else:
            self.api_key = os.getenv('OANDA_API_KEY')
            self.account_id = os.getenv('OANDA_ACCOUNT_ID')
            self.base_url = "https://api-fxtrade.oanda.com/v3"

        if not self.api_key or not self.account_id:
            raise ValueError("Missing OANDA API credentials in .env file")

        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # State file
        self.state_file = f'data/oanda_cache/{self.pair}_state.json'

        # Notifier
        self.notifier = NotificationService()

        # Load state
        self.load_state()

    def load_state(self):
        """Load position state from file"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.position = state.get('position', 0)
                self.entry_price = state.get('entry_price')
                self.entry_date = state.get('entry_date')
                if self.entry_date:
                    self.entry_date = datetime.fromisoformat(self.entry_date)
                self.trade_id = state.get('trade_id')
                self.position_size = state.get('position_size', 0)
        else:
            self.position = 0
            self.entry_price = None
            self.entry_date = None
            self.trade_id = None
            self.position_size = 0

    def save_state(self):
        """Save position state to file"""
        state = {
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'trade_id': self.trade_id,
            'position_size': self.position_size
        }

        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def get_instrument_name(self, pair):
        """Convert pair name to OANDA instrument format"""
        return f"{pair[:3]}_{pair[3:]}"

    def get_open_positions(self):
        """Get all open positions from OANDA"""
        instrument = self.get_instrument_name(self.pair)
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

            return None  # No open position

        except requests.exceptions.RequestException as e:
            print(f"Error fetching positions: {e}")
            return None

    def get_current_price(self):
        """Get current mid price from OANDA"""
        instrument = self.get_instrument_name(self.pair)
        url = f"{self.base_url}/accounts/{self.account_id}/pricing"

        try:
            response = requests.get(
                url,
                headers=self.headers,
                params={'instruments': instrument}
            )
            response.raise_for_status()
            result = response.json()

            if 'prices' in result and len(result['prices']) > 0:
                price_data = result['prices'][0]
                bid = float(price_data['bids'][0]['price'])
                ask = float(price_data['asks'][0]['price'])
                mid = (bid + ask) / 2
                return mid

            return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching price: {e}")
            return None

    def check_position_status(self):
        """Check if position has closed and notify"""
        print(f"\n{'='*70}")
        print(f"POSITION MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Pair: {self.pair}")
        print(f"Mode: {'PRACTICE' if self.practice else 'LIVE'}")
        print(f"{'='*70}\n")

        # If we don't think we have a position, nothing to monitor
        if self.position == 0:
            print("No position in state file - nothing to monitor")
            return

        print(f"State file shows {'LONG' if self.position == 1 else 'SHORT'} position")
        print(f"Entry: {self.entry_price:.5f} on {self.entry_date.date()}")
        print(f"Position size: ${self.position_size:.2f}")

        # Check actual OANDA position
        actual_position = self.get_open_positions()

        if actual_position:
            # Position still open - show current P&L
            print(f"\nPosition still OPEN at OANDA")
            print(f"Unrealized P&L: ${actual_position['unrealized_pl']:.2f}")

            # Get current price and calculate percentage P&L
            current_price = self.get_current_price()
            if current_price:
                if self.position == 1:
                    pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
                else:
                    pnl_pct = (self.entry_price - current_price) / self.entry_price * 100

                print(f"Current price: {current_price:.5f}")
                print(f"P&L: {pnl_pct:.2f}%")
        else:
            # Position closed! Calculate final P&L and notify
            print(f"\n🚨 POSITION CLOSED at OANDA!")

            # Get current price as exit price estimate
            exit_price = self.get_current_price()

            if exit_price:
                # Calculate P&L
                if self.position == 1:
                    pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
                else:
                    pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100

                pnl_dollars = self.position_size * (pnl_pct / 100)

                print(f"Exit price (estimated): {exit_price:.5f}")
                print(f"P&L: {pnl_pct:.2f}% (${pnl_dollars:.2f})")

                # Send notification
                direction = 'LONG' if self.position == 1 else 'SHORT'

                # Calculate days held
                days_held = (datetime.now() - self.entry_date).days if self.entry_date else 0

                self.notifier.notify_trade_exit(
                    pair=self.pair,
                    direction=direction,
                    entry_price=self.entry_price,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    pnl_dollars=pnl_dollars,
                    exit_reason="Position closed at OANDA (likely stop-loss or take-profit)",
                    days_held=days_held
                )

            # Clear state
            print("\nClearing local state file")
            self.position = 0
            self.entry_price = None
            self.entry_date = None
            self.trade_id = None
            self.position_size = 0
            self.save_state()
            print("State cleared")

        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='OANDA Position Monitor')
    parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair to monitor')
    parser.add_argument('--live', action='store_true', help='Use LIVE account')
    args = parser.parse_args()

    try:
        monitor = PositionMonitor(args.pair, practice=not args.live)
        monitor.check_position_status()

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
