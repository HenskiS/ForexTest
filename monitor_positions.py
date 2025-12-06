"""
Multi-Pair Position Monitor - Checks if positions have closed and logs/notifies
Runs hourly to detect when positions are closed by stop-loss or take-profit
Also syncs local state with OANDA to catch desynchronization
"""
import sys
from datetime import datetime
from trading import (
    OandaClient,
    PositionManager,
    NotificationService
)


class MultiPairPositionMonitor:
    """Monitor positions across multiple currency pairs"""

    def __init__(self, pairs, practice=True):
        """
        Initialize multi-pair position monitor.

        Args:
            pairs: List of currency pairs to monitor
            practice: Use practice account if True
        """
        self.pairs = pairs
        self.practice = practice

        # Initialize per-pair components
        self.clients = {}
        self.position_managers = {}

        for pair in pairs:
            self.clients[pair] = OandaClient(pair, practice=practice)
            self.position_managers[pair] = PositionManager(pair)

        # Single notifier for all pairs
        self.notifier = NotificationService()

    def sync_and_monitor_positions(self):
        """Sync state with OANDA and monitor all positions"""
        print(f"\n{'='*70}")
        print(f"MULTI-PAIR POSITION MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Pairs: {', '.join(self.pairs)}")
        print(f"Mode: {'PRACTICE' if self.practice else 'LIVE'}")
        print(f"{'='*70}\n")

        for pair in self.pairs:
            print(f"\n{pair}:")
            print("-" * 70)
            self.check_pair_status(pair)

        print(f"\n{'='*70}")
        print("MONITOR CHECK COMPLETE")
        print(f"{'='*70}\n")

    def check_pair_status(self, pair):
        """Check and sync position status for a single pair"""
        pm = self.position_managers[pair]
        client = self.clients[pair]

        # Always check OANDA for actual positions (not just when local state says we have one)
        actual_position = client.get_open_positions()
        local_has_position = pm.has_position()

        if local_has_position and not actual_position:
            # Case 1: Local says open, OANDA says closed (stop/TP hit)
            print(f"Local state shows open position, but none at OANDA")
            print(f"  {'LONG' if pm.position == 1 else 'SHORT'} @ {pm.entry_price:.5f}")
            print(f"  Position likely closed by stop-loss or take-profit")

            # Get current price to estimate exit
            current_price_data = client.fetcher.get_current_price(pair)
            if current_price_data:
                exit_price = current_price_data['mid']

                # Calculate P&L
                pnl = pm.calculate_pnl(exit_price)
                if pnl:
                    print(f"  Estimated P&L: {pnl['pnl_pct']:.2f}% (${pnl['pnl_dollars']:.2f})")

                    # Send notification
                    days_held = (datetime.now().date() - pm.entry_date.date()).days if pm.entry_date else 0

                    self.notifier.notify_trade_exit(
                        pair=pair,
                        direction='LONG' if pm.position == 1 else 'SHORT',
                        entry_price=pm.entry_price,
                        exit_price=exit_price,
                        pnl_pct=pnl['pnl_pct'],
                        pnl_dollars=pnl['pnl_dollars'],
                        exit_reason='STOP_LOSS or TAKE_PROFIT',
                        days_held=days_held
                    )

            # Clear local state
            pm.close_position()
            print(f"  Local state cleared")

        elif not local_has_position and actual_position:
            # Case 2: Local says closed, OANDA says open (state desync)
            print(f"OANDA shows open position, but local state shows none!")
            print(f"  Position at OANDA: {actual_position}")
            print(f"  Syncing local state to match OANDA...")

            # Extract position info
            if actual_position.get('long_units', 0) > 0:
                direction = 1
                units = actual_position['long_units']
            elif actual_position.get('short_units', 0) < 0:
                direction = -1
                units = abs(actual_position['short_units'])
            else:
                print(f"  WARNING: Could not determine position direction")
                return

            # Get trade details from OANDA
            trades = client.get_open_trades()
            if trades and len(trades) > 0:
                trade = trades[0]
                entry_price = trade['price']
                # OANDA returns nanoseconds, Python only handles microseconds - truncate to 6 decimal places
                time_str = trade['openTime'].replace('Z', '+00:00')
                # Truncate fractional seconds to 6 digits (microseconds)
                if '.' in time_str:
                    parts = time_str.split('.')
                    time_str = parts[0] + '.' + parts[1][:6] + parts[1][9:]  # Keep first 6 digits of fractional seconds
                entry_time = datetime.fromisoformat(time_str)

                # Calculate position size
                instrument = client.fetcher.get_instrument_name(pair)
                if instrument.startswith('USD_'):
                    position_size = units
                else:
                    position_size = units * entry_price

                # Update local state
                pm.open_position(
                    direction=direction,
                    entry_price=entry_price,
                    position_size=position_size,
                    trade_id=trade['id']
                )

                # Override entry_date with actual time from OANDA
                pm.entry_date = entry_time
                pm.save_state()

                print(f"  Successfully synced: {'LONG' if direction == 1 else 'SHORT'} @ {entry_price:.5f}")
                print(f"  Entry time: {entry_time}")
                print(f"  Position size: ${position_size:.2f}")
                print(f"  Unrealized P&L: ${actual_position['unrealized_pl']:.2f}")
            else:
                print(f"  ERROR: Could not fetch trade details from OANDA")

        elif local_has_position and actual_position:
            # Case 3: Both agree there's a position - show current status
            print(f"Position OPEN: {'LONG' if pm.position == 1 else 'SHORT'} @ {pm.entry_price:.5f}")
            print(f"  Entry date: {pm.entry_date.date() if pm.entry_date else 'Unknown'}")
            print(f"  Position size: ${pm.position_size:.2f}")
            print(f"  Unrealized P&L: ${actual_position['unrealized_pl']:.2f}")

            # Get current price and calculate percentage P&L
            current_price_data = client.fetcher.get_current_price(pair)
            if current_price_data:
                current_price = current_price_data['mid']
                pnl = pm.calculate_pnl(current_price)
                if pnl:
                    print(f"  Current price: {current_price:.5f}")
                    print(f"  P&L: {pnl['pnl_pct']:.2f}%")

        else:
            # Case 4: Both agree there's no position
            print(f"No open position")


if __name__ == "__main__":
    import argparse

    # Default pairs (same as multi-pair trader)
    DEFAULT_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']

    parser = argparse.ArgumentParser(
        description='OANDA Multi-Pair Position Monitor - Sync state and detect closed positions'
    )
    parser.add_argument(
        '--pair',
        type=str,
        help='Currency pair to monitor (if not specified, monitors all default pairs)'
    )
    parser.add_argument(
        '--pairs',
        nargs='+',
        help='Multiple currency pairs to monitor (alternative to --pair)'
    )
    parser.add_argument('--live', action='store_true', help='Use LIVE account (default: practice)')
    args = parser.parse_args()

    # Determine which pairs to monitor
    if args.pair:
        pairs = [args.pair.upper()]
    elif args.pairs:
        pairs = [p.upper() for p in args.pairs]
    else:
        pairs = DEFAULT_PAIRS

    try:
        monitor = MultiPairPositionMonitor(pairs, practice=not args.live)
        monitor.sync_and_monitor_positions()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

        # Send error notification
        try:
            notifier = NotificationService()
            notifier.notify_error(f"Position monitor error: {str(e)}")
        except:
            pass

        sys.exit(1)
