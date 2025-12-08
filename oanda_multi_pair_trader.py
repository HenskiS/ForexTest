"""
OANDA Multi-Pair Production Trading Bot

Manages 4-pair diversified portfolio strategy:
- EURUSD, GBPUSD, AUDUSD, USDJPY
- Equal capital allocation per pair (25% each)
- 2:1 leverage ($250 position per pair on $500 account)
- Independent signal generation per pair
- Portfolio-level risk management

Based on backtest results:
- 36.22% annual return
- 68.6% of days have at least 1 win
- Low correlation (0.220) = good diversification

WARNING: This bot trades real money. Test thoroughly on practice account first!
"""
import sys
import argparse
from datetime import datetime
import pytz

from trading import (
    TradingConfig,
    OandaClient,
    PositionManager,
    TradingModel,
    is_forex_market_open,
    NotificationService
)


class MultiPairTrader:
    """Manages trading across multiple currency pairs"""

    def __init__(self, pairs, practice=True, leverage=2.0, dry_run=False, model_type='xgboost'):
        """
        Initialize multi-pair trader.

        Args:
            pairs: List of currency pairs to trade
            practice: Use practice account if True
            leverage: Leverage multiplier (e.g., 2.0 = 2:1 leverage)
            dry_run: Simulate only, don't place actual trades
            model_type: 'xgboost' or 'ann' (default: 'xgboost')
        """
        self.pairs = pairs
        self.practice = practice
        self.leverage = leverage
        self.dry_run = dry_run
        self.model_type = model_type.lower()

        # Initialize per-pair components
        self.clients = {}
        self.position_managers = {}
        self.models = {}

        for pair in pairs:
            self.clients[pair] = OandaClient(pair, practice=practice)
            self.position_managers[pair] = PositionManager(pair)
            self.models[pair] = TradingModel(pair, model_type=self.model_type)

        # Single notifier for all pairs
        self.notifier = NotificationService()

    def get_account_balance(self):
        """Get account balance (same for all pairs)"""
        # Just use the first pair's client
        return self.clients[self.pairs[0]].get_account_balance()

    def sync_positions(self):
        """Sync all position states with OANDA"""
        print("\n" + "="*70)
        print("SYNCING POSITION STATES WITH OANDA")
        print("="*70)

        for pair in self.pairs:
            pm = self.position_managers[pair]

            if self.dry_run:
                # In dry-run mode, just report local state
                if pm.has_position():
                    print(f"{pair}: Has open position (dry-run mode)")
                else:
                    print(f"{pair}: No open position (dry-run mode)")
                continue

            # Always check OANDA for actual positions (not just when local state says we have one)
            actual_position = self.clients[pair].get_open_positions()
            local_has_position = pm.has_position()

            if local_has_position and not actual_position:
                # Case 1: Local says open, OANDA says closed (stop/TP hit)
                print(f"\n{pair}: State shows open position, but none at OANDA")
                print(f"  Clearing local state (position likely closed by stop/TP)")
                pm.close_position()

            elif not local_has_position and actual_position:
                # Case 2: Local says closed, OANDA says open (state desync - this was the bug!)
                print(f"\n{pair}: OANDA shows open position, but local state shows none!")
                print(f"  Position at OANDA: {actual_position}")
                print(f"  Syncing local state to match OANDA...")

                # Extract position info from OANDA
                # actual_position has keys: 'long_units', 'short_units', 'unrealized_pl'
                if actual_position.get('long_units', 0) > 0:
                    direction = 1
                    units = actual_position['long_units']
                elif actual_position.get('short_units', 0) < 0:
                    direction = -1
                    units = abs(actual_position['short_units'])
                else:
                    print(f"  WARNING: Could not determine position direction, skipping sync")
                    continue

                # Get trade details (entry price, entry time) from OANDA
                trades = self.clients[pair].get_open_trades()
                if trades and len(trades) > 0:
                    # Use the first trade (there should only be one per instrument in our strategy)
                    trade = trades[0]
                    entry_price = trade['price']
                    # OANDA returns nanoseconds, Python only handles microseconds - truncate to 6 decimal places
                    time_str = trade['openTime'].replace('Z', '+00:00')
                    # Truncate fractional seconds to 6 digits (microseconds)
                    if '.' in time_str:
                        parts = time_str.split('.')
                        time_str = parts[0] + '.' + parts[1][:6] + parts[1][9:]  # Keep first 6 digits of fractional seconds
                    entry_time = datetime.fromisoformat(time_str)

                    # Calculate position size based on units and entry price
                    # For USD pairs (USD_JPY), units = dollars
                    # For other pairs (EUR_USD, GBP_USD, AUD_USD), units * price = dollars
                    instrument = self.clients[pair].fetcher.get_instrument_name(pair)
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
                    print(f"  Position exists but cannot fully sync - will prevent duplicate trades")

            elif local_has_position and actual_position:
                # Case 3: Both agree there's a position
                print(f"{pair}: Position synced (open)")

            else:
                # Case 4: Both agree there's no position
                print(f"{pair}: No open position")

    def process_pair_exits(self):
        """Check and exit positions that have reached holding period"""
        print("\n" + "="*70)
        print("CHECKING POSITIONS FOR TIME-BASED EXITS")
        print("="*70)

        for pair in self.pairs:
            pm = self.position_managers[pair]
            client = self.clients[pair]

            if not pm.has_position():
                print(f"\n{pair}: No position to check")
                continue

            print(f"\n{pair}: Current position")
            print(f"  Direction: {'LONG' if pm.position == 1 else 'SHORT'}")
            print(f"  Entry: {pm.entry_price:.5f}")
            print(f"  Entry date: {pm.entry_date.date()}")

            # Check if should exit by time
            should_exit = pm.should_exit_by_time(TradingConfig.HOLDING_PERIOD_DAYS)

            if should_exit:
                print(f"  Time-based exit triggered (held {TradingConfig.HOLDING_PERIOD_DAYS} day)")

                # Get current price
                current_price_data = client.fetcher.get_current_price(pair)
                if not current_price_data:
                    print(f"  ERROR: Failed to get current price")
                    continue

                current_price = current_price_data['mid']
                print(f"  Current price: {current_price:.5f}")

                # Calculate P&L
                pnl = pm.calculate_pnl(current_price)
                print(f"  P&L: {pnl['pnl_pct']:.2f}% (${pnl['pnl_dollars']:.2f})")

                if not self.dry_run:
                    # Close position at OANDA
                    success = client.close_position(pm.position)
                    if success:
                        # Log trade (will get prediction when generating signals)
                        pm.log_trade(
                            exit_price=current_price,
                            exit_reason='TIME_EXIT',
                            prediction=None  # We'll have prediction when entering new
                        )

                        # Send notification
                        self.notifier.notify_trade_exit(
                            pair=pair,
                            direction='LONG' if pm.position == 1 else 'SHORT',
                            entry_price=pm.entry_price,
                            exit_price=current_price,
                            pnl_pct=pnl['pnl_pct'],
                            pnl_dollars=pnl['pnl_dollars'],
                            exit_reason='TIME_EXIT',
                            days_held=(datetime.now().date() - pm.entry_date.date()).days
                        )

                        pm.close_position()
                        print(f"  Position closed successfully")
                    else:
                        print(f"  ERROR: Failed to close position")
                else:
                    print(f"  [DRY RUN] Would close position")
                    pm.close_position()
            else:
                print(f"  Holding (not yet {TradingConfig.HOLDING_PERIOD_DAYS} day old)")

    def train_all_models(self):
        """Train models for all pairs"""
        print("\n" + "="*70)
        print("TRAINING MODELS FOR ALL PAIRS")
        print("="*70)

        trained_models = {}

        for pair in self.pairs:
            print(f"\n{pair}:")
            print("-" * 70)

            # Fetch data
            df = self.clients[pair].fetch_latest_data(count=TradingConfig.TRAIN_WINDOW_SIZE + 300)
            if df.empty:
                print(f"  ERROR: Failed to fetch data")
                continue

            df = df.set_index('date')
            print(f"  Data: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

            # Train model
            model_obj, scaler, df_clean = self.models[pair].train(df)

            # Generate prediction
            prediction = self.models[pair].predict(df_clean)

            trained_models[pair] = {
                'prediction': prediction,
                'df_clean': df_clean
            }

        return trained_models

    def process_pair_entries(self, trained_models, account_balance):
        """Generate signals and enter new positions for all pairs"""
        print("\n" + "="*70)
        print("GENERATING SIGNALS AND ENTERING POSITIONS")
        print("="*70)

        # Calculate per-pair position size
        # With $500 account and 2:1 leverage, we use $125 capital per pair = $250 position
        capital_per_pair = account_balance / len(self.pairs)
        position_size_per_pair = capital_per_pair * self.leverage

        print(f"\nAccount balance: ${account_balance:.2f}")
        print(f"Capital per pair: ${capital_per_pair:.2f}")
        print(f"Position size per pair: ${position_size_per_pair:.2f} ({self.leverage:.1f}x leverage)")

        for pair in self.pairs:
            pm = self.position_managers[pair]
            client = self.clients[pair]

            if pair not in trained_models:
                print(f"\n{pair}: Skipping (no trained model)")
                continue

            if pm.has_position():
                print(f"\n{pair}: Already in position, skipping entry")
                continue

            print(f"\n{pair}:")
            print("-" * 70)

            prediction = trained_models[pair]['prediction']

            # Generate signal
            signal = self.models[pair].generate_signal(prediction)

            if signal == 0:
                print(f"  No trade signal (HOLD)")
                continue

            # Get current price
            current_price_data = client.fetcher.get_current_price(pair)
            if not current_price_data:
                print(f"  ERROR: Failed to get current price")
                continue

            current_price = current_price_data['mid']
            print(f"  Current price: {current_price:.5f}")

            # Use model-specific take profit
            if self.model_type == 'ann':
                take_profit_pct = TradingConfig.ANN_TAKE_PROFIT_PCT
            else:
                take_profit_pct = TradingConfig.TAKE_PROFIT_PCT

            if self.dry_run:
                print(f"  [DRY RUN] Would place {'LONG' if signal == 1 else 'SHORT'} order:")
                print(f"    Position size: ${position_size_per_pair:.2f}")
                print(f"    Stop Loss: {TradingConfig.STOP_LOSS_PCT*100:.2f}%")
                print(f"    Take Profit: {take_profit_pct*100:.2f}%")
            else:
                # Place order
                order_result = client.place_order(
                    signal=signal,
                    current_price=current_price,
                    position_size_dollars=position_size_per_pair,
                    stop_loss_pct=TradingConfig.STOP_LOSS_PCT,
                    take_profit_pct=take_profit_pct
                )

                if order_result and order_result['success']:
                    # Update position manager
                    pm.open_position(
                        direction=signal,
                        entry_price=order_result['entry_price'],
                        position_size=position_size_per_pair,
                        trade_id=order_result['trade_id']
                    )

                    # Send notification
                    self.notifier.notify_trade_entry(
                        pair=pair,
                        direction='LONG' if signal == 1 else 'SHORT',
                        entry_price=order_result['entry_price'],
                        position_size=position_size_per_pair,
                        stop_loss=order_result['stop_price'],
                        take_profit=order_result['target_price']
                    )

                    print(f"  [OK] Trade executed successfully")
                else:
                    print(f"  [FAIL] Trade execution failed")

    def run_daily_update(self):
        """Execute daily multi-pair trading workflow"""
        print(f"\n{'='*70}")
        print(f"MULTI-PAIR DAILY UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Pairs: {', '.join(self.pairs)}")
        print(f"Model: {self.model_type.upper()}")
        print(f"Mode: {'DRY RUN' if self.dry_run else ('PRACTICE' if self.practice else 'LIVE')}")
        print(f"Leverage: {self.leverage:.1f}x")
        print(f"{'='*70}")

        # Check if market is open
        if not is_forex_market_open():
            et_tz = pytz.timezone('America/New_York')
            now_et = datetime.now(et_tz)
            print(f"\nForex market is CLOSED (current time: {now_et.strftime('%A %Y-%m-%d %H:%M:%S %Z')})")
            print(f"Market hours: Sunday 5:00 PM ET to Friday 5:00 PM ET")
            return

        # Step 1: Sync positions with OANDA
        self.sync_positions()

        # Step 2: Process exits for positions that hit holding period
        self.process_pair_exits()

        # Step 3: Get account balance
        account_balance = self.get_account_balance()
        if account_balance is None:
            print("\nERROR: Failed to fetch account balance")
            return

        # Step 4: Train all models and generate predictions
        trained_models = self.train_all_models()

        # Step 5: Process entries for pairs without positions
        self.process_pair_entries(trained_models, account_balance)

        # Step 6: Summary
        print("\n" + "="*70)
        print("DAILY UPDATE COMPLETE")
        print("="*70)

        active_positions = sum(1 for pm in self.position_managers.values() if pm.has_position())
        print(f"\nActive positions: {active_positions}/{len(self.pairs)}")

        for pair in self.pairs:
            pm = self.position_managers[pair]
            if pm.has_position():
                print(f"  {pair}: {'LONG' if pm.position == 1 else 'SHORT'} @ {pm.entry_price:.5f}")
            else:
                print(f"  {pair}: No position")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OANDA Multi-Pair Trader - 4 Pair Diversified Strategy')
    parser.add_argument('--live', action='store_true', help='Use LIVE account (default: practice)')
    parser.add_argument('--dry-run', action='store_true', help='Simulate only, do not place trades')
    parser.add_argument('--leverage', type=float, default=2.0, help='Leverage multiplier (default: 2.0)')
    parser.add_argument('--pairs', nargs='+', default=TradingConfig.DEFAULT_PAIRS,
                        help='Currency pairs to trade (default: EURUSD GBPUSD AUDUSD USDJPY)')
    parser.add_argument('--model', type=str, default='ann', choices=['xgboost', 'ann'],
                        help='Model type: xgboost or ann (default: ann)')
    parser.add_argument('--yes', action='store_true', help='Skip confirmation prompts (for automated runs)')
    args = parser.parse_args()

    # Validate leverage
    if args.leverage < 1.0:
        print("ERROR: Leverage must be >= 1.0")
        sys.exit(1)
    if args.leverage > 50.0:
        print("ERROR: Leverage cannot exceed 50x (OANDA maximum)")
        sys.exit(1)

    # Confirm if using live account
    practice = not args.live
    if args.live and not args.dry_run and not args.yes:
        print(f"\nWARNING: You are about to trade on a LIVE account")
        print(f"Model: {args.model.upper()}")
        print(f"Pairs: {', '.join(args.pairs)}")
        print(f"Leverage: {args.leverage:.1f}x")
        print(f"\nWith {len(args.pairs)} pairs at {args.leverage:.1f}x leverage:")
        print(f"  Each pair uses {100/len(args.pairs):.1f}% of capital")
        print(f"  Total exposure: {args.leverage * 100:.0f}% of account balance")
        confirm = input("\nType 'YES' to confirm: ")
        if confirm != 'YES':
            print("Aborted.")
            sys.exit(0)

    # Additional leverage warning
    if args.leverage > 1.0 and not args.dry_run and not args.yes:
        print(f"\nWARNING: Using {args.leverage:.1f}x leverage across {len(args.pairs)} pairs!")
        print(f"Max loss if all pairs hit stop: ~{TradingConfig.STOP_LOSS_PCT * args.leverage * 100:.2f}% per pair")
        print(f"Total max loss: ~{TradingConfig.STOP_LOSS_PCT * args.leverage * len(args.pairs) * 100:.2f}% of account")
        confirm_leverage = input("Type 'YES' to confirm leverage: ")
        if confirm_leverage != 'YES':
            print("Aborted.")
            sys.exit(0)

    try:
        trader = MultiPairTrader(
            pairs=[p.upper() for p in args.pairs],
            practice=practice,
            leverage=args.leverage,
            dry_run=args.dry_run,
            model_type=args.model
        )

        trader.run_daily_update()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

        # Send error notification
        try:
            from trading import NotificationService
            notifier = NotificationService()
            notifier.notify_error(f"Multi-pair trading bot error: {str(e)}")
        except:
            pass

        sys.exit(1)
