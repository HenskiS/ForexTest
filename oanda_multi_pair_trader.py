"""
OANDA Multi-Pair Production Trading Bot

"Sleep Well" Configuration (8 pairs):
- EURUSD, GBPUSD, AUDUSD, USDJPY, EURJPY, USDCAD, USDCHF, NZDUSD
- 10/90 percentile thresholds (fewer, higher-quality trades)
- 5-day holding period
- 2.5% DCA stop loss (averaged across slots), no take profit (time-based exit)
- 2.0x leverage, 22.5% allocation per slot for ~98% annual

DCA Averaged Stops (OANDA netting account behavior):
- OANDA netting accounts only support ONE stop per instrument
- When adding slots, stop is recalculated at 2.5% below weighted average entry
- This actually performs BETTER than independent stops (~98% vs ~94% annual)

Based on backtest results (4500 days, bug-fixed Dec 2025):
- ~98% annual return (with DCA averaged stops)
- 60.8% win rate
- ~16.1% max drawdown
- 4.69 Sharpe ratio
- Average 6.4 positions/day, ~2.9x effective leverage

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

    def __init__(self, pairs, practice=True, leverage=2.0, dry_run=False, model_type='ann'):
        """
        Initialize multi-pair trader.

        Args:
            pairs: List of currency pairs to trade
            practice: Use practice account if True
            leverage: Leverage multiplier (e.g., 2.0 = 2:1 leverage)
            dry_run: Simulate only, don't place actual trades
            model_type: 'xgboost' or 'ann' (default: 'ann')
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

            print(f"\n{pair}: {pm.slot_count()} slots open ({'LONG' if pm.direction == 1 else 'SHORT'})")

            # Get slots that should exit by time
            slots_to_exit = pm.get_slots_to_exit_by_time(TradingConfig.HOLDING_PERIOD_DAYS)

            if not slots_to_exit:
                # Show status of open slots
                for i, slot in enumerate(pm.get_slots()):
                    entry_date = slot['entry_date'].date()
                    today = datetime.now().date()
                    days_held = pm._count_business_days(entry_date, today)
                    print(f"  Slot {i+1}: {slot['entry_price']:.5f} ({days_held}/{TradingConfig.HOLDING_PERIOD_DAYS} days)")
                continue

            print(f"  {len(slots_to_exit)} slot(s) ready for time exit")

            # Get current price for P&L calculation
            current_price_data = client.fetcher.get_current_price(pair)
            if not current_price_data:
                print(f"  ERROR: Failed to get current price")
                continue
            current_price = current_price_data['mid']
            print(f"  Current price: {current_price:.5f}")

            # Process exits in reverse order (so indices stay valid)
            for slot_idx in sorted(slots_to_exit, reverse=True):
                slot = pm.slots[slot_idx]
                entry_date = slot['entry_date'].date()
                today = datetime.now().date()
                days_held = pm._count_business_days(entry_date, today)

                # Calculate P&L for this slot
                pnl = pm.calculate_pnl(current_price, slot_idx)
                print(f"  Slot {slot_idx+1}: Entry {slot['entry_price']:.5f}, P&L: {pnl['pnl_pct']:.2f}%")

                if not self.dry_run:
                    # Try to close specific trade if we have trade_id
                    if slot.get('trade_id'):
                        success = client.close_trade(slot['trade_id'])
                    else:
                        # Fall back to closing all (less precise)
                        success = client.close_position(pm.direction)

                    if success:
                        # Log trade
                        pm.log_trade(
                            exit_price=current_price,
                            exit_reason='TIME_EXIT',
                            slot_index=slot_idx,
                            prediction=None
                        )

                        # Send notification
                        self.notifier.notify_trade_exit(
                            pair=pair,
                            direction='LONG' if pm.direction == 1 else 'SHORT',
                            entry_price=slot['entry_price'],
                            exit_price=current_price,
                            pnl_pct=pnl['pnl_pct'],
                            pnl_dollars=pnl['pnl_dollars'],
                            exit_reason='TIME_EXIT',
                            days_held=days_held
                        )

                        # Remove slot from position manager
                        pm.remove_slot(slot_idx)
                else:
                    print(f"  [DRY RUN] Would close slot {slot_idx+1}")

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

            # Note: prediction is added to buffer in generate_signal()

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

        # Calculate per-slot position size (matches backtest: 22.5% allocation per slot)
        # With $500 account, 22.5% allocation, 2.0x leverage = $225 position per slot
        ALLOCATION_PER_SLOT = 0.225  # 22.5% of account per slot (matches backtest)
        capital_per_slot = account_balance * ALLOCATION_PER_SLOT
        position_size_per_slot = capital_per_slot * self.leverage

        print(f"\nAccount balance: ${account_balance:.2f}")
        print(f"Allocation per slot: {ALLOCATION_PER_SLOT*100:.1f}% = ${capital_per_slot:.2f}")
        print(f"Position size per slot: ${position_size_per_slot:.2f} ({self.leverage:.1f}x leverage)")

        for pair in self.pairs:
            pm = self.position_managers[pair]
            client = self.clients[pair]

            if pair not in trained_models:
                print(f"\n{pair}: Skipping (no trained model)")
                continue

            # Check if we can add a slot (respects direction and max slots)
            # First peek at the signal to check direction compatibility
            prediction = trained_models[pair]['prediction']
            signal = self.models[pair].generate_signal(prediction)

            if signal == 0:
                if pm.has_position():
                    print(f"\n{pair}: {pm.slot_count()}/{pm.MAX_SLOTS} slots open, no new signal")
                continue

            if not pm.can_add_slot(signal):
                if pm.direction != 0 and pm.direction != signal:
                    print(f"\n{pair}: Signal is opposite direction, skipping (FIFO)")
                elif pm.has_slot_entered_today():
                    print(f"\n{pair}: Already entered today, skipping (one entry per day)")
                else:
                    print(f"\n{pair}: Max slots ({pm.MAX_SLOTS}) reached, skipping")
                continue

            print(f"\n{pair}: (slot {pm.slot_count() + 1}/{pm.MAX_SLOTS})")
            print("-" * 70)

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

            # Calculate DCA averaged stop if adding to existing position
            # OANDA netting accounts only support ONE stop per position
            is_adding_to_position = pm.slot_count() > 0
            if is_adding_to_position:
                # Calculate averaged stop price including new slot
                dca_stop_price = pm.calculate_dca_stop_price(
                    new_entry_price=current_price,
                    new_position_size=position_size_per_slot,
                    signal=signal,
                    stop_loss_pct=TradingConfig.STOP_LOSS_PCT
                )
                print(f"  DCA Stop: {dca_stop_price:.5f} (avg entry-based)")
            else:
                dca_stop_price = None  # First slot uses simple stop

            if self.dry_run:
                print(f"  [DRY RUN] Would place {'LONG' if signal == 1 else 'SHORT'} order:")
                print(f"    Position size: ${position_size_per_slot:.2f}")
                if dca_stop_price:
                    print(f"    Stop Loss: DCA averaged at {dca_stop_price:.5f}")
                else:
                    print(f"    Stop Loss: {TradingConfig.STOP_LOSS_PCT*100:.2f}%")
                if take_profit_pct:
                    print(f"    Take Profit: {take_profit_pct*100:.2f}%")
                else:
                    print(f"    Take Profit: None (time-based exit)")
            else:
                # Place order (with DCA stop if adding to position)
                order_result = client.place_order(
                    signal=signal,
                    current_price=current_price,
                    position_size_dollars=position_size_per_slot,
                    stop_loss_pct=TradingConfig.STOP_LOSS_PCT,
                    take_profit_pct=take_profit_pct,
                    allow_add_to_position=is_adding_to_position,
                    custom_stop_price=dca_stop_price
                )

                if order_result and order_result['success']:
                    # Update position manager
                    pm.open_position(
                        direction=signal,
                        entry_price=order_result['entry_price'],
                        position_size=position_size_per_slot,
                        trade_id=order_result['trade_id']
                    )

                    # Send notification
                    self.notifier.notify_trade_entry(
                        pair=pair,
                        direction='LONG' if signal == 1 else 'SHORT',
                        entry_price=order_result['entry_price'],
                        position_size=position_size_per_slot,
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

        total_slots = sum(pm.slot_count() for pm in self.position_managers.values())
        pairs_with_positions = sum(1 for pm in self.position_managers.values() if pm.has_position())
        max_slots = len(self.pairs) * PositionManager.MAX_SLOTS
        print(f"\nActive slots: {total_slots}/{max_slots} ({pairs_with_positions}/{len(self.pairs)} pairs)")

        for pair in self.pairs:
            pm = self.position_managers[pair]
            if pm.has_position():
                print(f"  {pair}: {'LONG' if pm.direction == 1 else 'SHORT'} - {pm.slot_count()}/{pm.MAX_SLOTS} slots")
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
