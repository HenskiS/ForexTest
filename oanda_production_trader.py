"""
OANDA Production Trading Bot - Refactored v4

Clean modular implementation using trading/ infrastructure.

Implements rolling daily retraining with:
- 1-day target predictions (eliminates look-ahead bias)
- 1-day holding period with tight stops
- Optimized parameters: 0.18% stop loss, 2.00% take profit
- Position state persistence
- Realistic same-day exit then re-entry logic

WARNING: This bot trades real money. Test thoroughly on practice account first!
"""
import sys
import argparse
from datetime import datetime
import pytz

# Import modular trading infrastructure
from trading import (
    TradingConfig,
    OandaClient,
    PositionManager,
    TradingModel,
    is_forex_market_open,
    NotificationService
)


def run_daily_update(pair, practice=True, leverage=1.0, dry_run=False):
    """
    Run daily trading workflow:
    1. Sync state with OANDA
    2. Check and close existing position if needed
    3. Generate new signal
    4. Place new order if signal generated

    Args:
        pair: Currency pair to trade (e.g., 'EURUSD')
        practice: Use practice account if True
        leverage: Leverage multiplier (1.0 = no leverage)
        dry_run: Simulate only, don't place actual trades
    """
    print(f"\n{'='*70}")
    print(f"DAILY UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pair: {pair}")
    print(f"Mode: {'DRY RUN' if dry_run else ('PRACTICE' if practice else 'LIVE')}")
    print(f"Leverage: {leverage:.1f}x")
    print(f"{'='*70}\n")

    # Check if market is open
    if not is_forex_market_open():
        et_tz = pytz.timezone('America/New_York')
        now_et = datetime.now(et_tz)
        print(f"Forex market is CLOSED (current time: {now_et.strftime('%A %Y-%m-%d %H:%M:%S %Z')})")
        print(f"Market hours: Sunday 5:00 PM ET to Friday 5:00 PM ET")
        print(f"Exiting without executing any trading logic.")
        return

    # Initialize modules
    client = OandaClient(pair, practice=practice)
    position_mgr = PositionManager(pair)
    model = TradingModel(pair)
    notifier = NotificationService()

    # Step 1: Sync state with OANDA (in case stop/TP hit)
    if position_mgr.has_position() and not dry_run:
        actual_position = client.get_open_positions()
        if not actual_position:
            print("\nWARNING: State file shows open position, but no position at OANDA")
            print("Clearing local state to sync with OANDA (position likely closed by stop/TP)")
            position_mgr.close_position()
            print("State synced with OANDA\n")

    # Step 2: Fetch latest data and train model
    print("\nFetching latest data...")
    df = client.fetch_latest_data(count=TradingConfig.TRAIN_WINDOW_SIZE + 300)
    if df.empty:
        print("ERROR: Failed to fetch data")
        return

    df = df.set_index('date')
    print(f"Data: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

    print("\nTraining model...")
    model_obj, scaler, df_clean = model.train(df)

    # Step 3: Generate prediction
    print("\nGenerating prediction...")
    prediction = model.predict(df_clean)

    # Step 4: Check if should exit existing position
    if position_mgr.has_position():
        print(f"\nCurrent position: {'LONG' if position_mgr.position == 1 else 'SHORT'}")
        print(f"Entry: {position_mgr.entry_price:.5f}")
        print(f"Entry date: {position_mgr.entry_date.date()}")

        # Check if should exit by time (1 day holding period)
        should_exit = position_mgr.should_exit_by_time(
            holding_period_days=TradingConfig.HOLDING_PERIOD_DAYS
        )

        if should_exit:
            print(f"\nTime-based exit triggered (held for {TradingConfig.HOLDING_PERIOD_DAYS} day)")

            # Get current price
            current_price_data = client.fetcher.get_current_price(pair)
            if not current_price_data:
                print("ERROR: Failed to get current price for exit")
                return

            current_price = current_price_data['mid']
            print(f"Current price: {current_price:.5f}")

            # Calculate P&L
            pnl = position_mgr.calculate_pnl(current_price)
            print(f"P&L: {pnl['pnl_pct']:.2f}% (${pnl['pnl_dollars']:.2f})")

            if not dry_run:
                # Close position at OANDA
                success = client.close_position(position_mgr.position)
                if success:
                    # Log trade
                    position_mgr.log_trade(
                        exit_price=current_price,
                        exit_reason='TIME_EXIT',
                        prediction=prediction
                    )

                    # Send notification
                    notifier.notify_trade_exit(
                        pair=pair,
                        direction='LONG' if position_mgr.position == 1 else 'SHORT',
                        entry_price=position_mgr.entry_price,
                        exit_price=current_price,
                        pnl_pct=pnl['pnl_pct'],
                        pnl_dollars=pnl['pnl_dollars'],
                        exit_reason='TIME_EXIT',
                        days_held=(datetime.now().date() - position_mgr.entry_date.date()).days
                    )

                    # Clear position state
                    position_mgr.close_position()
                    print("Position closed successfully")
                else:
                    print("ERROR: Failed to close position")
                    return
            else:
                print("[DRY RUN] Would close position")
                position_mgr.close_position()  # Clear state in dry-run too

    # Step 5: Generate trading signal
    print("\nGenerating signal...")
    signal = model.generate_signal(prediction)

    if signal == 0:
        print("\nNo trade signal (HOLD)")
        return

    # Step 6: Place new order if not already in position
    if position_mgr.has_position():
        print("\nAlready in position, skipping new entry")
        return

    # Get current price
    current_price_data = client.fetcher.get_current_price(pair)
    if not current_price_data:
        print("ERROR: Failed to get current price")
        return

    current_price = current_price_data['mid']
    print(f"\nCurrent price: {current_price:.5f}")

    # Get account balance
    account_balance = client.get_account_balance()
    if account_balance is None:
        print("ERROR: Failed to fetch account balance")
        return

    # Calculate position size
    position_size_dollars = account_balance * leverage

    if dry_run:
        # Dry run - just show what would happen
        print(f"\n[DRY RUN] Would place {'LONG' if signal == 1 else 'SHORT'} order:")
        print(f"  Position size: ${position_size_dollars:.2f}")
        print(f"  Entry: {current_price:.5f}")
        print(f"  Stop Loss: {TradingConfig.STOP_LOSS_PCT*100:.2f}%")
        print(f"  Take Profit: {TradingConfig.TAKE_PROFIT_PCT*100:.2f}%")
    else:
        # Live trading - place order
        order_result = client.place_order(
            signal=signal,
            current_price=current_price,
            position_size_dollars=position_size_dollars,
            stop_loss_pct=TradingConfig.STOP_LOSS_PCT,
            take_profit_pct=TradingConfig.TAKE_PROFIT_PCT
        )

        if order_result and order_result['success']:
            # Update position manager
            position_mgr.open_position(
                direction=signal,
                entry_price=order_result['entry_price'],
                position_size=position_size_dollars,
                trade_id=order_result['trade_id']
            )

            # Send notification
            notifier.notify_trade_entry(
                pair=pair,
                direction='LONG' if signal == 1 else 'SHORT',
                entry_price=order_result['entry_price'],
                position_size=position_size_dollars,
                stop_loss=order_result['stop_price'],
                take_profit=order_result['target_price']
            )

            print("\n[OK] Trade executed successfully!")
        else:
            print("\n[FAIL] Trade execution failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OANDA Production Trader v4 - Modular')
    parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair to trade')
    parser.add_argument('--live', action='store_true', help='Use LIVE account (default: practice)')
    parser.add_argument('--dry-run', action='store_true', help='Simulate only, do not place trades')
    parser.add_argument('--leverage', type=float, default=1.0, help='Leverage multiplier (default: 1.0 = no leverage)')
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
        print(f"Leverage: {args.leverage:.1f}x")
        print(f"Pair: {args.pair}")
        confirm = input("\nType 'YES' to confirm: ")
        if confirm != 'YES':
            print("Aborted.")
            sys.exit(0)

    # Warn about leverage
    if args.leverage > 1.0 and not args.dry_run and not args.yes:
        print(f"\nWARNING: Using {args.leverage:.1f}x leverage increases risk!")
        print(f"Max loss per trade: ~{TradingConfig.STOP_LOSS_PCT * args.leverage * 100:.2f}% of account balance")
        confirm_leverage = input("Type 'YES' to confirm leverage: ")
        if confirm_leverage != 'YES':
            print("Aborted.")
            sys.exit(0)

    try:
        run_daily_update(
            pair=args.pair.upper(),
            practice=practice,
            leverage=args.leverage,
            dry_run=args.dry_run
        )
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
            notifier.notify_error(f"Trading bot error for {args.pair}: {str(e)}")
        except:
            pass

        sys.exit(1)
