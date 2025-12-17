"""
OANDA FTMO Signal Reversal Trading Bot

FTMO-Safe Configuration (8 pairs):
- EURUSD, GBPUSD, AUDUSD, USDJPY, EURJPY, USDCAD, USDCHF, NZDUSD
- 10/90 percentile thresholds
- Signal reversal exit (hold until opposite signal)
- 2.5% SL / 2.5% TP (independent per trade)
- 1.5x leverage, 6.2% allocation per slot
- 5 slots max per pair

FTMO Constraints:
- Max 5% daily drawdown
- Max 10% overall drawdown

Based on backtest (conservative buffer):
- ~38% annual return
- ~8% max drawdown (buffer from 10% limit)
- ~2.3% worst day (buffer from 5% limit)
- 8.05 Sharpe ratio

WARNING: Trade on FTMO demo/challenge account first!
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


class FTMOConfig:
    """FTMO-safe trading configuration"""

    # FTMO Risk Parameters
    STOP_LOSS_PCT = 0.025      # 2.5% stop loss
    TAKE_PROFIT_PCT = 0.025    # 2.5% take profit

    # Position Sizing (FTMO-safe)
    ALLOCATION_PER_SLOT = 0.062  # 6.2% per slot
    LEVERAGE = 1.5              # 1.5x leverage
    MAX_SLOTS = 5               # 5 slots max per pair

    # Signal Parameters (same as Sleep Well)
    PERCENTILE_LOWER = 10       # 10th percentile for short
    PERCENTILE_UPPER = 90       # 90th percentile for long
    BUFFER_SIZE = 200           # Prediction buffer size
    BUFFER_WARMUP = 50          # Warmup period

    # Pairs
    DEFAULT_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']


class FTMOPositionManager:
    """Position manager for FTMO signal reversal strategy"""

    def __init__(self, pair):
        self.pair = pair
        self.slots = []  # List of {direction, entry_price, trade_id, entry_date}
        self.direction = 0
        self.MAX_SLOTS = FTMOConfig.MAX_SLOTS
        self.state_file = f'ftmo_state_{pair}.pkl'
        self.load_state()

    def load_state(self):
        """Load state from file"""
        import pickle
        import os
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.slots = state.get('slots', [])
                    self.direction = state.get('direction', 0)
            except:
                self.slots = []
                self.direction = 0

    def save_state(self):
        """Save state to file"""
        import pickle
        with open(self.state_file, 'wb') as f:
            pickle.dump({
                'slots': self.slots,
                'direction': self.direction
            }, f)

    def has_position(self):
        return len(self.slots) > 0

    def slot_count(self):
        return len(self.slots)

    def can_add_slot(self, signal):
        """Check if we can add a slot for this signal"""
        if len(self.slots) >= self.MAX_SLOTS:
            return False
        if self.direction != 0 and self.direction != signal:
            return False
        # Check if already entered today
        today = datetime.now().date()
        for slot in self.slots:
            if slot.get('entry_date') and slot['entry_date'].date() == today:
                return False
        return True

    def add_slot(self, direction, entry_price, trade_id=None):
        """Add a new slot"""
        self.slots.append({
            'direction': direction,
            'entry_price': entry_price,
            'trade_id': trade_id,
            'entry_date': datetime.now()
        })
        self.direction = direction
        self.save_state()

    def close_all_slots(self):
        """Close all slots (for signal reversal)"""
        self.slots = []
        self.direction = 0
        self.save_state()

    def remove_slot(self, index):
        """Remove a specific slot"""
        if 0 <= index < len(self.slots):
            self.slots.pop(index)
            if not self.slots:
                self.direction = 0
            self.save_state()

    def get_slots(self):
        return self.slots


class FTMOTrader:
    """FTMO Signal Reversal Trader"""

    def __init__(self, pairs, practice=True, dry_run=False):
        self.pairs = pairs
        self.practice = practice
        self.dry_run = dry_run

        # Initialize per-pair components
        self.clients = {}
        self.position_managers = {}
        self.models = {}

        for pair in pairs:
            self.clients[pair] = OandaClient(pair, practice=practice)
            self.position_managers[pair] = FTMOPositionManager(pair)
            self.models[pair] = TradingModel(pair, model_type='ann')

        self.notifier = NotificationService()

    def get_account_balance(self):
        """Get account balance"""
        return self.clients[self.pairs[0]].get_account_balance()

    def sync_positions(self):
        """Sync position states with OANDA"""
        print("\n" + "="*70)
        print("SYNCING POSITIONS WITH OANDA")
        print("="*70)

        for pair in self.pairs:
            pm = self.position_managers[pair]

            if self.dry_run:
                if pm.has_position():
                    print(f"{pair}: {pm.slot_count()} slots (dry-run)")
                else:
                    print(f"{pair}: No position (dry-run)")
                continue

            # Check OANDA for actual positions
            actual_position = self.clients[pair].get_open_positions()

            if pm.has_position() and not actual_position:
                print(f"{pair}: Position closed at OANDA (SL/TP hit)")
                pm.close_all_slots()
            elif pm.has_position():
                print(f"{pair}: {pm.slot_count()} slots synced")
            else:
                print(f"{pair}: No position")

    def check_signal_reversal_exits(self, signals):
        """Check for signal reversal exits - close ALL slots if opposite signal"""
        print("\n" + "="*70)
        print("CHECKING SIGNAL REVERSAL EXITS")
        print("="*70)

        for pair in self.pairs:
            pm = self.position_managers[pair]
            client = self.clients[pair]

            if not pm.has_position():
                continue

            signal = signals.get(pair, 0)

            # Check for opposite signal (signal reversal)
            if signal != 0 and signal != pm.direction:
                print(f"\n{pair}: SIGNAL REVERSAL - {pm.direction} -> {signal}")
                print(f"  Closing {pm.slot_count()} slots")

                if not self.dry_run:
                    # Close all trades for this pair
                    for slot in pm.get_slots():
                        if slot.get('trade_id'):
                            client.close_trade(slot['trade_id'])

                    # Also close any remaining position
                    client.close_position(pm.direction)

                    # Clear local state
                    pm.close_all_slots()

                    self.notifier.notify_trade_exit(
                        pair=pair,
                        direction='LONG' if pm.direction == 1 else 'SHORT',
                        entry_price=0,  # Multiple slots
                        exit_price=0,
                        pnl_pct=0,
                        pnl_dollars=0,
                        exit_reason='SIGNAL_REVERSAL',
                        days_held=0
                    )
                else:
                    print(f"  [DRY RUN] Would close all slots")

    def train_and_generate_signals(self):
        """Train models and generate signals for all pairs"""
        print("\n" + "="*70)
        print("TRAINING MODELS AND GENERATING SIGNALS")
        print("="*70)

        signals = {}

        for pair in self.pairs:
            print(f"\n{pair}:")

            # Fetch data
            df = self.clients[pair].fetch_latest_data(count=TradingConfig.TRAIN_WINDOW_SIZE + 300)
            if df.empty:
                print(f"  ERROR: Failed to fetch data")
                continue

            df = df.set_index('date')

            # Train model
            model_obj, scaler, df_clean = self.models[pair].train(df)

            # Generate prediction
            prediction = self.models[pair].predict(df_clean)

            # Generate signal using FTMO thresholds
            signal = self.models[pair].generate_signal(prediction)
            signals[pair] = signal

            signal_str = 'LONG' if signal == 1 else ('SHORT' if signal == -1 else 'NEUTRAL')
            print(f"  Prediction: {prediction:.6f}, Signal: {signal_str}")

        return signals

    def process_entries(self, signals, account_balance):
        """Enter new positions based on signals"""
        print("\n" + "="*70)
        print("PROCESSING ENTRIES")
        print("="*70)

        capital_per_slot = account_balance * FTMOConfig.ALLOCATION_PER_SLOT
        position_size = capital_per_slot * FTMOConfig.LEVERAGE

        print(f"Account: ${account_balance:.2f}")
        print(f"Per slot: ${capital_per_slot:.2f} x {FTMOConfig.LEVERAGE}x = ${position_size:.2f}")

        for pair in self.pairs:
            pm = self.position_managers[pair]
            client = self.clients[pair]
            signal = signals.get(pair, 0)

            if signal == 0:
                continue

            if not pm.can_add_slot(signal):
                if pm.direction != 0 and pm.direction != signal:
                    # Opposite signal should have been handled in reversal check
                    continue
                elif pm.slot_count() >= pm.MAX_SLOTS:
                    print(f"\n{pair}: Max slots reached")
                continue

            print(f"\n{pair}: {'LONG' if signal == 1 else 'SHORT'} (slot {pm.slot_count()+1}/{pm.MAX_SLOTS})")

            # Get current price
            price_data = client.fetcher.get_current_price(pair)
            if not price_data:
                print(f"  ERROR: Failed to get price")
                continue

            current_price = price_data['mid']
            print(f"  Price: {current_price:.5f}")
            print(f"  SL: {FTMOConfig.STOP_LOSS_PCT*100:.1f}%, TP: {FTMOConfig.TAKE_PROFIT_PCT*100:.1f}%")

            if self.dry_run:
                print(f"  [DRY RUN] Would place order")
                # Still update local state for dry run
                pm.add_slot(signal, current_price, trade_id=None)
            else:
                # Place order with SL and TP
                is_adding = pm.slot_count() > 0
                order_result = client.place_order(
                    signal=signal,
                    current_price=current_price,
                    position_size_dollars=position_size,
                    stop_loss_pct=FTMOConfig.STOP_LOSS_PCT,
                    take_profit_pct=FTMOConfig.TAKE_PROFIT_PCT,
                    allow_add_to_position=is_adding
                )

                if order_result and order_result['success']:
                    pm.add_slot(signal, order_result['entry_price'], order_result['trade_id'])
                    print(f"  [OK] Order placed")

                    self.notifier.notify_trade_entry(
                        pair=pair,
                        direction='LONG' if signal == 1 else 'SHORT',
                        entry_price=order_result['entry_price'],
                        position_size=position_size,
                        stop_loss=order_result['stop_price'],
                        take_profit=order_result['target_price']
                    )
                else:
                    print(f"  [FAIL] Order failed")

    def run_daily_update(self):
        """Execute daily trading workflow"""
        print(f"\n{'='*70}")
        print(f"FTMO SIGNAL REVERSAL - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Pairs: {', '.join(self.pairs)}")
        print(f"Mode: {'DRY RUN' if self.dry_run else ('PRACTICE' if self.practice else 'LIVE')}")
        print(f"Config: {FTMOConfig.ALLOCATION_PER_SLOT*100:.1f}%/slot, {FTMOConfig.LEVERAGE}x lev")
        print(f"{'='*70}")

        # Check market
        if not is_forex_market_open():
            et_tz = pytz.timezone('America/New_York')
            now_et = datetime.now(et_tz)
            print(f"\nMarket CLOSED ({now_et.strftime('%A %H:%M %Z')})")
            return

        # Step 1: Sync positions
        self.sync_positions()

        # Step 2: Get account balance
        if self.dry_run:
            account_balance = 100000  # Mock $100k FTMO account for dry run
            print(f"\n[DRY RUN] Using mock balance: ${account_balance:,.0f}")
        else:
            account_balance = self.get_account_balance()
            if account_balance is None:
                print("\nERROR: Failed to get account balance")
                return

        # Step 3: Train and generate signals
        signals = self.train_and_generate_signals()

        # Step 4: Check for signal reversal exits
        self.check_signal_reversal_exits(signals)

        # Step 5: Process new entries
        self.process_entries(signals, account_balance)

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        total_slots = sum(pm.slot_count() for pm in self.position_managers.values())
        max_slots = len(self.pairs) * FTMOConfig.MAX_SLOTS
        print(f"Active slots: {total_slots}/{max_slots}")

        for pair in self.pairs:
            pm = self.position_managers[pair]
            if pm.has_position():
                dir_str = 'LONG' if pm.direction == 1 else 'SHORT'
                print(f"  {pair}: {dir_str} - {pm.slot_count()}/{pm.MAX_SLOTS} slots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FTMO Signal Reversal Trader')
    parser.add_argument('--live', action='store_true', help='Use LIVE account')
    parser.add_argument('--dry-run', action='store_true', help='Simulate only')
    parser.add_argument('--pairs', nargs='+', default=FTMOConfig.DEFAULT_PAIRS)
    parser.add_argument('--yes', action='store_true', help='Skip confirmations')
    args = parser.parse_args()

    practice = not args.live

    if args.live and not args.dry_run and not args.yes:
        print("\nWARNING: LIVE FTMO TRADING")
        print(f"Pairs: {', '.join(args.pairs)}")
        print(f"Allocation: {FTMOConfig.ALLOCATION_PER_SLOT*100:.1f}%/slot")
        print(f"Leverage: {FTMOConfig.LEVERAGE}x")
        print(f"SL/TP: {FTMOConfig.STOP_LOSS_PCT*100:.1f}%/{FTMOConfig.TAKE_PROFIT_PCT*100:.1f}%")
        confirm = input("\nType 'YES' to confirm: ")
        if confirm != 'YES':
            print("Aborted.")
            sys.exit(0)

    try:
        trader = FTMOTrader(
            pairs=[p.upper() for p in args.pairs],
            practice=practice,
            dry_run=args.dry_run
        )
        trader.run_daily_update()

    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
