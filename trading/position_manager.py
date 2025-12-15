"""
Position Manager

Handles position state tracking, persistence, and trade logging.
Supports multiple slots per pair with independent stop losses.
"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from .config import TradingConfig


class PositionManager:
    """Manage trading position state and trade history with multi-slot support"""

    MAX_SLOTS = 5  # Maximum slots per pair

    def __init__(self, pair):
        """
        Initialize position manager for a trading pair.

        Args:
            pair: Forex pair (e.g., 'EURUSD')
        """
        self.pair = pair.upper()

        # State file for persistence
        self.state_file = f'data/oanda_cache/{self.pair}_state.json'
        os.makedirs('data/oanda_cache', exist_ok=True)

        # Trade log file
        self.trade_log_file = TradingConfig.get_trade_log(self.pair)

        # Multi-slot position state
        self.direction = 0  # 0 = no position, 1 = long, -1 = short (shared by all slots)
        self.slots = []  # List of slot dicts: {entry_price, entry_date, trade_id, position_size}

        # Legacy single-position fields for backwards compatibility
        self.position = 0
        self.entry_price = None
        self.entry_date = None
        self.trade_id = None
        self.position_size = 0

        # Load persisted state
        self.load_state()

    def load_state(self):
        """Load persisted state from disk"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)

                # Check for new multi-slot format
                if 'slots' in state:
                    self.direction = state.get('direction', 0)
                    self.slots = []
                    for slot in state.get('slots', []):
                        slot_data = {
                            'entry_price': slot['entry_price'],
                            'entry_date': datetime.fromisoformat(slot['entry_date']) if slot.get('entry_date') else None,
                            'trade_id': slot.get('trade_id'),
                            'position_size': slot.get('position_size', 0)
                        }
                        self.slots.append(slot_data)
                    print(f"Loaded state: direction={self.direction}, slots={len(self.slots)}")
                else:
                    # Legacy single-position format - migrate to multi-slot
                    self.direction = state.get('position', 0)
                    if self.direction != 0:
                        entry_date = state.get('entry_date')
                        if entry_date:
                            entry_date = datetime.fromisoformat(entry_date)
                        self.slots = [{
                            'entry_price': state.get('entry_price'),
                            'entry_date': entry_date,
                            'trade_id': state.get('trade_id'),
                            'position_size': state.get('position_size', 0)
                        }]
                    else:
                        self.slots = []
                    print(f"Migrated legacy state: direction={self.direction}, slots={len(self.slots)}")

                # Update legacy fields for backwards compatibility
                self._sync_legacy_fields()
        else:
            print("No saved state found - starting fresh")

    def _sync_legacy_fields(self):
        """Sync legacy single-position fields from slots for backwards compatibility"""
        if self.slots:
            # Use first slot's data for legacy fields
            self.position = self.direction
            self.entry_price = self.slots[0]['entry_price']
            self.entry_date = self.slots[0]['entry_date']
            self.trade_id = self.slots[0]['trade_id']
            self.position_size = sum(s['position_size'] for s in self.slots)
        else:
            self.position = 0
            self.entry_price = None
            self.entry_date = None
            self.trade_id = None
            self.position_size = 0

    def save_state(self):
        """Persist state to disk"""
        slots_data = []
        for slot in self.slots:
            slots_data.append({
                'entry_price': slot['entry_price'],
                'entry_date': slot['entry_date'].isoformat() if slot['entry_date'] else None,
                'trade_id': slot['trade_id'],
                'position_size': slot['position_size']
            })

        state = {
            'direction': self.direction,
            'slots': slots_data,
            # Keep legacy fields for backwards compatibility
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'trade_id': self.trade_id,
            'position_size': self.position_size
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def has_slot_entered_today(self):
        """Check if any slot was entered today (prevents duplicate entries on re-run)"""
        today = datetime.now().date()
        for slot in self.slots:
            if slot['entry_date'] and slot['entry_date'].date() == today:
                return True
        return False

    def can_add_slot(self, signal, max_slots=None, allow_same_day=False):
        """
        Check if a new slot can be added for the given signal.

        Args:
            signal: 1 for long, -1 for short
            max_slots: Maximum slots allowed (default: MAX_SLOTS)
            allow_same_day: If False, prevents adding slot if one was already entered today

        Returns:
            bool: True if slot can be added
        """
        if max_slots is None:
            max_slots = self.MAX_SLOTS

        # Can't add if at max slots
        if len(self.slots) >= max_slots:
            return False

        # Can't add opposite direction (FIFO compliance)
        if self.direction != 0 and self.direction != signal:
            return False

        # Can't add if already entered today (unless explicitly allowed)
        if not allow_same_day and self.has_slot_entered_today():
            return False

        return True

    def add_slot(self, direction, entry_price, position_size, trade_id=None):
        """
        Add a new slot to the position.

        Args:
            direction: 1 for long, -1 for short
            entry_price: Entry price for this slot
            position_size: Position size in dollars
            trade_id: Trade ID from broker (optional)

        Returns:
            int: Slot index (0-based)
        """
        if not self.can_add_slot(direction):
            print(f"WARNING: Cannot add slot - direction mismatch or max slots reached")
            return -1

        slot = {
            'entry_price': entry_price,
            'entry_date': datetime.now(),
            'trade_id': trade_id,
            'position_size': position_size
        }
        self.slots.append(slot)
        self.direction = direction
        self._sync_legacy_fields()
        self.save_state()

        slot_num = len(self.slots)
        print(f"Slot {slot_num}/{self.MAX_SLOTS} opened: {'LONG' if direction == 1 else 'SHORT'} {self.pair} @ {entry_price:.5f}")
        return slot_num - 1

    def remove_slot(self, slot_index=None, trade_id=None):
        """
        Remove a slot by index or trade_id.

        Args:
            slot_index: Index of slot to remove
            trade_id: Trade ID to match

        Returns:
            dict: Removed slot data, or None if not found
        """
        if trade_id is not None:
            for i, slot in enumerate(self.slots):
                if slot['trade_id'] == trade_id:
                    slot_index = i
                    break

        if slot_index is None or slot_index >= len(self.slots):
            return None

        removed = self.slots.pop(slot_index)

        # Clear direction if no slots left
        if not self.slots:
            self.direction = 0

        self._sync_legacy_fields()
        self.save_state()
        return removed

    def open_position(self, direction, entry_price, position_size, trade_id=None):
        """
        Record opening of a new position (legacy method - wraps add_slot).

        Args:
            direction: 1 for long, -1 for short
            entry_price: Entry price
            position_size: Position size in dollars
            trade_id: Trade ID from broker (optional)
        """
        self.add_slot(direction, entry_price, position_size, trade_id)

    def close_position(self):
        """Clear all slots (legacy method for full position close)"""
        self.direction = 0
        self.slots = []
        self._sync_legacy_fields()
        self.save_state()
        print(f"All positions closed: {self.pair}")

    def has_position(self):
        """Check if currently in a position (any slots open)"""
        return len(self.slots) > 0

    def slot_count(self):
        """Return number of open slots"""
        return len(self.slots)

    def get_slots(self):
        """Return list of all slots"""
        return self.slots.copy()

    def _count_business_days(self, start_date, end_date):
        """
        Count business days (weekdays) between two dates.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            int: Number of business days (excluding start, including end)
        """
        if start_date >= end_date:
            return 0

        business_days = 0
        current = start_date + timedelta(days=1)  # Start counting from day after entry

        while current <= end_date:
            # Monday = 0, Sunday = 6
            if current.weekday() < 5:  # Weekday
                business_days += 1
            current += timedelta(days=1)

        return business_days

    def get_slots_to_exit_by_time(self, holding_period_days=5):
        """
        Get list of slot indices that should be exited based on holding period.

        Args:
            holding_period_days: Maximum trading days to hold

        Returns:
            list: Indices of slots ready to exit
        """
        today = datetime.now().date()
        exit_slots = []

        for i, slot in enumerate(self.slots):
            if slot['entry_date'] is None:
                continue
            entry_date = slot['entry_date'].date()
            days_held = self._count_business_days(entry_date, today)
            if days_held >= holding_period_days:
                exit_slots.append(i)

        return exit_slots

    def should_exit_by_time(self, holding_period_days=1):
        """
        Check if ANY slot should be exited based on holding period.
        Legacy method - use get_slots_to_exit_by_time for multi-slot.

        Args:
            holding_period_days: Maximum trading days to hold position

        Returns:
            bool: True if any slot should be exited
        """
        return len(self.get_slots_to_exit_by_time(holding_period_days)) > 0

    def calculate_pnl(self, exit_price, slot_index=0):
        """
        Calculate P&L for a specific slot.

        Args:
            exit_price: Exit price
            slot_index: Index of slot to calculate (default: 0)

        Returns:
            dict: {'pnl_pct': float, 'pnl_dollars': float}
        """
        if slot_index >= len(self.slots):
            return None

        slot = self.slots[slot_index]

        # Calculate return based on direction
        if self.direction == 1:  # Long
            pnl_pct = ((exit_price - slot['entry_price']) / slot['entry_price']) * 100
        else:  # Short
            pnl_pct = ((slot['entry_price'] - exit_price) / slot['entry_price']) * 100

        # Account for transaction costs
        pnl_pct -= TradingConfig.TRANSACTION_COST_PCT * 100

        # Calculate dollar P&L
        pnl_dollars = (pnl_pct / 100) * slot['position_size']

        return {
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars
        }

    def log_trade(self, exit_price, exit_reason, slot_index=0, prediction=None):
        """
        Log completed trade to CSV file.

        Args:
            exit_price: Exit price
            exit_reason: Reason for exit (e.g., 'TIME_EXIT', 'STOP_LOSS', 'TAKE_PROFIT')
            slot_index: Index of slot being closed
            prediction: Model prediction value (optional)
        """
        if slot_index >= len(self.slots):
            print("WARNING: Invalid slot index for logging")
            return

        slot = self.slots[slot_index]

        # Calculate P&L
        pnl = self.calculate_pnl(exit_price, slot_index)
        if pnl is None:
            print("ERROR: Could not calculate P&L")
            return

        # Calculate business days held
        entry_date = slot['entry_date'].date() if slot['entry_date'] else datetime.now().date()
        today = datetime.now().date()
        days_held = self._count_business_days(entry_date, today)

        trade_record = {
            'entry_date': slot['entry_date'].strftime('%Y-%m-%d') if slot['entry_date'] else '',
            'exit_date': datetime.now().strftime('%Y-%m-%d'),
            'direction': 'LONG' if self.direction == 1 else 'SHORT',
            'entry_price': slot['entry_price'],
            'exit_price': exit_price,
            'position_size': slot['position_size'],
            'pnl_pct': round(pnl['pnl_pct'], 4),
            'pnl_dollars': round(pnl['pnl_dollars'], 2),
            'outcome': 'WIN' if pnl['pnl_pct'] > 0 else 'LOSS',
            'exit_reason': exit_reason,
            'days_held': days_held,
            'prediction': round(prediction, 6) if prediction is not None else None,
            'signal': self.direction,
            'slot_num': slot_index + 1,
            'total_slots': len(self.slots)
        }

        # Create DataFrame and append to CSV
        trade_df = pd.DataFrame([trade_record])

        if os.path.exists(self.trade_log_file):
            trade_df.to_csv(self.trade_log_file, mode='a', header=False, index=False)
        else:
            trade_df.to_csv(self.trade_log_file, mode='w', header=True, index=False)

        print(f"\nTrade logged to {self.trade_log_file}")
        print(f"Slot {slot_index + 1} P&L: {pnl['pnl_pct']:.2f}% (${pnl['pnl_dollars']:.2f})")

    def get_trade_history(self, limit=None):
        """
        Get trade history from log file.

        Args:
            limit: Maximum number of recent trades to return (None for all)

        Returns:
            DataFrame: Trade history
        """
        if not os.path.exists(self.trade_log_file):
            return pd.DataFrame()

        df = pd.read_csv(self.trade_log_file)

        if limit is not None:
            df = df.tail(limit)

        return df

    def get_performance_stats(self):
        """
        Calculate performance statistics from trade history.

        Returns:
            dict: Performance metrics (win rate, avg win, avg loss, etc.)
        """
        df = self.get_trade_history()

        if len(df) == 0:
            return None

        wins = df[df['outcome'] == 'WIN']
        losses = df[df['outcome'] == 'LOSS']

        stats = {
            'total_trades': len(df),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(df) * 100 if len(df) > 0 else 0,
            'avg_win': wins['pnl_pct'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl_pct'].mean() if len(losses) > 0 else 0,
            'total_pnl_pct': df['pnl_pct'].sum(),
            'total_pnl_dollars': df['pnl_dollars'].sum()
        }

        return stats
