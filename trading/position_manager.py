"""
Position Manager

Handles position state tracking, persistence, and trade logging.
"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from .config import TradingConfig


class PositionManager:
    """Manage trading position state and trade history"""

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

        # Position state
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = None
        self.entry_date = None
        self.trade_id = None
        self.position_size = 0  # Position size in dollars

        # Load persisted state
        self.load_state()

    def load_state(self):
        """Load persisted state from disk"""
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
                print(f"Loaded state: position={self.position}, entry_date={self.entry_date}")
        else:
            print("No saved state found - starting fresh")

    def save_state(self):
        """Persist state to disk"""
        state = {
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'trade_id': self.trade_id,
            'position_size': self.position_size
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def open_position(self, direction, entry_price, position_size, trade_id=None):
        """
        Record opening of a new position.

        Args:
            direction: 1 for long, -1 for short
            entry_price: Entry price
            position_size: Position size in dollars
            trade_id: Trade ID from broker (optional)
        """
        self.position = direction
        self.entry_price = entry_price
        self.entry_date = datetime.now()
        self.position_size = position_size
        self.trade_id = trade_id
        self.save_state()

        print(f"Position opened: {'LONG' if direction == 1 else 'SHORT'} {self.pair} @ {entry_price:.5f}")

    def close_position(self):
        """Clear position state after closing"""
        self.position = 0
        self.entry_price = None
        self.entry_date = None
        self.trade_id = None
        self.position_size = 0
        self.save_state()

        print(f"Position closed: {self.pair}")

    def has_position(self):
        """Check if currently in a position"""
        return self.position != 0

    def should_exit_by_time(self, holding_period_days=1):
        """
        Check if position should be exited based on holding period.

        Args:
            holding_period_days: Maximum days to hold position

        Returns:
            bool: True if position should be exited by time
        """
        if not self.has_position() or self.entry_date is None:
            return False

        # Calculate days held (use date, not datetime, to count calendar days)
        days_held = (datetime.now().date() - self.entry_date.date()).days

        return days_held >= holding_period_days

    def calculate_pnl(self, exit_price):
        """
        Calculate P&L for current position.

        Args:
            exit_price: Exit price

        Returns:
            dict: {'pnl_pct': float, 'pnl_dollars': float}
        """
        if not self.has_position() or self.entry_price is None:
            return None

        # Calculate return based on direction
        if self.position == 1:  # Long
            pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:  # Short
            pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100

        # Account for transaction costs
        pnl_pct -= TradingConfig.TRANSACTION_COST_PCT * 100

        # Calculate dollar P&L
        pnl_dollars = (pnl_pct / 100) * self.position_size

        return {
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars
        }

    def log_trade(self, exit_price, exit_reason, prediction=None):
        """
        Log completed trade to CSV file.

        Args:
            exit_price: Exit price
            exit_reason: Reason for exit (e.g., 'TIME_EXIT', 'STOP_LOSS', 'TAKE_PROFIT')
            prediction: Model prediction value (optional)
        """
        if not self.has_position():
            print("WARNING: Attempted to log trade with no open position")
            return

        # Calculate P&L
        pnl = self.calculate_pnl(exit_price)
        if pnl is None:
            print("ERROR: Could not calculate P&L")
            return

        # Calculate days held
        days_held = (datetime.now().date() - self.entry_date.date()).days

        trade_record = {
            'entry_date': self.entry_date.strftime('%Y-%m-%d'),
            'exit_date': datetime.now().strftime('%Y-%m-%d'),
            'direction': 'LONG' if self.position == 1 else 'SHORT',
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'position_size': self.position_size,
            'pnl_pct': round(pnl['pnl_pct'], 4),
            'pnl_dollars': round(pnl['pnl_dollars'], 2),
            'outcome': 'WIN' if pnl['pnl_pct'] > 0 else 'LOSS',
            'exit_reason': exit_reason,
            'days_held': days_held,
            'prediction': round(prediction, 6) if prediction is not None else None,
            'signal': self.position
        }

        # Create DataFrame and append to CSV
        trade_df = pd.DataFrame([trade_record])

        if os.path.exists(self.trade_log_file):
            trade_df.to_csv(self.trade_log_file, mode='a', header=False, index=False)
        else:
            trade_df.to_csv(self.trade_log_file, mode='w', header=True, index=False)

        print(f"\nTrade logged to {self.trade_log_file}")
        print(f"P&L: {pnl['pnl_pct']:.2f}% (${pnl['pnl_dollars']:.2f})")

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
