"""
True Ensemble Backtester

Runs multiple strategies in parallel (paper trading) and executes whichever
is performing best recently.

How it works:
1. Track all 3 variants: aggressive (0.75x/1.5x), baseline (1.0x/2.0x), conservative (1.25x/2.5x)
2. On each signal, simulate what ALL variants would do with their stops/targets
3. Every N bars, evaluate which variant has best recent Sharpe ratio
4. Execute real trades using the best-performing variant
5. Add hysteresis: need X% improvement to switch (avoid whipsaw)

This way we switch proactively based on what's working NOW, not what worked 10 trades ago.
"""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'strategies')

import pandas as pd
import numpy as np
from src.data_fetcher import FMPDataFetcher
from strategies.volume_divergence import VolumeDivergence


class EnsembleBacktester:
    """
    Backtester that runs multiple strategy variants in parallel and uses the best
    """

    def __init__(self, initial_capital=10000, risk_per_trade=0.02):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade

        # Strategy variants to track
        self.variants = {
            'aggressive': {'stop': 0.75, 'target': 1.5},
            'baseline': {'stop': 1.0, 'target': 2.0},
            'conservative': {'stop': 1.25, 'target': 2.5}
        }

        # Track paper trading for each variant
        self.paper_trades = {name: [] for name in self.variants}

        # Real trades (what we actually execute)
        self.real_trades = []
        self.capital = initial_capital
        self.active_variant = 'baseline'  # Start with baseline

        # Evaluation settings
        self.eval_frequency = 50  # Re-evaluate every 50 bars
        self.eval_window = 20  # Look at last 20 paper trades
        self.switch_threshold = 1.15  # Need 15% better Sharpe to switch

        self.bars_since_eval = 0

    def run(self, data, base_strategy):
        """
        Run ensemble backtest

        Args:
            data: OHLCV DataFrame
            base_strategy: Strategy to generate entry signals (uses default params)
        """
        print(f'Running ensemble backtest on {len(data)} bars...')

        # Generate signals using base strategy
        signals = base_strategy.generate_signals(data.copy())

        # Track open positions for each variant (paper trading)
        open_positions = {name: None for name in self.variants}
        real_position = None

        for idx in range(len(signals)):
            row = signals.iloc[idx]
            self.bars_since_eval += 1

            # Update all open paper positions
            for variant_name, params in self.variants.items():
                pos = open_positions[variant_name]

                if pos:
                    # Check if paper position should exit
                    exit_price, exit_reason = self._check_exit(row, pos)

                    if exit_price:
                        # Close paper position
                        pnl_pct = self._calculate_pnl_pct(pos, exit_price)
                        self.paper_trades[variant_name].append({
                            'entry_date': pos['entry_date'],
                            'exit_date': row['date'],
                            'pnl_pct': pnl_pct,
                            'bars_held': idx - pos['entry_idx']
                        })
                        open_positions[variant_name] = None

            # Check for new entry signal
            if row['signal'] != 0 and not any(open_positions.values()):
                # Open paper positions for all variants
                for variant_name, params in self.variants.items():
                    open_positions[variant_name] = self._open_position(
                        row, idx, params['stop'], params['target']
                    )

            # Check real position
            if real_position:
                exit_price, exit_reason = self._check_exit(row, real_position)

                if exit_price:
                    # Close real position
                    pnl_pct = self._calculate_pnl_pct(real_position, exit_price)
                    pnl = self.capital * pnl_pct
                    self.capital += pnl

                    self.real_trades.append({
                        'entry_date': real_position['entry_date'],
                        'exit_date': row['date'],
                        'variant': self.active_variant,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason
                    })
                    real_position = None

            # Open real position if signal and no position
            if row['signal'] != 0 and not real_position:
                params = self.variants[self.active_variant]
                real_position = self._open_position(
                    row, idx, params['stop'], params['target']
                )

            # Periodically evaluate and switch variants
            if self.bars_since_eval >= self.eval_frequency:
                self._evaluate_and_switch()
                self.bars_since_eval = 0

        # Calculate metrics
        metrics = self._calculate_metrics()

        return metrics

    def _open_position(self, row, idx, stop_mult, target_mult):
        """Open a position with given parameters"""
        if row['signal'] == 0 or pd.isna(row['atr']):
            return None

        direction = 'long' if row['signal'] == 1 else 'short'
        entry_price = row['close']
        atr = row['atr']

        if direction == 'long':
            stop_loss = entry_price - atr * stop_mult
            take_profit = entry_price + atr * target_mult
        else:
            stop_loss = entry_price + atr * stop_mult
            take_profit = entry_price - atr * target_mult

        return {
            'entry_date': row['date'],
            'entry_idx': idx,
            'entry_price': entry_price,
            'direction': direction,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    def _check_exit(self, row, position):
        """Check if position should exit, return (exit_price, reason) or (None, None)"""
        if not position:
            return None, None

        # Check stop loss
        if position['direction'] == 'long' and row['low'] <= position['stop_loss']:
            return position['stop_loss'], 'stop'
        if position['direction'] == 'short' and row['high'] >= position['stop_loss']:
            return position['stop_loss'], 'stop'

        # Check take profit
        if position['direction'] == 'long' and row['high'] >= position['take_profit']:
            return position['take_profit'], 'target'
        if position['direction'] == 'short' and row['low'] <= position['take_profit']:
            return position['take_profit'], 'target'

        return None, None

    def _calculate_pnl_pct(self, position, exit_price):
        """Calculate P&L percentage for a position"""
        if position['direction'] == 'long':
            return (exit_price - position['entry_price']) / position['entry_price']
        else:
            return (position['entry_price'] - exit_price) / position['entry_price']

    def _evaluate_and_switch(self):
        """Evaluate recent paper trading performance and switch if needed"""
        # Calculate recent Sharpe for each variant
        sharpes = {}

        for variant_name, trades in self.paper_trades.items():
            if len(trades) < self.eval_window:
                sharpes[variant_name] = -999  # Not enough data
                continue

            recent = trades[-self.eval_window:]
            returns = [t['pnl_pct'] for t in recent]

            avg_return = np.mean(returns)
            std_return = np.std(returns) if np.std(returns) > 0 else 0.01

            sharpes[variant_name] = avg_return / std_return

        # Find best variant
        best_variant = max(sharpes, key=sharpes.get)
        best_sharpe = sharpes[best_variant]
        current_sharpe = sharpes[self.active_variant]

        # Switch if significantly better
        if best_sharpe > current_sharpe * self.switch_threshold:
            print(f'[SWITCH] {self.active_variant} (Sharpe {current_sharpe:.2f}) -> {best_variant} (Sharpe {best_sharpe:.2f})')
            print(f'  All Sharpes: {", ".join([f"{k}: {v:.2f}" for k, v in sharpes.items()])}')
            self.active_variant = best_variant

    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.real_trades:
            return {
                'total_return_pct': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'total_trades': 0,
                'variant_distribution': {}
            }

        # Overall metrics
        total_return_pct = (self.capital - self.initial_capital) / self.initial_capital * 100
        wins = sum(1 for t in self.real_trades if t['pnl'] > 0)
        win_rate = wins / len(self.real_trades) * 100

        returns = [t['pnl_pct'] for t in self.real_trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Variant distribution
        variant_counts = {}
        for t in self.real_trades:
            variant_counts[t['variant']] = variant_counts.get(t['variant'], 0) + 1

        return {
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'total_trades': len(self.real_trades),
            'variant_distribution': variant_counts,
            'final_capital': self.capital
        }


def test_ensemble_backtester():
    """Test the true ensemble backtester"""
    print('='*70)
    print('TRUE ENSEMBLE BACKTESTER TEST')
    print('='*70)

    fetcher = FMPDataFetcher()
    data = fetcher.load_data('EURUSD_1day.csv')
    data['date_dt'] = pd.to_datetime(data['date'])

    periods = {
        'Pre-COVID (2016-2019)': data[(data['date_dt'] >= '2016-01-01') & (data['date_dt'] < '2020-01-01')],
        'COVID (2020-2021)': data[(data['date_dt'] >= '2020-01-01') & (data['date_dt'] < '2022-01-01')],
        'Recent (2022-2024)': data[(data['date_dt'] >= '2022-01-01') & (data['date_dt'] < '2025-01-01')],
        '2025': data[data['date_dt'] >= '2025-01-01'],
        'All Data (2015-2025)': data
    }

    for period_name, period_data in periods.items():
        if len(period_data) == 0:
            continue

        print(f'\n{"="*70}')
        print(f'{period_name}')
        print('='*70)

        # Run ensemble
        ensemble = EnsembleBacktester(10000, 0.02)
        strategy = VolumeDivergence(15, 25, 14, 1.0, 2.0)
        metrics = ensemble.run(period_data, strategy)

        print(f'\nResults:')
        print(f'  Return: {metrics["total_return_pct"]:.2f}%')
        print(f'  Win Rate: {metrics["win_rate"]:.1f}%')
        print(f'  Sharpe: {metrics["sharpe_ratio"]:.2f}')
        print(f'  Trades: {metrics["total_trades"]}')
        print(f'  Variant usage: {metrics["variant_distribution"]}')


if __name__ == "__main__":
    test_ensemble_backtester()
