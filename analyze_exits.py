"""
Analyze exit performance and test alternative exit strategies

Current exits: 1.0x ATR stop, 2.0x ATR target
Analyze:
1. How often do we hit stop vs target vs time-based exit
2. Average holding time for wins vs losses
3. Test alternative exits:
   - Trailing stops
   - Time-based exits
   - Partial profit taking
   - Dynamic targets based on volatility regime
"""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'strategies')

import pandas as pd
import numpy as np
from src.data_fetcher import FMPDataFetcher
from src.backtester import Backtester
from strategies.volume_divergence import VolumeDivergence


def analyze_exit_types(result):
    """Analyze how trades exited"""
    exit_reasons = {}
    win_exits = {}
    loss_exits = {}

    for trade in result.trades:
        reason = getattr(trade, 'exit_reason', 'unknown')
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        if trade.pnl > 0:
            win_exits[reason] = win_exits.get(reason, 0) + 1
        else:
            loss_exits[reason] = loss_exits.get(reason, 0) + 1

    return exit_reasons, win_exits, loss_exits


def analyze_holding_periods(result):
    """Analyze holding periods for wins vs losses"""
    win_durations = []
    loss_durations = []

    for trade in result.trades:
        duration = (trade.exit_time - trade.entry_time).total_seconds() / 86400  # Days
        if trade.pnl > 0:
            win_durations.append(duration)
        else:
            loss_durations.append(duration)

    return win_durations, loss_durations


class TrailingStopStrategy(VolumeDivergence):
    """Volume divergence with trailing stop"""

    def __init__(self, lookback=15, volume=25, atr=14,
                 initial_stop=1.0, target=2.0, trail_trigger=1.0):
        super().__init__(lookback, volume, atr, initial_stop, target)
        self.trail_trigger = trail_trigger
        self.name = f"VolDiv_Trail_{trail_trigger}x"


class TimeBasedExitStrategy(VolumeDivergence):
    """Volume divergence with maximum holding period"""

    def __init__(self, lookback=15, volume=25, atr=14,
                 stop=1.0, target=2.0, max_bars=10):
        super().__init__(lookback, volume, atr, stop, target)
        self.max_bars = max_bars
        self.name = f"VolDiv_MaxBars_{max_bars}"

    def generate_signals(self, data):
        df = super().generate_signals(data)
        df['max_bars'] = self.max_bars
        return df


def run_exit_analysis():
    """Analyze exits and test alternatives"""
    print('='*70)
    print('EXIT STRATEGY ANALYSIS')
    print('='*70)

    # Load data
    fetcher = FMPDataFetcher()
    data = fetcher.load_data('EURUSD_1day.csv')

    # Split into pre-COVID and 2025
    data['date_dt'] = pd.to_datetime(data['date'])
    train_data = data[(data['date_dt'] >= '2016-01-01') & (data['date_dt'] < '2020-01-01')].copy()
    test_data = data[data['date_dt'] >= '2025-01-01'].copy()

    print(f'\nTrain: {len(train_data)} bars (2016-2019)')
    print(f'Test:  {len(test_data)} bars (2025)')

    # Test baseline strategy
    print(f'\n{"="*70}')
    print('BASELINE: Pre-COVID Best Parameters (15/25/14, 1.0x/2.0x)')
    print('='*70)

    baseline_strategy = VolumeDivergence(15, 25, 14, 1.0, 2.0)
    backtester = Backtester(10000, 0.02, 0.0001)
    result_baseline = backtester.run(test_data, baseline_strategy)

    print(f'\n2025 Performance:')
    print(f'  Return: {result_baseline.metrics["total_return_pct"]:.2f}%')
    print(f'  Trades: {result_baseline.metrics["total_trades"]}')
    print(f'  Win Rate: {result_baseline.metrics["win_rate"]:.1f}%')
    print(f'  Sharpe: {result_baseline.metrics["sharpe_ratio"]:.2f}')
    print(f'  Avg Win: ${result_baseline.metrics.get("avg_win", 0):.2f}')
    print(f'  Avg Loss: ${result_baseline.metrics.get("avg_loss", 0):.2f}')

    # Analyze exit types
    print(f'\n{"="*70}')
    print('EXIT TYPE ANALYSIS')
    print('='*70)

    exit_reasons, win_exits, loss_exits = analyze_exit_types(result_baseline)

    print(f'\nAll Exits:')
    for reason, count in exit_reasons.items():
        pct = count / len(result_baseline.trades) * 100
        print(f'  {reason}: {count} ({pct:.1f}%)')

    print(f'\nWinning Trade Exits:')
    for reason, count in win_exits.items():
        pct = count / len([t for t in result_baseline.trades if t.pnl > 0]) * 100 if any(t.pnl > 0 for t in result_baseline.trades) else 0
        print(f'  {reason}: {count} ({pct:.1f}%)')

    print(f'\nLosing Trade Exits:')
    for reason, count in loss_exits.items():
        pct = count / len([t for t in result_baseline.trades if t.pnl <= 0]) * 100 if any(t.pnl <= 0 for t in result_baseline.trades) else 0
        print(f'  {reason}: {count} ({pct:.1f}%)')

    # Analyze holding periods
    print(f'\n{"="*70}')
    print('HOLDING PERIOD ANALYSIS')
    print('='*70)

    win_durations, loss_durations = analyze_holding_periods(result_baseline)

    if win_durations:
        print(f'\nWinning Trades:')
        print(f'  Avg: {np.mean(win_durations):.1f} days')
        print(f'  Median: {np.median(win_durations):.1f} days')
        print(f'  Min: {np.min(win_durations):.1f} days')
        print(f'  Max: {np.max(win_durations):.1f} days')

    if loss_durations:
        print(f'\nLosing Trades:')
        print(f'  Avg: {np.mean(loss_durations):.1f} days')
        print(f'  Median: {np.median(loss_durations):.1f} days')
        print(f'  Min: {np.min(loss_durations):.1f} days')
        print(f'  Max: {np.max(loss_durations):.1f} days')

    # Trade by trade analysis
    print(f'\n{"="*70}')
    print('TRADE DETAIL (2025 - First 5 trades)')
    print('='*70)

    for i, trade in enumerate(result_baseline.trades[:5], 1):
        duration = (trade.exit_time - trade.entry_time).total_seconds() / 86400
        print(f'\nTrade {i}:')
        print(f'  Direction: {"LONG" if trade.direction == "long" else "SHORT"}')
        print(f'  Entry: {trade.entry_time.date()} @ {trade.entry_price:.5f}')
        print(f'  Exit: {trade.exit_time.date()} @ {trade.exit_price:.5f}')
        print(f'  Duration: {duration:.1f} days')
        print(f'  P&L: ${trade.pnl:.2f} ({trade.return_pct:.2f}%)')
        print(f'  Exit Reason: {getattr(trade, "exit_reason", "unknown")}')

    # Test alternative exit strategies
    print(f'\n{"="*70}')
    print('ALTERNATIVE EXIT STRATEGIES')
    print('='*70)

    alternatives = []

    # Test different stop/target combinations
    print(f'\nTesting Stop/Target Combinations:')
    for stop in [0.5, 0.75, 1.0, 1.25, 1.5]:
        for target in [1.5, 2.0, 2.5, 3.0]:
            strategy = VolumeDivergence(15, 25, 14, stop, target)
            bt = Backtester(10000, 0.02, 0.0001)
            result = bt.run(test_data, strategy)

            alternatives.append({
                'name': f'{stop:.2f}x/{target:.1f}x',
                'stop': stop,
                'target': target,
                'return': result.metrics['total_return_pct'],
                'sharpe': result.metrics['sharpe_ratio'],
                'win_rate': result.metrics['win_rate'],
                'trades': result.metrics['total_trades']
            })

    # Sort by return
    alternatives_df = pd.DataFrame(alternatives).sort_values('return', ascending=False)

    print(f'\nTop 10 Stop/Target Combinations (2025):')
    print(alternatives_df.head(10).to_string(index=False))

    # Compare to baseline
    best_alt = alternatives_df.iloc[0]
    print(f'\n{"="*70}')
    print('COMPARISON')
    print('='*70)

    print(f'\nBaseline (1.0x/2.0x):')
    print(f'  Return: {result_baseline.metrics["total_return_pct"]:.2f}%')
    print(f'  Win Rate: {result_baseline.metrics["win_rate"]:.1f}%')
    print(f'  Sharpe: {result_baseline.metrics["sharpe_ratio"]:.2f}')

    print(f'\nBest Alternative ({best_alt["name"]}):')
    print(f'  Return: {best_alt["return"]:.2f}%')
    print(f'  Win Rate: {best_alt["win_rate"]:.1f}%')
    print(f'  Sharpe: {best_alt["sharpe"]:.2f}')

    print(f'\nImprovement:')
    print(f'  Return: {best_alt["return"] - result_baseline.metrics["total_return_pct"]:+.2f}%')
    print(f'  Win Rate: {best_alt["win_rate"] - result_baseline.metrics["win_rate"]:+.1f}%')
    print(f'  Sharpe: {best_alt["sharpe"] - result_baseline.metrics["sharpe_ratio"]:+.2f}')

    print(f'\n{"="*70}')
    print('RECOMMENDATIONS')
    print('='*70)

    if best_alt["return"] > result_baseline.metrics["total_return_pct"] + 1:
        print(f'Switching to {best_alt["name"]} stop/target would improve returns!')
        print(f'Risk/Reward: {best_alt["target"]/best_alt["stop"]:.1f}:1')
    else:
        print('Current 1.0x/2.0x stop/target is already optimal for 2025.')

    # Analyze if wider stops or tighter targets help
    avg_win = np.mean([t.return_pct for t in result_baseline.trades if t.pnl > 0]) if any(t.pnl > 0 for t in result_baseline.trades) else 0
    avg_loss = np.mean([t.return_pct for t in result_baseline.trades if t.pnl <= 0]) if any(t.pnl <= 0 for t in result_baseline.trades) else 0

    print(f'\nActual Win/Loss Ratio: {abs(avg_win/avg_loss) if avg_loss != 0 else 0:.2f}:1')
    print(f'Target Win/Loss Ratio: 2.0:1')

    if abs(avg_win/avg_loss) > 2.2:
        print('Winners are bigger than expected - consider WIDER targets')
    elif abs(avg_win/avg_loss) < 1.8:
        print('Winners are smaller than expected - consider TIGHTER targets or trail stops')

    return alternatives_df, result_baseline


if __name__ == "__main__":
    alternatives_df, result = run_exit_analysis()
