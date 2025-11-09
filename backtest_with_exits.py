"""
Backtest with improved exit strategy: stop-loss and take-profit.
Test different risk/reward ratios to improve Sharpe ratio.
"""

import pandas as pd
import numpy as np
import pickle
import json

print("BACKTESTING WITH STOP-LOSS AND TAKE-PROFIT EXITS")
print("="*80)

# Load data and results
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)
df_clean = df.dropna()

with open('xgboost_results_target_5day_return.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} windows of predictions")

# Configuration
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS


def generate_windows(df, window_size, roll_days, min_windows=40):
    """Generate rolling walk-forward windows."""
    windows = []
    start_idx = 0

    while start_idx + window_size <= len(df):
        train_end = start_idx + TRAIN_DAYS
        val_end = train_end + VAL_DAYS
        test_end = val_end + TEST_DAYS

        window = {
            'window_id': len(windows),
            'train_start': start_idx,
            'train_end': train_end,
            'val_start': train_end,
            'val_end': val_end,
            'test_start': val_end,
            'test_end': test_end,
            'date_start': df.index[start_idx],
            'date_end': df.index[test_end - 1]
        }

        windows.append(window)
        start_idx += roll_days

        if len(windows) >= min_windows:
            break

    return windows


def generate_signals_regression(predictions):
    """Generate signals from regression predictions."""
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    signals[predictions >= q3] = 1
    signals[predictions <= q1] = -1

    return signals, q1, q3


def backtest_with_stops(actuals, signals, df_prices, test_indices,
                        initial_capital, stop_loss_pips=None, take_profit_pips=None,
                        transaction_cost_pct=0.0002):
    """
    Backtest with stop-loss and take-profit exits.

    Args:
        stop_loss_pips: Exit if loss reaches this many pips (e.g., 200)
        take_profit_pips: Exit if profit reaches this many pips (e.g., 300)
    """
    PIP_VALUE = 0.1
    PIP_SIZE = 0.0001

    capital = initial_capital
    position = 0
    entry_price = 0.0
    entry_lot_size = 0.0

    equity_curve = [capital]
    returns = []
    trades = []

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values

    for i in range(len(signals)):
        signal = signals[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]

        # Check stop-loss and take-profit if we have a position
        if position != 0:
            # Calculate current P&L in pips
            if position == 1:
                # Long position
                current_pips_high = (high_price - entry_price) / PIP_SIZE
                current_pips_low = (low_price - entry_price) / PIP_SIZE
            else:
                # Short position
                current_pips_high = (entry_price - low_price) / PIP_SIZE
                current_pips_low = (entry_price - high_price) / PIP_SIZE

            # Check if stop-loss or take-profit hit
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if stop_loss_pips is not None and current_pips_low <= -stop_loss_pips:
                # Stop-loss hit
                exit_triggered = True
                exit_reason = 'stop_loss'
                # Use the price that would hit the stop
                if position == 1:
                    exit_price = entry_price - (stop_loss_pips * PIP_SIZE)
                else:
                    exit_price = entry_price + (stop_loss_pips * PIP_SIZE)

            elif take_profit_pips is not None and current_pips_high >= take_profit_pips:
                # Take-profit hit
                exit_triggered = True
                exit_reason = 'take_profit'
                # Use the price that would hit the target
                if position == 1:
                    exit_price = entry_price + (take_profit_pips * PIP_SIZE)
                else:
                    exit_price = entry_price - (take_profit_pips * PIP_SIZE)

            if exit_triggered:
                # Exit position
                if position == 1:
                    pips = (exit_price - entry_price) / PIP_SIZE
                else:
                    pips = (entry_price - exit_price) / PIP_SIZE

                pnl = pips * PIP_VALUE * entry_lot_size / 1000
                cost = capital * transaction_cost_pct
                pnl -= cost
                capital += pnl

                trade_return = pnl / (capital - pnl)
                returns.append(trade_return)
                trades.append({
                    'type': 'long' if position == 1 else 'short',
                    'entry': entry_price,
                    'exit': exit_price,
                    'pips': pips,
                    'lot_size': entry_lot_size,
                    'pnl': pnl,
                    'return': trade_return,
                    'exit_reason': exit_reason
                })

                position = 0

        # Exit current position if signal changes
        if position != 0 and signal != position:
            exit_price = open_price

            if position == 1:
                pips = (exit_price - entry_price) / PIP_SIZE
            else:
                pips = (entry_price - exit_price) / PIP_SIZE

            pnl = pips * PIP_VALUE * entry_lot_size / 1000
            cost = capital * transaction_cost_pct
            pnl -= cost
            capital += pnl

            trade_return = pnl / (capital - pnl)
            returns.append(trade_return)
            trades.append({
                'type': 'long' if position == 1 else 'short',
                'entry': entry_price,
                'exit': exit_price,
                'pips': pips,
                'lot_size': entry_lot_size,
                'pnl': pnl,
                'return': trade_return,
                'exit_reason': 'signal_change'
            })

            position = 0

        # Enter new position
        if position == 0 and signal != 0:
            cost = capital * transaction_cost_pct
            capital -= cost
            entry_price = open_price
            position = signal
            entry_lot_size = capital

        equity_curve.append(capital)

    # Close final position
    if position != 0:
        exit_price = closes[-1]

        if position == 1:
            pips = (exit_price - entry_price) / PIP_SIZE
        else:
            pips = (entry_price - exit_price) / PIP_SIZE

        pnl = pips * PIP_VALUE * entry_lot_size / 1000
        cost = capital * transaction_cost_pct
        pnl -= cost
        capital += pnl

        trade_return = pnl / (capital - pnl)
        returns.append(trade_return)
        trades.append({
            'type': 'long' if position == 1 else 'short',
            'entry': entry_price,
            'exit': exit_price,
            'pips': pips,
            'lot_size': entry_lot_size,
            'pnl': pnl,
            'return': trade_return,
            'exit_reason': 'end_of_period'
        })

    final_return = (capital - initial_capital) / initial_capital
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]

    # Count exit reasons
    exit_reasons = {}
    for t in trades:
        reason = t.get('exit_reason', 'unknown')
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'strategy_returns': returns,
        'final_capital': capital,
        'final_return': final_return,
        'n_trades': len(trades),
        'n_long': len([t for t in trades if t['type'] == 'long']),
        'n_short': len([t for t in trades if t['type'] == 'short']),
        'n_winning': len(winning_trades),
        'n_losing': len(losing_trades),
        'exit_reasons': exit_reasons
    }


def backtest_all_windows_with_stops(results, df_clean, windows,
                                    stop_loss_pips, take_profit_pips,
                                    initial_capital=1000.0, transaction_cost=0.0002):
    """Backtest all windows with stop-loss and take-profit."""
    capital = initial_capital
    all_backtests = []

    for window_idx, result in enumerate(results):
        window = windows[window_idx]
        test_start = window['test_start']
        test_end = window['test_end']
        test_indices = range(test_start, test_end)

        # Generate signals
        signals, q1, q3 = generate_signals_regression(result['predictions'])

        backtest = backtest_with_stops(
            result['actuals'],
            signals,
            df_clean,
            test_indices,
            capital,
            stop_loss_pips,
            take_profit_pips,
            transaction_cost
        )
        all_backtests.append(backtest)
        capital = backtest['final_capital']

    return all_backtests


def calculate_performance_metrics(backtests):
    """Calculate comprehensive performance metrics."""
    all_returns = []
    for bt in backtests:
        all_returns.extend(bt['strategy_returns'])

    if len(all_returns) == 0:
        return None

    all_returns = np.array(all_returns)

    equity_curve = np.cumprod(1 + all_returns)
    total_return = equity_curve[-1] - 1

    n_days = len(backtests) * TEST_DAYS
    n_years = n_days / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1

    volatility = np.std(all_returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = np.min(drawdown)

    winning_trades = all_returns[all_returns > 0]
    losing_trades = all_returns[all_returns < 0]

    win_rate = len(winning_trades) / len(all_returns[all_returns != 0]) if len(all_returns[all_returns != 0]) > 0 else 0
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
    avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0

    total_wins = np.sum(winning_trades)
    total_losses = np.abs(np.sum(losing_trades))
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

    # Aggregate exit reasons
    all_exit_reasons = {}
    for bt in backtests:
        for reason, count in bt['exit_reasons'].items():
            all_exit_reasons[reason] = all_exit_reasons.get(reason, 0) + count

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'n_trades': len(all_returns[all_returns != 0]),
        'n_winning': len(winning_trades),
        'n_losing': len(losing_trades),
        'exit_reasons': all_exit_reasons,
        'final_capital': backtests[-1]['final_capital']
    }


# Generate windows
windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS, min_windows=40)

# Test different stop-loss and take-profit combinations
test_configs = [
    # (stop_loss_pips, take_profit_pips, description)
    (None, None, "No stops (original)"),
    (150, 150, "1:1 risk/reward (150 pips)"),
    (150, 225, "1:1.5 risk/reward"),
    (150, 300, "1:2 risk/reward"),
    (200, 300, "Wider stops 1:1.5"),
    (200, 400, "Wider stops 1:2"),
    (100, 150, "Tighter stops 1:1.5"),
    (100, 200, "Tighter stops 1:2"),
]

print("\n" + "="*80)
print("TESTING DIFFERENT STOP-LOSS AND TAKE-PROFIT CONFIGURATIONS")
print("="*80)

results_summary = []

for stop_loss, take_profit, description in test_configs:
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"  Stop-loss: {stop_loss} pips" if stop_loss else "  Stop-loss: None")
    print(f"  Take-profit: {take_profit} pips" if take_profit else "  Take-profit: None")
    print(f"{'='*80}")

    backtests = backtest_all_windows_with_stops(
        all_results,
        df_clean,
        windows,
        stop_loss,
        take_profit,
        initial_capital=1000.0,
        transaction_cost=0.0002
    )

    metrics = calculate_performance_metrics(backtests)

    if metrics:
        print(f"\nPerformance:")
        print(f"  Final Capital:       ${metrics['final_capital']:.2f}")
        print(f"  Total Return:        {metrics['total_return']*100:8.2f}%")
        print(f"  Annualized Return:   {metrics['annualized_return']*100:8.2f}%")
        print(f"  Volatility (annual): {metrics['volatility']*100:8.2f}%")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:8.3f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']*100:8.2f}%")
        print(f"  Win Rate:            {metrics['win_rate']*100:8.2f}%")
        print(f"  Average Win:         {metrics['avg_win']*100:8.4f}%")
        print(f"  Average Loss:        {metrics['avg_loss']*100:8.4f}%")
        print(f"  Profit Factor:       {metrics['profit_factor']:8.3f}")
        print(f"  Total Trades:        {metrics['n_trades']:8d}")

        print(f"\nExit Reasons:")
        for reason, count in metrics['exit_reasons'].items():
            pct = count / metrics['n_trades'] * 100
            print(f"  {reason:20s}: {count:4d} ({pct:5.1f}%)")

        results_summary.append({
            'config': description,
            'stop_loss_pips': stop_loss,
            'take_profit_pips': take_profit,
            **metrics
        })

# Display comparison table
print("\n" + "="*80)
print("COMPARISON TABLE")
print("="*80)

# Sort by Sharpe ratio
results_summary_sorted = sorted(results_summary, key=lambda x: x['sharpe_ratio'], reverse=True)

print(f"\n{'Config':<30} {'Return':>8} {'Annual':>8} {'Sharpe':>8} {'Max DD':>8} {'Win%':>6} {'Trades':>7}")
print("-" * 100)

for r in results_summary_sorted:
    print(f"{r['config']:<30} "
          f"{r['total_return']*100:7.1f}% "
          f"{r['annualized_return']*100:7.2f}% "
          f"{r['sharpe_ratio']:7.3f} "
          f"{r['max_drawdown']*100:7.1f}% "
          f"{r['win_rate']*100:5.1f}% "
          f"{r['n_trades']:6d}")

# Save results
output_file = 'stop_loss_take_profit_results.json'
with open(output_file, 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    results_json = []
    for r in results_summary:
        r_copy = {k: (int(v) if isinstance(v, (np.integer, np.int64)) else
                      float(v) if isinstance(v, (np.floating, np.float64)) else v)
                  for k, v in r.items() if k != 'exit_reasons'}
        r_copy['exit_reasons'] = {k: int(v) for k, v in r['exit_reasons'].items()}
        results_json.append(r_copy)
    json.dump(results_json, f, indent=2)

print(f"\n{'='*80}")
print(f"Results saved to: {output_file}")
print(f"{'='*80}")

# Identify best configuration
best_sharpe = results_summary_sorted[0]
print(f"\nBEST CONFIGURATION (by Sharpe Ratio):")
print(f"  {best_sharpe['config']}")
print(f"  Stop-loss: {best_sharpe['stop_loss_pips']} pips" if best_sharpe['stop_loss_pips'] else "  Stop-loss: None")
print(f"  Take-profit: {best_sharpe['take_profit_pips']} pips" if best_sharpe['take_profit_pips'] else "  Take-profit: None")
print(f"  Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f}")
print(f"  Annualized Return: {best_sharpe['annualized_return']*100:.2f}%")
print(f"{'='*80}")
