"""
Backtest with percentage-based stop-loss and take-profit ONLY.
No signal change exits - hold until stop hit or period ends.
"""

import pandas as pd
import numpy as np
import pickle
import json

print("BACKTESTING WITH PERCENTAGE-BASED EXITS ONLY")
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


def backtest_percent_exits_only(actuals, signals, df_prices, test_indices,
                                initial_capital, stop_loss_pct=None, take_profit_pct=None,
                                transaction_cost_pct=0.0002):
    """
    Backtest with ONLY percentage-based exits. Ignores signal changes.

    Args:
        stop_loss_pct: Exit if loss reaches this percentage (e.g., 0.0075 = 0.75%)
        take_profit_pct: Exit if profit reaches this percentage (e.g., 0.0125 = 1.25%)
    """
    capital = initial_capital
    position = 0
    entry_price = 0.0
    entry_capital = 0.0

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

        # Check percentage-based stops if we have a position
        if position != 0:
            # Calculate current P&L percentage
            if position == 1:
                # Long position
                pct_high = (high_price - entry_price) / entry_price
                pct_low = (low_price - entry_price) / entry_price
            else:
                # Short position
                pct_high = (entry_price - low_price) / entry_price
                pct_low = (entry_price - high_price) / entry_price

            # Check if stop-loss or take-profit hit
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if stop_loss_pct is not None and pct_low <= -stop_loss_pct:
                # Stop-loss hit
                exit_triggered = True
                exit_reason = 'stop_loss'
                if position == 1:
                    exit_price = entry_price * (1 - stop_loss_pct)
                else:
                    exit_price = entry_price * (1 + stop_loss_pct)

            elif take_profit_pct is not None and pct_high >= take_profit_pct:
                # Take-profit hit
                exit_triggered = True
                exit_reason = 'take_profit'
                if position == 1:
                    exit_price = entry_price * (1 + take_profit_pct)
                else:
                    exit_price = entry_price * (1 - take_profit_pct)

            if exit_triggered:
                # Calculate P&L
                if position == 1:
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                pnl = entry_capital * pnl_pct
                cost = capital * transaction_cost_pct
                pnl -= cost
                capital += pnl

                trade_return = pnl / (capital - pnl)
                returns.append(trade_return)
                trades.append({
                    'type': 'long' if position == 1 else 'short',
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl': pnl,
                    'return': trade_return,
                    'exit_reason': exit_reason
                })

                position = 0

        # Enter new position (ONLY when we don't have a position)
        # Signal changes are IGNORED while in a position
        if position == 0 and signal != 0:
            cost = capital * transaction_cost_pct
            capital -= cost
            entry_price = open_price
            position = signal
            entry_capital = capital

        equity_curve.append(capital)

    # Close final position at end of period
    if position != 0:
        exit_price = closes[-1]

        if position == 1:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        pnl = entry_capital * pnl_pct
        cost = capital * transaction_cost_pct
        pnl -= cost
        capital += pnl

        trade_return = pnl / (capital - pnl)
        returns.append(trade_return)
        trades.append({
            'type': 'long' if position == 1 else 'short',
            'entry': entry_price,
            'exit': exit_price,
            'pnl_pct': pnl_pct,
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


def backtest_all_windows_percent_exits(results, df_clean, windows,
                                      stop_loss_pct, take_profit_pct,
                                      initial_capital=1000.0, transaction_cost=0.0002):
    """Backtest all windows with percentage-based exits only."""
    capital = initial_capital
    all_backtests = []

    for window_idx, result in enumerate(results):
        window = windows[window_idx]
        test_start = window['test_start']
        test_end = window['test_end']
        test_indices = range(test_start, test_end)

        # Generate signals
        signals, q1, q3 = generate_signals_regression(result['predictions'])

        backtest = backtest_percent_exits_only(
            result['actuals'],
            signals,
            df_clean,
            test_indices,
            capital,
            stop_loss_pct,
            take_profit_pct,
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

# Test different percentage-based stop configurations
# Focus on asymmetric risk/reward
test_configs = [
    # (stop_loss_pct, take_profit_pct, description)
    (None, None, "No stops (original)"),
    (0.0075, 0.0125, "0.75% loss, 1.25% profit (1.67:1)"),
    (0.0075, 0.0150, "0.75% loss, 1.50% profit (2:1)"),
    (0.0075, 0.0100, "0.75% loss, 1.00% profit (1.33:1)"),
    (0.0100, 0.0150, "1.00% loss, 1.50% profit (1.5:1)"),
    (0.0100, 0.0125, "1.00% loss, 1.25% profit (1.25:1)"),
    (0.0050, 0.0100, "0.50% loss, 1.00% profit (2:1)"),
    (0.0050, 0.0125, "0.50% loss, 1.25% profit (2.5:1)"),
    (0.0060, 0.0120, "0.60% loss, 1.20% profit (2:1)"),
    (0.0080, 0.0160, "0.80% loss, 1.60% profit (2:1)"),
]

print("\n" + "="*80)
print("TESTING PERCENTAGE-BASED EXITS (NO SIGNAL CHANGE EXITS)")
print("="*80)

results_summary = []

for stop_loss_pct, take_profit_pct, description in test_configs:
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    if stop_loss_pct:
        print(f"  Stop-loss: {stop_loss_pct*100:.2f}%")
    else:
        print(f"  Stop-loss: None")
    if take_profit_pct:
        print(f"  Take-profit: {take_profit_pct*100:.2f}%")
    else:
        print(f"  Take-profit: None")
    print(f"{'='*80}")

    backtests = backtest_all_windows_percent_exits(
        all_results,
        df_clean,
        windows,
        stop_loss_pct,
        take_profit_pct,
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
        for reason, count in sorted(metrics['exit_reasons'].items()):
            pct = count / metrics['n_trades'] * 100
            print(f"  {reason:20s}: {count:4d} ({pct:5.1f}%)")

        results_summary.append({
            'config': description,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            **metrics
        })

# Display comparison table
print("\n" + "="*80)
print("COMPARISON TABLE")
print("="*80)

# Sort by Sharpe ratio
results_summary_sorted = sorted(results_summary, key=lambda x: x['sharpe_ratio'], reverse=True)

print(f"\n{'Config':<35} {'Return':>8} {'Annual':>8} {'Sharpe':>8} {'Max DD':>8} {'Win%':>6} {'Trades':>7}")
print("-" * 105)

for r in results_summary_sorted:
    print(f"{r['config']:<35} "
          f"{r['total_return']*100:7.1f}% "
          f"{r['annualized_return']*100:7.2f}% "
          f"{r['sharpe_ratio']:7.3f} "
          f"{r['max_drawdown']*100:7.1f}% "
          f"{r['win_rate']*100:5.1f}% "
          f"{r['n_trades']:6d}")

# Save results
output_file = 'percent_exits_results.json'
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
if best_sharpe['stop_loss_pct']:
    print(f"  Stop-loss: {best_sharpe['stop_loss_pct']*100:.2f}%")
else:
    print(f"  Stop-loss: None")
if best_sharpe['take_profit_pct']:
    print(f"  Take-profit: {best_sharpe['take_profit_pct']*100:.2f}%")
else:
    print(f"  Take-profit: None")
print(f"  Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f}")
print(f"  Annualized Return: {best_sharpe['annualized_return']*100:.2f}%")
print(f"  Risk/Reward Ratio: {best_sharpe['take_profit_pct']/best_sharpe['stop_loss_pct']:.2f}:1" if best_sharpe['stop_loss_pct'] else "")
print(f"{'='*80}")
