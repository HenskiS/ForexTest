"""
Backtest with signal-change based exits.
Exit when model prediction changes rather than fixed P/L stops.

Exit rules:
1. Exit long when signal changes to short or neutral
2. Exit short when signal changes to long or neutral
3. Optional: Add safety stop-loss to prevent catastrophic losses
"""

import pandas as pd
import numpy as np
import pickle
import json

print("BACKTESTING WITH SIGNAL CHANGE EXITS")
print("="*80)

# Load data and results
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

# Drop rows where technical features are NaN (same as training script)
technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr', 'target_5day_return'
]
df_clean = df.dropna(subset=technical_features)

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


def generate_signals_regression(predictions, lower_percentile=25, upper_percentile=75):
    """Generate signals from regression predictions with configurable thresholds."""
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    lower_threshold = np.percentile(predictions, lower_percentile)
    upper_threshold = np.percentile(predictions, upper_percentile)
    signals[predictions >= upper_threshold] = 1
    signals[predictions <= lower_threshold] = -1

    return signals, lower_threshold, upper_threshold


def backtest_signal_change_exits(actuals, signals, df_prices, test_indices,
                                  initial_capital,
                                  exit_on_neutral=True,
                                  safety_stop_pct=None,
                                  transaction_cost_pct=0.0002):
    """
    Backtest with signal-change based exits.

    Args:
        exit_on_neutral: If True, exit when signal becomes neutral (0)
        safety_stop_pct: Optional safety stop-loss percentage (e.g., 0.05 = 5%)
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

        # Check exits if we have a position
        if position != 0:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # Check safety stop-loss first (if enabled)
            if safety_stop_pct is not None:
                if position == 1:
                    pct_low = (low_price - entry_price) / entry_price
                    if pct_low <= -safety_stop_pct:
                        exit_triggered = True
                        exit_reason = 'safety_stop'
                        exit_price = entry_price * (1 - safety_stop_pct)
                else:  # position == -1
                    pct_high = (entry_price - high_price) / entry_price
                    if pct_high <= -safety_stop_pct:
                        exit_triggered = True
                        exit_reason = 'safety_stop'
                        exit_price = entry_price * (1 + safety_stop_pct)

            # Check signal change exit
            if not exit_triggered:
                signal_changed = False

                if position == 1:  # Long position
                    if signal == -1:  # Signal flipped to short
                        signal_changed = True
                    elif signal == 0 and exit_on_neutral:  # Signal went neutral
                        signal_changed = True
                elif position == -1:  # Short position
                    if signal == 1:  # Signal flipped to long
                        signal_changed = True
                    elif signal == 0 and exit_on_neutral:  # Signal went neutral
                        signal_changed = True

                if signal_changed:
                    exit_triggered = True
                    exit_reason = 'signal_change'
                    exit_price = open_price

            # Execute exit if triggered
            if exit_triggered:
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

        # Enter new position (only when we don't have a position)
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


def backtest_all_windows_signal_change(results, df_clean, windows, config,
                                       initial_capital=1000.0, transaction_cost=0.0002):
    """Backtest all windows with signal change exits."""
    capital = initial_capital
    all_backtests = []

    for window_idx, result in enumerate(results):
        window = windows[window_idx]
        test_start = window['test_start']
        test_end = window['test_end']
        test_indices = range(test_start, test_end)

        # Generate signals with configurable thresholds
        signals, lower_thresh, upper_thresh = generate_signals_regression(
            result['predictions'],
            lower_percentile=config.get('lower_percentile', 25),
            upper_percentile=config.get('upper_percentile', 75)
        )

        backtest = backtest_signal_change_exits(
            result['actuals'],
            signals,
            df_clean,
            test_indices,
            capital,
            exit_on_neutral=config['exit_on_neutral'],
            safety_stop_pct=config.get('safety_stop_pct'),
            transaction_cost_pct=transaction_cost
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

# Test configurations - testing different confidence thresholds
test_configs = [
    # Baseline (25/75 percentile - Q1/Q3)
    {
        'name': 'Q1/Q3 (25/75%ile) - exit on neutral',
        'lower_percentile': 25,
        'upper_percentile': 75,
        'exit_on_neutral': True,
        'safety_stop_pct': None
    },
    {
        'name': 'Q1/Q3 (25/75%ile) - no neutral exit',
        'lower_percentile': 25,
        'upper_percentile': 75,
        'exit_on_neutral': False,
        'safety_stop_pct': None
    },

    # Higher confidence (20/80)
    {
        'name': '20/80%ile - exit on neutral',
        'lower_percentile': 20,
        'upper_percentile': 80,
        'exit_on_neutral': True,
        'safety_stop_pct': None
    },
    {
        'name': '20/80%ile - no neutral exit',
        'lower_percentile': 20,
        'upper_percentile': 80,
        'exit_on_neutral': False,
        'safety_stop_pct': None
    },

    # Higher confidence (15/85)
    {
        'name': '15/85%ile - exit on neutral',
        'lower_percentile': 15,
        'upper_percentile': 85,
        'exit_on_neutral': True,
        'safety_stop_pct': None
    },
    {
        'name': '15/85%ile - no neutral exit',
        'lower_percentile': 15,
        'upper_percentile': 85,
        'exit_on_neutral': False,
        'safety_stop_pct': None
    },

    # High confidence (10/90)
    {
        'name': '10/90%ile - exit on neutral',
        'lower_percentile': 10,
        'upper_percentile': 90,
        'exit_on_neutral': True,
        'safety_stop_pct': None
    },
    {
        'name': '10/90%ile - no neutral exit',
        'lower_percentile': 10,
        'upper_percentile': 90,
        'exit_on_neutral': False,
        'safety_stop_pct': None
    },

    # Very high confidence (5/95)
    {
        'name': '5/95%ile - exit on neutral',
        'lower_percentile': 5,
        'upper_percentile': 95,
        'exit_on_neutral': True,
        'safety_stop_pct': None
    },
    {
        'name': '5/95%ile - no neutral exit',
        'lower_percentile': 5,
        'upper_percentile': 95,
        'exit_on_neutral': False,
        'safety_stop_pct': None
    },

    # Extreme confidence (2/98)
    {
        'name': '2/98%ile - exit on neutral',
        'lower_percentile': 2,
        'upper_percentile': 98,
        'exit_on_neutral': True,
        'safety_stop_pct': None
    },
    {
        'name': '2/98%ile - no neutral exit',
        'lower_percentile': 2,
        'upper_percentile': 98,
        'exit_on_neutral': False,
        'safety_stop_pct': None
    },

    # Best thresholds with safety stops
    {
        'name': '10/90%ile + 3% safety',
        'lower_percentile': 10,
        'upper_percentile': 90,
        'exit_on_neutral': True,
        'safety_stop_pct': 0.03
    },
    {
        'name': '5/95%ile + 3% safety',
        'lower_percentile': 5,
        'upper_percentile': 95,
        'exit_on_neutral': True,
        'safety_stop_pct': 0.03
    },
    {
        'name': '10/90%ile + 5% safety',
        'lower_percentile': 10,
        'upper_percentile': 90,
        'exit_on_neutral': True,
        'safety_stop_pct': 0.05
    },
    {
        'name': '5/95%ile + 5% safety',
        'lower_percentile': 5,
        'upper_percentile': 95,
        'exit_on_neutral': True,
        'safety_stop_pct': 0.05
    },
]

print("\n" + "="*80)
print("TESTING SIGNAL CHANGE EXIT STRATEGIES")
print("="*80)

results_summary = []

for config in test_configs:
    print(f"\n{'='*80}")
    print(f"Testing: {config['name']}")
    print(f"  Thresholds: {config.get('lower_percentile', 25)}/{config.get('upper_percentile', 75)} percentile")
    print(f"  Exit on neutral: {config['exit_on_neutral']}")
    if config.get('safety_stop_pct'):
        print(f"  Safety stop: {config['safety_stop_pct']*100:.2f}%")
    else:
        print(f"  Safety stop: None")
    print(f"{'='*80}")

    backtests = backtest_all_windows_signal_change(
        all_results,
        df_clean,
        windows,
        config,
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
            'config': config['name'],
            **config,
            **metrics
        })

# Display comparison table
print("\n" + "="*80)
print("COMPARISON TABLE (Sorted by Sharpe Ratio)")
print("="*80)

# Sort by Sharpe ratio
results_summary_sorted = sorted(results_summary, key=lambda x: x['sharpe_ratio'], reverse=True)

print(f"\n{'Config':<45} {'Return':>8} {'Annual':>8} {'Sharpe':>8} {'Max DD':>8} {'Win%':>6} {'Trades':>7}")
print("-" * 115)

for r in results_summary_sorted:
    print(f"{r['config']:<45} "
          f"{r['total_return']*100:7.1f}% "
          f"{r['annualized_return']*100:7.2f}% "
          f"{r['sharpe_ratio']:7.3f} "
          f"{r['max_drawdown']*100:7.1f}% "
          f"{r['win_rate']*100:5.1f}% "
          f"{r['n_trades']:6d}")

# Save results
output_file = 'signal_change_exits_results.json'
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
print(f"  Thresholds: {best_sharpe.get('lower_percentile', 25)}/{best_sharpe.get('upper_percentile', 75)} percentile")
print(f"  Exit on neutral: {best_sharpe['exit_on_neutral']}")
if best_sharpe.get('safety_stop_pct'):
    print(f"  Safety stop: {best_sharpe['safety_stop_pct']*100:.2f}%")
else:
    print(f"  Safety stop: None")
print(f"  Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f}")
print(f"  Annualized Return: {best_sharpe['annualized_return']*100:.2f}%")
print(f"  Total Return: {best_sharpe['total_return']*100:.2f}%")
print(f"  Max Drawdown: {best_sharpe['max_drawdown']*100:.2f}%")
print(f"  Total Trades: {best_sharpe['n_trades']}")

# Compare with optimal fixed stop strategy
print(f"\n{'='*80}")
print("COMPARISON WITH OPTIMAL FIXED STOP STRATEGY")
print(f"{'='*80}")
print(f"\nFixed stops (0.40%/1.00% vol-adjusted):")
print(f"  Sharpe Ratio: 0.676")
print(f"  Annualized Return: 7.32%")
print(f"  Total Return: +311%")
print(f"  Max Drawdown: -16.8%")
print(f"\nBest signal change strategy ({best_sharpe['config']}):")
print(f"  Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f}")
print(f"  Annualized Return: {best_sharpe['annualized_return']*100:.2f}%")
print(f"  Total Return: {best_sharpe['total_return']*100:.1f}%")
print(f"  Max Drawdown: {best_sharpe['max_drawdown']*100:.1f}%")
print(f"{'='*80}")
