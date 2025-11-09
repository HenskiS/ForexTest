"""
Advanced exit strategies:
1. Volatility-adjusted stop-loss and take-profit
2. Trailing stops
3. Combination of both
"""

import pandas as pd
import numpy as np
import pickle
import json

print("ADVANCED EXIT STRATEGIES BACKTEST")
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


def generate_signals_regression(predictions):
    """Generate signals from regression predictions."""
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    signals[predictions >= q3] = 1
    signals[predictions <= q1] = -1

    return signals, q1, q3


def backtest_advanced_exits(actuals, signals, df_prices, test_indices,
                           initial_capital,
                           base_stop_loss_pct=0.0050,
                           base_take_profit_pct=0.0125,
                           use_vol_adjustment=False,
                           vol_window=20,
                           use_trailing_stop=False,
                           trail_activate_pct=0.0075,
                           trail_distance_pct=0.0025,
                           transaction_cost_pct=0.0002):
    """
    Advanced backtest with volatility adjustment and/or trailing stops.

    Args:
        base_stop_loss_pct: Base stop-loss percentage
        base_take_profit_pct: Base take-profit percentage
        use_vol_adjustment: If True, scale stops by volatility
        vol_window: Window for volatility calculation
        use_trailing_stop: If True, use trailing stop
        trail_activate_pct: Profit level to activate trailing stop
        trail_distance_pct: Distance to trail behind current price
    """
    capital = initial_capital
    position = 0
    entry_price = 0.0
    entry_capital = 0.0
    trailing_stop_price = None

    equity_curve = [capital]
    returns = []
    trades = []

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values

    # Get ATR for volatility adjustment
    if use_vol_adjustment:
        atr = test_data['atr'].values
        # Calculate median ATR for normalization
        median_atr = np.median(atr[~np.isnan(atr)])

    for i in range(len(signals)):
        signal = signals[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]

        # Adjust stops by volatility if enabled
        if use_vol_adjustment and not np.isnan(atr[i]):
            vol_ratio = atr[i] / median_atr
            stop_loss_pct = base_stop_loss_pct * vol_ratio
            take_profit_pct = base_take_profit_pct * vol_ratio
        else:
            stop_loss_pct = base_stop_loss_pct
            take_profit_pct = base_take_profit_pct

        # Check stops if we have a position
        if position != 0:
            # Calculate current P&L percentage
            if position == 1:
                pct_high = (high_price - entry_price) / entry_price
                pct_low = (low_price - entry_price) / entry_price
                current_price_for_trail = high_price
            else:
                pct_high = (entry_price - low_price) / entry_price
                pct_low = (entry_price - high_price) / entry_price
                current_price_for_trail = low_price

            # Update trailing stop if enabled and activated
            if use_trailing_stop and pct_high >= trail_activate_pct:
                if trailing_stop_price is None:
                    # Activate trailing stop at breakeven
                    trailing_stop_price = entry_price
                else:
                    # Trail the stop
                    if position == 1:
                        new_stop = current_price_for_trail - (entry_price * trail_distance_pct)
                        trailing_stop_price = max(trailing_stop_price, new_stop)
                    else:
                        new_stop = current_price_for_trail + (entry_price * trail_distance_pct)
                        trailing_stop_price = min(trailing_stop_price, new_stop)

            # Check exits
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # Check trailing stop first (if active)
            if use_trailing_stop and trailing_stop_price is not None:
                if position == 1 and low_price <= trailing_stop_price:
                    exit_triggered = True
                    exit_price = trailing_stop_price
                    exit_reason = 'trailing_stop'
                elif position == -1 and high_price >= trailing_stop_price:
                    exit_triggered = True
                    exit_price = trailing_stop_price
                    exit_reason = 'trailing_stop'

            # Check regular stop-loss
            if not exit_triggered and pct_low <= -stop_loss_pct:
                exit_triggered = True
                exit_reason = 'stop_loss'
                if position == 1:
                    exit_price = entry_price * (1 - stop_loss_pct)
                else:
                    exit_price = entry_price * (1 + stop_loss_pct)

            # Check take-profit
            elif not exit_triggered and pct_high >= take_profit_pct:
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
                trailing_stop_price = None

        # Enter new position
        if position == 0 and signal != 0:
            cost = capital * transaction_cost_pct
            capital -= cost
            entry_price = open_price
            position = signal
            entry_capital = capital
            trailing_stop_price = None

        equity_curve.append(capital)

    # Close final position
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


def backtest_all_windows_advanced(results, df_clean, windows, config,
                                  initial_capital=1000.0, transaction_cost=0.0002):
    """Backtest all windows with advanced exits."""
    capital = initial_capital
    all_backtests = []

    for window_idx, result in enumerate(results):
        window = windows[window_idx]
        test_start = window['test_start']
        test_end = window['test_end']
        test_indices = range(test_start, test_end)

        signals, q1, q3 = generate_signals_regression(result['predictions'])

        backtest = backtest_advanced_exits(
            result['actuals'],
            signals,
            df_clean,
            test_indices,
            capital,
            base_stop_loss_pct=config['base_stop_loss_pct'],
            base_take_profit_pct=config['base_take_profit_pct'],
            use_vol_adjustment=config.get('use_vol_adjustment', False),
            vol_window=config.get('vol_window', 20),
            use_trailing_stop=config.get('use_trailing_stop', False),
            trail_activate_pct=config.get('trail_activate_pct', 0.0075),
            trail_distance_pct=config.get('trail_distance_pct', 0.0025),
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

# Test configurations
test_configs = [
    # Baseline
    {
        'name': 'Baseline (0.50% SL / 1.25% TP)',
        'base_stop_loss_pct': 0.0050,
        'base_take_profit_pct': 0.0125,
        'use_vol_adjustment': False,
        'use_trailing_stop': False
    },

    # Volatility-adjusted only
    {
        'name': 'Vol-adjusted (base 0.50% / 1.25%)',
        'base_stop_loss_pct': 0.0050,
        'base_take_profit_pct': 0.0125,
        'use_vol_adjustment': True,
        'vol_window': 20,
        'use_trailing_stop': False
    },
    {
        'name': 'Vol-adjusted (base 0.40% / 1.00%)',
        'base_stop_loss_pct': 0.0040,
        'base_take_profit_pct': 0.0100,
        'use_vol_adjustment': True,
        'vol_window': 20,
        'use_trailing_stop': False
    },
    {
        'name': 'Vol-adjusted (base 0.60% / 1.50%)',
        'base_stop_loss_pct': 0.0060,
        'base_take_profit_pct': 0.0150,
        'use_vol_adjustment': True,
        'vol_window': 20,
        'use_trailing_stop': False
    },

    # Trailing stop only
    {
        'name': 'Trailing (activate 0.75%, trail 0.25%)',
        'base_stop_loss_pct': 0.0050,
        'base_take_profit_pct': 0.0125,
        'use_vol_adjustment': False,
        'use_trailing_stop': True,
        'trail_activate_pct': 0.0075,
        'trail_distance_pct': 0.0025
    },
    {
        'name': 'Trailing (activate 0.50%, trail 0.20%)',
        'base_stop_loss_pct': 0.0050,
        'base_take_profit_pct': 0.0125,
        'use_vol_adjustment': False,
        'use_trailing_stop': True,
        'trail_activate_pct': 0.0050,
        'trail_distance_pct': 0.0020
    },
    {
        'name': 'Trailing (activate 1.00%, trail 0.30%)',
        'base_stop_loss_pct': 0.0050,
        'base_take_profit_pct': 0.0125,
        'use_vol_adjustment': False,
        'use_trailing_stop': True,
        'trail_activate_pct': 0.0100,
        'trail_distance_pct': 0.0030
    },

    # Combined: Vol-adjusted + Trailing
    {
        'name': 'Vol-adj + Trailing (0.75%/0.25%)',
        'base_stop_loss_pct': 0.0050,
        'base_take_profit_pct': 0.0125,
        'use_vol_adjustment': True,
        'vol_window': 20,
        'use_trailing_stop': True,
        'trail_activate_pct': 0.0075,
        'trail_distance_pct': 0.0025
    },
    {
        'name': 'Vol-adj + Trailing (0.50%/0.20%)',
        'base_stop_loss_pct': 0.0050,
        'base_take_profit_pct': 0.0125,
        'use_vol_adjustment': True,
        'vol_window': 20,
        'use_trailing_stop': True,
        'trail_activate_pct': 0.0050,
        'trail_distance_pct': 0.0020
    },
]

print("\n" + "="*80)
print("TESTING ADVANCED EXIT STRATEGIES")
print("="*80)

results_summary = []

for config in test_configs:
    print(f"\n{'='*80}")
    print(f"Testing: {config['name']}")
    print(f"{'='*80}")

    backtests = backtest_all_windows_advanced(
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

results_summary_sorted = sorted(results_summary, key=lambda x: x['sharpe_ratio'], reverse=True)

print(f"\n{'Config':<45} {'Return':>8} {'Annual':>8} {'Sharpe':>8} {'Max DD':>8} {'Trades':>7}")
print("-" * 110)

for r in results_summary_sorted:
    print(f"{r['config']:<45} "
          f"{r['total_return']*100:7.1f}% "
          f"{r['annualized_return']*100:7.2f}% "
          f"{r['sharpe_ratio']:7.3f} "
          f"{r['max_drawdown']*100:7.1f}% "
          f"{r['n_trades']:6d}")

# Save results
output_file = 'advanced_exits_results.json'
with open(output_file, 'w') as f:
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
print(f"  Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f}")
print(f"  Annualized Return: {best_sharpe['annualized_return']*100:.2f}%")
print(f"  Total Return: {best_sharpe['total_return']*100:.2f}%")
print(f"  Max Drawdown: {best_sharpe['max_drawdown']*100:.2f}%")
print(f"{'='*80}")
