"""
Backtest XGBoost predictions from alternative targets (binary, 5-day return).

Usage:
  python backtest_multitarget.py --target target_binary
  python backtest_multitarget.py --target target_5day_return
"""

import pandas as pd
import numpy as np
import pickle
import json
import copy
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True,
                    help='Target name (e.g., target_binary, target_5day_return)')
parser.add_argument('--pair', type=str, default='EURUSD',
                    help='Currency pair (e.g., EURUSD, GBPUSD, USDJPY, AUDUSD)')
args = parser.parse_args()

TARGET = args.target
PAIR = args.pair.upper()
IS_BINARY = 'binary' in TARGET.lower()

print(f"Backtesting XGBoost predictions for: {TARGET}")
print(f"Currency Pair: {PAIR}")
print(f"Prediction type: {'CLASSIFICATION' if IS_BINARY else 'REGRESSION'}")
print("="*70)

# Load data and results
print("\nLoading data and results...")
df = pd.read_csv(f'data/{PAIR}_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

# Drop rows where technical features are NaN (same as training script)
technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]
df_clean = df.dropna(subset=technical_features + [TARGET])

print(f"Loaded data: {df_clean.shape[0]} days ({df_clean.index.min()} to {df_clean.index.max()})")

with open(f'xgboost_results_{PAIR}_{TARGET}.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} windows of predictions")

# Configuration
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS


def generate_windows(df, window_size, roll_days, min_windows=None):
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

        if min_windows is not None and len(windows) >= min_windows:
            break

    return windows


def generate_signals_binary(predictions, strategy='buy_sell'):
    """
    Generate signals from binary classification probabilities.
    predictions: probabilities of up move (0-1)
    """
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    if strategy == 'buy_sell':
        # Use probability thresholds
        # Long if prob > 0.55, short if prob < 0.45
        signals[predictions >= 0.55] = 1
        signals[predictions <= 0.45] = -1
        return signals, 0.45, 0.55

    elif strategy == 'only_buy':
        # Long if prob > 0.5
        signals[predictions > 0.5] = 1
        return signals, None, 0.5

    elif strategy == 'only_sell':
        # Short if prob < 0.5
        signals[predictions < 0.5] = -1
        return signals, 0.5, None


def generate_signals_regression(predictions, strategy='buy_sell'):
    """
    Generate signals from regression predictions (continuous returns).
    Same as before but with predicted returns.
    """
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    if strategy == 'buy_sell':
        q1 = np.percentile(predictions, 25)
        q3 = np.percentile(predictions, 75)
        signals[predictions >= q3] = 1
        signals[predictions <= q1] = -1
        return signals, q1, q3

    elif strategy == 'only_buy':
        signals[predictions > 0] = 1
        return signals, None, 0

    elif strategy == 'only_sell':
        signals[predictions < 0] = -1
        return signals, 0, None


def add_signals_to_results(all_results, strategy, is_binary):
    """Add trading signals to results for all windows."""
    for result in all_results:
        if is_binary:
            signals, threshold_low, threshold_high = generate_signals_binary(
                result['predictions'], strategy
            )
        else:
            signals, threshold_low, threshold_high = generate_signals_regression(
                result['predictions'], strategy
            )

        result['signals'] = signals.tolist()
        result['signal_strategy'] = strategy

        if strategy == 'buy_sell':
            result['threshold_q1'] = threshold_low
            result['threshold_q3'] = threshold_high
        elif strategy == 'only_buy':
            result['threshold'] = threshold_high
        elif strategy == 'only_sell':
            result['threshold'] = threshold_low

        n_buy = np.sum(signals == 1)
        n_sell = np.sum(signals == -1)
        n_hold = np.sum(signals == 0)

        result['signal_distribution'] = {
            'buy': int(n_buy),
            'sell': int(n_sell),
            'hold': int(n_hold)
        }

    return all_results


def backtest_strategy_forex_dynamic(actuals, signals, df_prices, test_indices,
                                    initial_capital, transaction_cost_pct=0.0002):
    """Backtest with dynamic position sizing."""
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
    closes = test_data['close'].values

    for i in range(len(signals)):
        signal = signals[i]
        open_price = opens[i]

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
                'return': trade_return
            })

            position = 0

        if position == 0 and signal != 0:
            cost = capital * transaction_cost_pct
            capital -= cost
            entry_price = open_price
            position = signal
            entry_lot_size = capital

        equity_curve.append(capital)

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
            'return': trade_return
        })

    final_return = (capital - initial_capital) / initial_capital
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]

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
        'n_losing': len(losing_trades)
    }


def backtest_all_windows_forex_dynamic(results_by_strategy, df_clean, windows,
                                       initial_capital=1000.0, transaction_cost=0.0002):
    """Backtest all windows with dynamic position sizing."""
    for strategy, results in results_by_strategy.items():
        capital = initial_capital

        for window_idx, result in enumerate(results):
            window = windows[window_idx]
            test_start = window['test_start']
            test_end = window['test_end']
            test_indices = range(test_start, test_end)

            backtest = backtest_strategy_forex_dynamic(
                result['actuals'],
                result['signals'],
                df_clean,
                test_indices,
                capital,
                transaction_cost
            )
            result['backtest'] = backtest
            capital = backtest['final_capital']

    return results_by_strategy


def calculate_performance_metrics(results, risk_free_rate=0.0):
    """Calculate comprehensive performance metrics."""
    all_returns = []
    for r in results:
        all_returns.extend(r['backtest']['strategy_returns'])

    if len(all_returns) == 0:
        return {
            'total_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'n_trades': 0,
            'n_winning': 0,
            'n_losing': 0
        }

    all_returns = np.array(all_returns)

    equity_curve = np.cumprod(1 + all_returns)
    total_return = equity_curve[-1] - 1

    # Calculate actual number of calendar days (not trades!)
    n_days = len(results) * TEST_DAYS  # 40 windows * 126 days each
    n_years = n_days / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1

    volatility = np.std(all_returns) * np.sqrt(252)
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0

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
        'n_losing': len(losing_trades)
    }


# Generate windows (generate all available windows to match training)
windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS)
print(f"\nGenerated {len(windows)} windows")
print(f"All results has {len(all_results)} windows")

# Generate signals for all strategies
strategies = ['buy_sell', 'only_buy', 'only_sell']
results_by_strategy = {}

for strategy in strategies:
    print(f"\n{'='*70}")
    print(f"Strategy: {strategy.upper()}")
    print(f"{'='*70}")

    strategy_results = copy.deepcopy(all_results)
    strategy_results = add_signals_to_results(strategy_results, strategy, IS_BINARY)
    results_by_strategy[strategy] = strategy_results

    total_buy = sum(r['signal_distribution']['buy'] for r in strategy_results)
    total_sell = sum(r['signal_distribution']['sell'] for r in strategy_results)
    total_hold = sum(r['signal_distribution']['hold'] for r in strategy_results)
    total_signals = total_buy + total_sell + total_hold

    print(f"\nSignal Distribution (across all windows):")
    print(f"  Buy:  {total_buy:4d} ({total_buy/total_signals*100:5.1f}%)")
    print(f"  Sell: {total_sell:4d} ({total_sell/total_signals*100:5.1f}%)")
    print(f"  Hold: {total_hold:4d} ({total_hold/total_signals*100:5.1f}%)")

# Run backtest
print(f"\n{'='*70}")
print("Running forex backtests with DYNAMIC position sizing...")
print(f"{'='*70}")

INITIAL_CAPITAL = 1000.0
TRANSACTION_COST = 0.0002

results_by_strategy = backtest_all_windows_forex_dynamic(
    results_by_strategy,
    df_clean,
    windows,
    INITIAL_CAPITAL,
    TRANSACTION_COST
)

# Display results
for strategy in strategies:
    results = results_by_strategy[strategy]

    print(f"\n{'='*70}")
    print(f"Strategy: {strategy.upper()}")
    print(f"{'='*70}")

    final_capital = results[-1]['backtest']['final_capital']
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    total_trades = sum(r['backtest']['n_trades'] for r in results)

    if total_trades > 0:
        total_long = sum(r['backtest']['n_long'] for r in results)
        total_short = sum(r['backtest']['n_short'] for r in results)

        print(f"\nFinal Results (after {len(results)} windows):")
        print(f"  Initial Capital: ${INITIAL_CAPITAL:.2f}")
        print(f"  Final Capital:   ${final_capital:.2f}")
        print(f"  Total Return:    {total_return*100:.2f}%")
        print(f"  Total Trades:    {total_trades}")
        print(f"    Long:  {total_long} ({total_long/total_trades*100:.1f}%)")
        print(f"    Short: {total_short} ({total_short/total_trades*100:.1f}%)")

        print(f"\nCapital Progression:")
        for i in [0, 9, 19, 29, 39]:
            if i < len(results):
                print(f"  Window {i+1:2d}: ${results[i]['backtest']['final_capital']:.2f}")
    else:
        print(f"\n  NO TRADES - Strategy generated no signals!")

# Calculate performance metrics
print(f"\n{'='*80}")
print("PERFORMANCE SUMMARY")
print(f"{'='*80}")

performance_summary = {}
for strategy in strategies:
    results = results_by_strategy[strategy]
    metrics = calculate_performance_metrics(results)
    performance_summary[strategy] = metrics

    print(f"\n{strategy.upper()}:")
    print("-" * 80)

    if metrics['n_trades'] > 0:
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
        print(f"    Winning:           {metrics['n_winning']:8d}")
        print(f"    Losing:            {metrics['n_losing']:8d}")
    else:
        print("  NO TRADES")

# Save results
summary_file = f'performance_summary_{TARGET}.json'
with open(summary_file, 'w') as f:
    json.dump(performance_summary, f, indent=2)

for strategy, results in results_by_strategy.items():
    results_file = f'backtest_results_{TARGET}_{strategy}.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

print(f"\n{'='*80}")
print("Results saved:")
print(f"  {summary_file}")
for strategy in strategies:
    print(f"  backtest_results_{TARGET}_{strategy}.pkl")
print(f"{'='*80}")
