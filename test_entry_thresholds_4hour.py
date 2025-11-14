"""
Test various entry threshold levels for 4-hour ML strategy.

Instead of using Q1/Q3 (25th/75th percentile), test different threshold levels
to find optimal signal selectivity.

Usage:
    python test_entry_thresholds_4hour.py --pair EURUSD
"""

import pandas as pd
import numpy as np
import pickle
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD',
                    help='Currency pair (e.g., EURUSD, GBPUSD)')
parser.add_argument('--target', type=str, default='target_1day_return',
                    help='Target column used in training')
parser.add_argument('--model-suffix', type=str, default='',
                    help='Model file suffix (e.g., optuna). Leave empty for default model.')
args = parser.parse_args()

CURRENCY_PAIR = args.pair.upper()
TARGET_COLUMN = args.target
MODEL_SUFFIX = args.model_suffix

print("="*80)
print(f"TESTING ENTRY THRESHOLDS - {CURRENCY_PAIR} 4-HOUR")
print("="*80)

# Load 4-hour data
print(f"\nLoading 4-hour {CURRENCY_PAIR} data...")
df = pd.read_csv(f'data/{CURRENCY_PAIR}_4hour_with_features.csv',
                 index_col='date', parse_dates=True)

technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]

df_clean = df.dropna(subset=technical_features + [TARGET_COLUMN])
print(f"Data shape: {df_clean.shape}")
print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")

# Load trained model results
if MODEL_SUFFIX:
    results_file = f'xgboost_results_{CURRENCY_PAIR}_4hour_{TARGET_COLUMN}_{MODEL_SUFFIX}.pkl'
else:
    results_file = f'xgboost_results_{CURRENCY_PAIR}_4hour_{TARGET_COLUMN}.pkl'

print(f"\nLoading results from: {results_file}")
with open(results_file, 'rb') as f:
    all_results = pickle.load(f)

print(f"Loaded predictions for {len(all_results)} bars")

# Walk-forward configuration (BAR-BASED)
TRAIN_BARS = 1800
VAL_BARS = 468
TEST_BARS = 378
ROLL_BARS = 378
WINDOW_SIZE = TRAIN_BARS + VAL_BARS + TEST_BARS


def generate_windows(df, window_size, roll_bars):
    """Generate rolling walk-forward windows (BAR-BASED)."""
    windows = []
    start_idx = 0

    while start_idx + window_size <= len(df):
        train_end = start_idx + TRAIN_BARS
        val_end = train_end + VAL_BARS
        test_end = val_end + TEST_BARS

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
        start_idx += roll_bars

    return windows


def generate_signals_with_threshold(predictions, lower_percentile, upper_percentile):
    """
    Generate signals from regression predictions using custom percentile thresholds.

    Args:
        predictions: Array of ML predictions
        lower_percentile: Percentile for SHORT threshold (e.g., 25 = Q1)
        upper_percentile: Percentile for LONG threshold (e.g., 75 = Q3)
    """
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    lower_threshold = np.percentile(predictions, lower_percentile)
    upper_threshold = np.percentile(predictions, upper_percentile)

    signals[predictions >= upper_threshold] = 1   # LONG
    signals[predictions <= lower_threshold] = -1  # SHORT

    return signals, lower_threshold, upper_threshold


def backtest_advanced_exits_4hour(signals, df_prices, test_indices,
                                   initial_capital,
                                   base_stop_loss_pct=0.0030,
                                   base_take_profit_pct=0.0075,
                                   use_vol_adjustment=True,
                                   vol_window=20,
                                   max_hold_bars=30,
                                   loss_cooldown_bars=6,
                                   transaction_cost_pct=0.0002):
    """Backtest 4-hour strategy with BAR-BASED holds and cooldowns."""
    capital = initial_capital
    position = 0
    entry_price = 0.0
    entry_capital = 0.0
    entry_bar = 0
    cooldown_remaining = 0

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
        median_atr = np.median(atr[~np.isnan(atr)])

    for i in range(len(signals)):
        signal = signals[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]

        # Decrement cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Adjust stops by volatility
        if use_vol_adjustment and not np.isnan(atr[i]):
            vol_ratio = atr[i] / median_atr
            stop_loss_pct = base_stop_loss_pct * vol_ratio
            take_profit_pct = base_take_profit_pct * vol_ratio
        else:
            stop_loss_pct = base_stop_loss_pct
            take_profit_pct = base_take_profit_pct

        # Check exits if we have a position
        if position != 0:
            bars_held = i - entry_bar

            # Calculate P&L
            if position == 1:
                pct_high = (high_price - entry_price) / entry_price
                pct_low = (low_price - entry_price) / entry_price
            else:
                pct_high = (entry_price - low_price) / entry_price
                pct_low = (entry_price - high_price) / entry_price

            exit_triggered = False
            exit_price = None
            exit_reason = None

            # Check stop-loss
            if pct_low <= -stop_loss_pct:
                exit_triggered = True
                exit_reason = 'stop_loss'
                if position == 1:
                    exit_price = entry_price * (1 - stop_loss_pct)
                else:
                    exit_price = entry_price * (1 + stop_loss_pct)

            # Check take-profit
            elif pct_high >= take_profit_pct:
                exit_triggered = True
                exit_reason = 'take_profit'
                if position == 1:
                    exit_price = entry_price * (1 + take_profit_pct)
                else:
                    exit_price = entry_price * (1 - take_profit_pct)

            # Check max hold period (time exit)
            elif bars_held >= max_hold_bars:
                exit_triggered = True
                exit_reason = 'time_exit'
                exit_price = open_price

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
                    'bars_held': bars_held,
                    'pnl_pct': pnl_pct,
                    'pnl': pnl,
                    'return': trade_return,
                    'exit_reason': exit_reason
                })

                # Set cooldown after losing trade
                if pnl < 0 and loss_cooldown_bars > 0:
                    cooldown_remaining = loss_cooldown_bars

                position = 0

        # Enter new position (only if not in cooldown)
        if position == 0 and signal != 0 and cooldown_remaining == 0:
            cost = capital * transaction_cost_pct
            capital -= cost
            entry_price = open_price
            position = signal
            entry_capital = capital
            entry_bar = i

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
            'bars_held': len(signals) - entry_bar,
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
        'n_winning': len(winning_trades),
        'n_losing': len(losing_trades),
        'exit_reasons': exit_reasons
    }


def backtest_threshold_config(results_df, df_clean, windows,
                               lower_percentile, upper_percentile,
                               initial_capital=1000.0, transaction_cost=0.0002):
    """Backtest all windows with given threshold configuration."""
    capital = initial_capital
    all_backtests = []

    # Best exit config from previous tests
    exit_config = {
        'base_stop_loss_pct': 0.0030,
        'base_take_profit_pct': 0.0075,
        'use_vol_adjustment': True,
        'vol_window': 20,
        'max_hold_bars': 30,
        'loss_cooldown_bars': 6
    }

    for window in windows:
        # Get test period data
        test_start_date = df_clean.index[window['test_start']]
        test_end_date = df_clean.index[window['test_end'] - 1]

        # Filter results for this window
        window_results = results_df[
            (results_df.index >= test_start_date) &
            (results_df.index <= test_end_date)
        ]

        if len(window_results) == 0:
            continue

        # Generate signals with custom thresholds
        signals, lower_thresh, upper_thresh = generate_signals_with_threshold(
            window_results['predicted_return'].values,
            lower_percentile,
            upper_percentile
        )

        # Backtest
        test_indices = range(window['test_start'], window['test_end'])
        backtest = backtest_advanced_exits_4hour(
            signals,
            df_clean,
            test_indices,
            capital,
            **exit_config,
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

    # Annualize returns (4-hour bars: ~6 bars per day, ~252 trading days)
    n_bars = len(backtests) * TEST_BARS
    n_days = n_bars / 6
    n_years = n_days / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1

    # Daily volatility
    volatility = np.std(all_returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = np.min(drawdown)

    winning_trades = all_returns[all_returns > 0]
    losing_trades = all_returns[all_returns < 0]

    win_rate = len(winning_trades) / len(all_returns) if len(all_returns) > 0 else 0
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
    avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0

    total_wins = np.sum(winning_trades)
    total_losses = np.abs(np.sum(losing_trades))
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

    # Calculate trades per year
    trades_per_year = len(all_returns) / n_years if n_years > 0 else 0

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
        'n_trades': len(all_returns),
        'n_winning': len(winning_trades),
        'n_losing': len(losing_trades),
        'trades_per_year': trades_per_year
    }


# Generate windows
print("\nGenerating walk-forward windows...")
windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_BARS)
print(f"Generated {len(windows)} windows")

# Test different threshold levels
# Format: (lower_percentile, upper_percentile, description)
threshold_configs = [
    (25, 75, "Q1/Q3 (Current - Top/Bottom 25%)"),
    (30, 70, "Top/Bottom 30%"),
    (35, 65, "Top/Bottom 35%"),
    (40, 60, "Top/Bottom 40%"),
    (42, 58, "Top/Bottom 42%"),
    (44, 56, "Top/Bottom 44%"),
    (46, 54, "Top/Bottom 46%"),
    (48, 52, "Top/Bottom 48%"),
    (49, 51, "Top/Bottom 49% (Least Selective)"),
]

print("\n" + "="*80)
print("TESTING ENTRY THRESHOLD LEVELS")
print("="*80)

results_summary = []

for lower_pct, upper_pct, description in threshold_configs:
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"  SHORT when prediction <= {lower_pct}th percentile")
    print(f"  LONG when prediction >= {upper_pct}th percentile")
    print(f"{'='*80}")

    backtests = backtest_threshold_config(
        all_results,
        df_clean,
        windows,
        lower_pct,
        upper_pct,
        initial_capital=1000.0,
        transaction_cost=0.0002
    )

    metrics = calculate_performance_metrics(backtests)

    if metrics:
        print(f"\nPerformance:")
        print(f"  Total Return:        {metrics['total_return']*100:8.2f}%")
        print(f"  Annualized Return:   {metrics['annualized_return']*100:8.2f}%")
        print(f"  Monthly Return:      {(metrics['annualized_return']/12)*100:8.2f}%")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:8.3f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']*100:8.2f}%")
        print(f"  Win Rate:            {metrics['win_rate']*100:8.2f}%")
        print(f"  Profit Factor:       {metrics['profit_factor']:8.3f}")
        print(f"  Total Trades:        {metrics['n_trades']:8d}")
        print(f"  Trades/Year:         {metrics['trades_per_year']:8.1f}")
        print(f"  Trades/Month:        {metrics['trades_per_year']/12:8.1f}")

        results_summary.append({
            'description': description,
            'lower_pct': lower_pct,
            'upper_pct': upper_pct,
            'selectivity': f"{100 - upper_pct}% + {lower_pct}% = {100 - upper_pct + lower_pct}%",
            **metrics
        })

# Display comparison table
print("\n" + "="*80)
print("COMPARISON TABLE (Sorted by Sharpe Ratio)")
print("="*80)

results_summary_sorted = sorted(results_summary, key=lambda x: x['sharpe_ratio'], reverse=True)

print(f"\n{'Threshold Config':<35} {'Select%':>8} {'Annual':>8} {'Sharpe':>8} {'Trades/Yr':>10} {'Win%':>6}")
print("-" * 100)

for r in results_summary_sorted:
    print(f"{r['description']:<35} "
          f"{r['selectivity']:>8} "
          f"{r['annualized_return']*100:7.2f}% "
          f"{r['sharpe_ratio']:7.3f} "
          f"{r['trades_per_year']:9.1f} "
          f"{r['win_rate']*100:5.1f}%")

print("\n" + "="*80)
print("SORTED BY ANNUAL RETURN")
print("="*80)

results_summary_sorted_return = sorted(results_summary, key=lambda x: x['annualized_return'], reverse=True)

print(f"\n{'Threshold Config':<35} {'Select%':>8} {'Annual':>8} {'Sharpe':>8} {'Trades/Yr':>10} {'Win%':>6}")
print("-" * 100)

for r in results_summary_sorted_return:
    print(f"{r['description']:<35} "
          f"{r['selectivity']:>8} "
          f"{r['annualized_return']*100:7.2f}% "
          f"{r['sharpe_ratio']:7.3f} "
          f"{r['trades_per_year']:9.1f} "
          f"{r['win_rate']*100:5.1f}%")

# Identify best configurations
if len(results_summary_sorted) > 0:
    best_sharpe = results_summary_sorted[0]
    best_return = results_summary_sorted_return[0]

    print(f"\n{'='*80}")
    print("BEST CONFIGURATION (by Sharpe Ratio):")
    print(f"  {best_sharpe['description']}")
    print(f"  Sharpe Ratio:        {best_sharpe['sharpe_ratio']:.3f}")
    print(f"  Annualized Return:   {best_sharpe['annualized_return']*100:.2f}%")
    print(f"  Monthly Return:      {(best_sharpe['annualized_return']/12)*100:.2f}%")
    print(f"  Trades/Year:         {best_sharpe['trades_per_year']:.1f}")
    print(f"  Trades/Month:        {best_sharpe['trades_per_year']/12:.1f}")

    print(f"\n{'='*80}")
    print("BEST CONFIGURATION (by Annualized Return):")
    print(f"  {best_return['description']}")
    print(f"  Annualized Return:   {best_return['annualized_return']*100:.2f}%")
    print(f"  Monthly Return:      {(best_return['annualized_return']/12)*100:.2f}%")
    print(f"  Sharpe Ratio:        {best_return['sharpe_ratio']:.3f}")
    print(f"  Trades/Year:         {best_return['trades_per_year']:.1f}")
    print(f"  Trades/Month:        {best_return['trades_per_year']/12:.1f}")
    print(f"{'='*80}")
