"""
Optimize hold time and threshold parameters for ANN strategy
Uses saved predictions to quickly test different configurations
"""
import pandas as pd
import numpy as np
import pickle
import os
from itertools import product

print("="*100)
print("OPTIMIZING HOLD TIME & THRESHOLDS - ANN")
print("="*100)
print()

PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
PREDICTION_DIR = 'optimized_ann_predictions'

# Load all predictions
print("Loading predictions...")
pair_data = {}

for pair in PAIRS:
    pred_file = os.path.join(PREDICTION_DIR, f'predictions_{pair}.pkl')

    if not os.path.exists(pred_file):
        print(f"ERROR: {pred_file} not found!")
        exit(1)

    with open(pred_file, 'rb') as f:
        data = pickle.load(f)

    pair_data[pair] = data
    print(f"  {pair}: {len(data['predictions'])} predictions loaded")

print()

def backtest_pair(pair, stop_loss, take_profit, hold_length, lower_pct, upper_pct):
    """Backtest with specific parameters"""

    # Get predictions and indices
    predictions = pair_data[pair]['predictions']
    test_indices = pair_data[pair]['test_indices']

    # Load price data
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    test_df = df.iloc[test_indices].copy()
    test_df['prediction'] = predictions

    # Load spread data
    try:
        spread_df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
        spread_df['date'] = pd.to_datetime(spread_df['date'])
        spread_df = spread_df.set_index('date')
        test_df = test_df.join(spread_df[['spread_pct']], how='left')
    except FileNotFoundError:
        test_df['spread_pct'] = 0.00025

    # Backtest parameters
    BUFFER_SIZE = 200
    BUFFER_WARMUP = 50

    trades = []
    prediction_buffer = []
    daily_returns = []

    for i in range(len(test_df)):
        current_pred = test_df.iloc[i]['prediction']

        if pd.isna(current_pred):
            daily_returns.append(0)
            continue

        # Update buffer
        if i >= BUFFER_WARMUP:
            prediction_buffer.append(current_pred)
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer = prediction_buffer[-BUFFER_SIZE:]

        if len(prediction_buffer) < BUFFER_WARMUP:
            daily_returns.append(0)
            continue

        # Calculate thresholds
        buffer_array = np.array(prediction_buffer)
        lower_threshold = np.percentile(buffer_array, lower_pct)
        upper_threshold = np.percentile(buffer_array, upper_pct)

        # Generate signal
        if current_pred <= lower_threshold:
            direction = -1
        elif current_pred >= upper_threshold:
            direction = 1
        else:
            daily_returns.append(0)
            continue

        # Entry on next day
        if i + 1 >= len(test_df):
            break

        entry_price = test_df.iloc[i + 1]['open']
        spread_pct = test_df.iloc[i + 1]['spread_pct']
        if pd.isna(spread_pct):
            spread_pct = 0.00025

        # Check exit over hold period
        exit_day = min(i + 1 + hold_length, len(test_df) - 1)

        for hold_day in range(i + 2, exit_day + 1):
            if hold_day >= len(test_df):
                break

            exit_row = test_df.iloc[hold_day]

            if direction == 1:  # LONG
                actual_entry = entry_price * (1 + spread_pct)

                # Stop loss
                if exit_row['low'] <= actual_entry * (1 - stop_loss):
                    exit_price = actual_entry * (1 - stop_loss) * (1 - spread_pct)
                    pnl_pct = (exit_price / actual_entry) - 1
                    trades.append(pnl_pct)
                    daily_returns.append(pnl_pct)
                    break

                # Take profit
                if exit_row['high'] >= actual_entry * (1 + take_profit):
                    exit_price = actual_entry * (1 + take_profit) * (1 - spread_pct)
                    pnl_pct = (exit_price / actual_entry) - 1
                    trades.append(pnl_pct)
                    daily_returns.append(pnl_pct)
                    break

                # End of hold
                if hold_day == exit_day:
                    exit_price = exit_row['close'] * (1 - spread_pct)
                    pnl_pct = (exit_price / actual_entry) - 1
                    trades.append(pnl_pct)
                    daily_returns.append(pnl_pct)
                    break

            else:  # SHORT
                actual_entry = entry_price * (1 - spread_pct)

                # Stop loss
                if exit_row['high'] >= actual_entry * (1 + stop_loss):
                    exit_price = actual_entry * (1 + stop_loss) * (1 + spread_pct)
                    pnl_pct = (actual_entry / exit_price) - 1
                    trades.append(pnl_pct)
                    daily_returns.append(pnl_pct)
                    break

                # Take profit
                if exit_row['low'] <= actual_entry * (1 - take_profit):
                    exit_price = actual_entry * (1 - take_profit) * (1 + spread_pct)
                    pnl_pct = (actual_entry / exit_price) - 1
                    trades.append(pnl_pct)
                    daily_returns.append(pnl_pct)
                    break

                # End of hold
                if hold_day == exit_day:
                    exit_price = exit_row['close'] * (1 + spread_pct)
                    pnl_pct = (actual_entry / exit_price) - 1
                    trades.append(pnl_pct)
                    daily_returns.append(pnl_pct)
                    break
        else:
            # No trade taken
            daily_returns.append(0)

    return np.array(daily_returns)

# Fixed parameters
STOP_LOSS = 0.0018  # 0.18%
TAKE_PROFIT = 0.05  # 5%

# Parameters to optimize
HOLD_TIMES = [1, 2, 3, 5, 7]
THRESHOLD_CONFIGS = [
    {'lower': 40, 'upper': 60, 'name': '40/60 (looser)'},
    {'lower': 45, 'upper': 55, 'name': '45/55'},
    {'lower': 48, 'upper': 52, 'name': '48/52 (current)'},
    {'lower': 49, 'upper': 51, 'name': '49/51'},
    {'lower': 50, 'upper': 50, 'name': '50/50 (median split)'},
]

print("="*100)
print("TESTING CONFIGURATIONS")
print("="*100)
print()
print(f"Fixed parameters:")
print(f"  Stop Loss: {STOP_LOSS*100:.2f}%")
print(f"  Take Profit: {TAKE_PROFIT*100:.0f}%")
print()
print(f"Testing {len(HOLD_TIMES)} hold times × {len(THRESHOLD_CONFIGS)} threshold configs = {len(HOLD_TIMES) * len(THRESHOLD_CONFIGS)} combinations")
print()

all_results = []

# Test all combinations
for hold_time in HOLD_TIMES:
    for threshold_config in THRESHOLD_CONFIGS:
        lower_pct = threshold_config['lower']
        upper_pct = threshold_config['upper']

        # Backtest each pair
        pair_returns = {}
        for pair in PAIRS:
            returns = backtest_pair(pair, STOP_LOSS, TAKE_PROFIT, hold_time, lower_pct, upper_pct)
            pair_returns[pair] = returns

        # Calculate multi-pair portfolio returns
        n_pairs = len(PAIRS)
        allocation = 1.0 / n_pairs

        # Combine daily returns (equal weight)
        portfolio_returns = np.zeros(len(returns))
        for pair, returns in pair_returns.items():
            portfolio_returns += allocation * returns

        # Calculate metrics at 2x leverage
        leverage = 2.0
        leveraged_returns = portfolio_returns * leverage

        # Calculate equity curve
        balance = 1000
        equity_curve = [balance]
        blew_up = False

        for ret in leveraged_returns:
            balance *= (1 + ret)

            if balance <= 100:  # 90% drawdown = blowup
                blew_up = True
                break

            equity_curve.append(balance)

        if blew_up:
            continue

        # Calculate metrics
        equity_curve = np.array(equity_curve)

        years = len(leveraged_returns) / 250
        annual_return = ((balance / 1000) ** (1 / years) - 1) * 100

        sharpe = (leveraged_returns.mean() / leveraged_returns.std()) * np.sqrt(252) if leveraged_returns.std() > 0 else 0

        # Max drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - cummax) / cummax * 100
        max_dd = drawdowns.min()

        # Win rate
        non_zero = leveraged_returns[leveraged_returns != 0]
        win_rate = (non_zero > 0).sum() / len(non_zero) * 100 if len(non_zero) > 0 else 0

        # Trade frequency
        trades_per_year = (non_zero != 0).sum() / years

        all_results.append({
            'hold_time': hold_time,
            'thresholds': threshold_config['name'],
            'lower_pct': lower_pct,
            'upper_pct': upper_pct,
            'annual': annual_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'trades_per_year': trades_per_year,
            'balance': balance
        })

print("="*100)
print("RESULTS BY HOLD TIME")
print("="*100)
print()

for hold_time in HOLD_TIMES:
    print(f"\nHold Time: {hold_time} day(s)")
    print(f"  {'Thresholds':<20} {'Annual':<10} {'Sharpe':<8} {'Max DD':<10} {'WinRate':<8} {'Trades/Yr':<10}")
    print("  " + "-"*80)

    hold_results = [r for r in all_results if r['hold_time'] == hold_time]
    for r in sorted(hold_results, key=lambda x: x['annual'], reverse=True):
        print(f"  {r['thresholds']:<20} {r['annual']:>8.2f}% {r['sharpe']:>7.2f} {r['max_dd']:>8.2f}% {r['win_rate']:>7.1f}% {r['trades_per_year']:>9.1f}")

print("\n\n" + "="*100)
print("RESULTS BY THRESHOLD")
print("="*100)
print()

for threshold_config in THRESHOLD_CONFIGS:
    print(f"\nThresholds: {threshold_config['name']}")
    print(f"  {'Hold':<8} {'Annual':<10} {'Sharpe':<8} {'Max DD':<10} {'WinRate':<8} {'Trades/Yr':<10}")
    print("  " + "-"*70)

    thresh_results = [r for r in all_results if r['thresholds'] == threshold_config['name']]
    for r in sorted(thresh_results, key=lambda x: x['annual'], reverse=True):
        print(f"  {r['hold_time']:>6}d {r['annual']:>8.2f}% {r['sharpe']:>7.2f} {r['max_dd']:>8.2f}% {r['win_rate']:>7.1f}% {r['trades_per_year']:>9.1f}")

print("\n\n" + "="*100)
print("TOP 10 CONFIGURATIONS (BY ANNUAL RETURN)")
print("="*100)
print()

print(f"{'Rank':<6} {'Hold':<8} {'Thresholds':<20} {'Annual':<10} {'Sharpe':<8} {'Max DD':<10} {'WinRate':<8} {'Trades/Yr':<10}")
print("-"*100)

top_results = sorted(all_results, key=lambda x: x['annual'], reverse=True)[:10]
for i, r in enumerate(top_results, 1):
    print(f"{i:<6} {r['hold_time']:>6}d {r['thresholds']:<20} {r['annual']:>8.2f}% {r['sharpe']:>7.2f} {r['max_dd']:>8.2f}% {r['win_rate']:>7.1f}% {r['trades_per_year']:>9.1f}")

print("\n\n" + "="*100)
print("TOP 10 CONFIGURATIONS (BY SHARPE RATIO)")
print("="*100)
print()

print(f"{'Rank':<6} {'Hold':<8} {'Thresholds':<20} {'Annual':<10} {'Sharpe':<8} {'Max DD':<10} {'WinRate':<8} {'Trades/Yr':<10}")
print("-"*100)

top_sharpe = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)[:10]
for i, r in enumerate(top_sharpe, 1):
    print(f"{i:<6} {r['hold_time']:>6}d {r['thresholds']:<20} {r['annual']:>8.2f}% {r['sharpe']:>7.2f} {r['max_dd']:>8.2f}% {r['win_rate']:>7.1f}% {r['trades_per_year']:>9.1f}")

print("\n\n" + "="*100)
print("TOP 10 CONFIGURATIONS (BY LOWEST DRAWDOWN)")
print("="*100)
print()

print(f"{'Rank':<6} {'Hold':<8} {'Thresholds':<20} {'Annual':<10} {'Sharpe':<8} {'Max DD':<10} {'WinRate':<8} {'Trades/Yr':<10}")
print("-"*100)

top_dd = sorted(all_results, key=lambda x: x['max_dd'], reverse=True)[:10]
for i, r in enumerate(top_dd, 1):
    print(f"{i:<6} {r['hold_time']:>6}d {r['thresholds']:<20} {r['annual']:>8.2f}% {r['sharpe']:>7.2f} {r['max_dd']:>8.2f}% {r['win_rate']:>7.1f}% {r['trades_per_year']:>9.1f}")

# Find best overall (balance of metrics)
print("\n\n" + "="*100)
print("RECOMMENDED CONFIGURATION")
print("="*100)
print()

# Score based on annual return, sharpe, and low drawdown
for r in all_results:
    # Normalize metrics to 0-1 scale
    max_annual = max(x['annual'] for x in all_results)
    max_sharpe = max(x['sharpe'] for x in all_results)
    min_dd = min(x['max_dd'] for x in all_results)

    annual_score = r['annual'] / max_annual
    sharpe_score = r['sharpe'] / max_sharpe
    dd_score = (r['max_dd'] - min_dd) / (0 - min_dd) if min_dd != 0 else 1

    r['composite_score'] = (annual_score * 0.4) + (sharpe_score * 0.4) + (dd_score * 0.2)

best = max(all_results, key=lambda x: x['composite_score'])

print(f"Best Overall Configuration:")
print(f"  Hold Time: {best['hold_time']} day(s)")
print(f"  Thresholds: {best['thresholds']} ({best['lower_pct']}/{best['upper_pct']} percentile)")
print(f"  Annual Return @ 2x: {best['annual']:.2f}%")
print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
print(f"  Max Drawdown: {best['max_dd']:.2f}%")
print(f"  Win Rate: {best['win_rate']:.1f}%")
print(f"  Trades per Year: {best['trades_per_year']:.1f}")
print()

# Compare to current config
current = [r for r in all_results if r['hold_time'] == 1 and r['lower_pct'] == 48][0]
print(f"Current Configuration (1 day, 48/52):")
print(f"  Annual Return @ 2x: {current['annual']:.2f}%")
print(f"  Sharpe Ratio: {current['sharpe']:.2f}")
print(f"  Max Drawdown: {current['max_dd']:.2f}%")
print()

if best != current:
    annual_improvement = best['annual'] - current['annual']
    sharpe_improvement = best['sharpe'] - current['sharpe']
    dd_improvement = best['max_dd'] - current['max_dd']

    print(f"Improvement:")
    print(f"  Annual: {annual_improvement:+.2f}% ({annual_improvement/current['annual']*100:+.1f}%)")
    print(f"  Sharpe: {sharpe_improvement:+.2f} ({sharpe_improvement/current['sharpe']*100:+.1f}%)")
    print(f"  Max DD: {dd_improvement:+.2f}% ({dd_improvement/abs(current['max_dd'])*100:+.1f}%)")
else:
    print("Current configuration is already optimal!")

print("\n" + "="*100)
print("OPTIMIZATION COMPLETE")
print("="*100)
