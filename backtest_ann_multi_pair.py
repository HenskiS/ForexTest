"""
Multi-pair backtest for optimized ANN predictions
Tests different SL/TP combinations and leverage levels
Compares to XGBoost baseline
"""
import pandas as pd
import numpy as np
import pickle
import os
from itertools import product

print("="*100)
print("ANN MULTI-PAIR BACKTEST - OPTIMIZED HYPERPARAMETERS")
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

def backtest_pair(pair, stop_loss, take_profit, hold_length=1):
    """Backtest a single pair with given parameters"""

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
    LOWER_PCT = 48
    UPPER_PCT = 52
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
        lower_threshold = np.percentile(buffer_array, LOWER_PCT)
        upper_threshold = np.percentile(buffer_array, UPPER_PCT)

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

# Test configurations
CONFIGS = [
    {'sl': 0.001, 'tp': 0.05, 'name': '0.10% SL / 5% TP'},
    {'sl': 0.0018, 'tp': 0.05, 'name': '0.18% SL / 5% TP'},
    {'sl': 0.002, 'tp': 0.05, 'name': '0.20% SL / 5% TP'},
]

LEVERAGE_LEVELS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

print("="*100)
print("RUNNING BACKTESTS")
print("="*100)
print()

all_results = []

for config in CONFIGS:
    print(f"\n{'='*100}")
    print(f"TESTING: {config['name']}")
    print(f"{'='*100}\n")

    # Backtest each pair
    pair_returns = {}
    for pair in PAIRS:
        print(f"  Backtesting {pair}...")
        returns = backtest_pair(pair, config['sl'], config['tp'])
        pair_returns[pair] = returns

    print()

    # Calculate multi-pair portfolio returns
    n_pairs = len(PAIRS)
    allocation = 1.0 / n_pairs

    # Combine daily returns (equal weight)
    portfolio_returns = np.zeros(len(returns))
    for pair, returns in pair_returns.items():
        portfolio_returns += allocation * returns

    # Test different leverage levels
    for leverage in LEVERAGE_LEVELS:
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
            print(f"  {leverage}x: BLEW UP")
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

        all_results.append({
            'config': config['name'],
            'leverage': leverage,
            'annual': annual_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'balance': balance
        })

        print(f"  {leverage}x: {annual_return:>6.2f}% annual | Sharpe {sharpe:.2f} | DD {max_dd:>6.2f}% | WR {win_rate:.1f}%")

# Summary
print(f"\n\n{'='*100}")
print("RESULTS SUMMARY")
print(f"{'='*100}\n")

print(f"{'Config':<25} {'Leverage':<10} {'Annual':<12} {'Sharpe':<10} {'Max DD':<12} {'Win Rate':<10}")
print("-" * 100)

for r in all_results:
    print(f"{r['config']:<25} {r['leverage']:<10.1f}x {r['annual']:>10.2f}% {r['sharpe']:>9.2f} {r['max_dd']:>10.2f}% {r['win_rate']:>9.1f}%")

# Compare to XGBoost
print(f"\n\n{'='*100}")
print("COMPARISON TO XGBOOST")
print(f"{'='*100}\n")

xgboost_results = {
    '1x': {'annual': 24.95, 'sharpe': 4.73, 'dd': -5.65},
    '2x': {'annual': 55.74, 'sharpe': 4.73, 'dd': -11.00},
    '3.5x': {'annual': 102.40, 'sharpe': 4.73, 'dd': -18.80}
}

print("XGBoost (0.18% SL, 3% TP):")
for lev, stats in xgboost_results.items():
    print(f"  {lev:>4}: {stats['annual']:>6.2f}% annual | Sharpe {stats['sharpe']:.2f} | DD {stats['dd']:>6.2f}%")

print()

# Find best ANN configs for each leverage
for lev in LEVERAGE_LEVELS:
    lev_results = [r for r in all_results if r['leverage'] == lev]
    if lev_results:
        best = max(lev_results, key=lambda x: x['annual'])
        print(f"Best ANN @ {lev}x: {best['config']}")
        print(f"  {best['annual']:>6.2f}% annual | Sharpe {best['sharpe']:.2f} | DD {best['max_dd']:>6.2f}%")

        # Compare to XGBoost if available
        xgb_key = f"{lev}x"
        if xgb_key in xgboost_results:
            xgb = xgboost_results[xgb_key]
            diff = best['annual'] - xgb['annual']
            pct_diff = (diff / xgb['annual']) * 100
            print(f"  vs XGBoost: {diff:+.2f}% ({pct_diff:+.1f}%)")
        print()

# Recommendations
print(f"\n{'='*100}")
print("RECOMMENDATIONS")
print(f"{'='*100}\n")

# Find safest high-performer
safe_results = [r for r in all_results if r['max_dd'] > -15]
if safe_results:
    best_safe = max(safe_results, key=lambda x: x['annual'])
    print(f"Safest High Performer:")
    print(f"  {best_safe['config']} @ {best_safe['leverage']}x")
    print(f"  {best_safe['annual']:.2f}% annual | Sharpe {best_safe['sharpe']:.2f} | DD {best_safe['max_dd']:.2f}%")
    print()

print(f"\n{'='*100}")
print("BACKTEST COMPLETE")
print(f"{'='*100}")
