"""
Backtest multi-pair 4500-day ANN checkpoints
"""
import pickle
import pandas as pd
import numpy as np
from glob import glob
import os

# Find all available checkpoints in 4500ANN folder
checkpoint_files = glob('4500ANN/checkpoint_predictions_*.pkl')
available_pairs = [os.path.basename(f).replace('checkpoint_predictions_', '').replace('.pkl', '')
                   for f in checkpoint_files]

print("="*80)
print("MULTI-PAIR 4500-DAY BACKTEST FROM CHECKPOINTS")
print("="*80)
print(f"Available pairs: {', '.join(available_pairs)}")
print()

# Backtest parameters
STOP_LOSS_PCT = 0.0018
TAKE_PROFIT_PCT = 0.0300
LOWER_PCT = 48
UPPER_PCT = 52
BUFFER_SIZE = 200
BUFFER_WARMUP = 50

def backtest_pair(pair):
    """Backtest a single pair from checkpoint"""
    print(f"\n{pair}:")
    print("-" * 80)

    # Load checkpoint
    with open(f'4500ANN/checkpoint_predictions_{pair}.pkl', 'rb') as f:
        checkpoint = pickle.load(f)

    predictions = checkpoint['predictions']
    test_indices = checkpoint['test_indices']
    df = checkpoint['df']

    print(f"  Predictions: {len(predictions)}")
    print(f"  Date range: {df.index[test_indices[0]].date()} to {df.index[test_indices[-1]].date()}")

    # Get test data
    test_df = df.iloc[test_indices].copy()

    # Load spread data if not already present
    if 'spread_pct' not in test_df.columns:
        try:
            spread_df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
            spread_df['date'] = pd.to_datetime(spread_df['date'])
            spread_df = spread_df.set_index('date')
            test_df = test_df.join(spread_df[['spread_pct']], how='left')
            print(f"  Mean spread: {test_df['spread_pct'].mean()*100:.4f}%")
        except FileNotFoundError:
            test_df['spread_pct'] = 0.00025
            print(f"  Using default spread: 0.025%")
    else:
        print(f"  Using spread from checkpoint: {test_df['spread_pct'].mean()*100:.4f}%")

    # Initialize
    daily_returns = []
    prediction_buffer = []
    position = 0
    entry_price = 0.0
    holding_days = 0
    trades = []

    for i in range(len(predictions)):
        current_pred = predictions[i]

        if pd.isna(current_pred):
            daily_returns.append(0.0)
            continue

        # Update rolling buffer
        if i >= BUFFER_WARMUP:
            prediction_buffer.append(current_pred)
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer = prediction_buffer[-BUFFER_SIZE:]

        # Calculate thresholds from rolling buffer
        if len(prediction_buffer) < BUFFER_WARMUP:
            daily_returns.append(0.0)
            continue

        buffer_array = np.array(prediction_buffer)
        lower_threshold = np.percentile(buffer_array, LOWER_PCT)
        upper_threshold = np.percentile(buffer_array, UPPER_PCT)

        # Generate signal
        if current_pred <= lower_threshold:
            signal = -1  # SHORT
        elif current_pred >= upper_threshold:
            signal = 1   # LONG
        else:
            signal = 0

        day_return = 0.0

        # Check for exit
        if position != 0:
            holding_days += 1

            row = test_df.iloc[i]
            spread_pct = row['spread_pct'] if not pd.isna(row['spread_pct']) else 0.00025

            if position == 1:  # LONG
                pct_high = (row['high'] - entry_price) / entry_price
                pct_low = (row['low'] - entry_price) / entry_price
            else:  # SHORT
                pct_high = (entry_price - row['low']) / entry_price
                pct_low = (entry_price - row['high']) / entry_price

            exit_triggered = False
            exit_price = None

            if pct_low <= -STOP_LOSS_PCT:
                exit_triggered = True
                if position == 1:
                    exit_price = entry_price * (1 - STOP_LOSS_PCT)
                else:
                    exit_price = entry_price * (1 + STOP_LOSS_PCT)
            elif pct_high >= TAKE_PROFIT_PCT:
                exit_triggered = True
                if position == 1:
                    exit_price = entry_price * (1 + TAKE_PROFIT_PCT)
                else:
                    exit_price = entry_price * (1 - TAKE_PROFIT_PCT)
            elif holding_days >= 1:  # 1-day holding period
                exit_triggered = True
                exit_price = row['close']

            if exit_triggered:
                if position == 1:
                    raw_return_pct = (exit_price - entry_price) / entry_price
                else:
                    raw_return_pct = (entry_price - exit_price) / entry_price

                net_return_pct = raw_return_pct - spread_pct
                day_return = net_return_pct

                trades.append(net_return_pct)
                position = 0
                holding_days = 0

        daily_returns.append(day_return)

        # Check for entry
        if position == 0 and signal != 0 and i + 1 < len(test_df):
            position = signal
            entry_price = test_df.iloc[i + 1]['open']
            holding_days = 0

    # Calculate metrics
    daily_returns = np.array(daily_returns)
    trades = np.array(trades)

    total_return = (np.prod(1 + daily_returns) - 1) * 100
    years = len(daily_returns) / 250
    annual_return = ((np.prod(1 + daily_returns) ** (1 / years)) - 1) * 100 if years > 0 else 0

    # Sharpe
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    cum_returns = np.cumprod(1 + daily_returns)
    cummax = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - cummax) / cummax * 100
    max_dd = drawdowns.min()

    # Win rate
    wins = len(trades[trades > 0])
    win_rate = wins / len(trades) * 100 if len(trades) > 0 else 0

    print(f"  Total Return: {total_return:+.2f}%")
    print(f"  Annual Return: {annual_return:+.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"  Trades: {len(trades)}")
    print(f"  Win Rate: {win_rate:.1f}%")

    return daily_returns, {
        'pair': pair,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': len(trades),
        'win_rate': win_rate
    }

# Backtest each pair
pair_returns = {}
pair_stats = []

for pair in available_pairs:
    returns, stats = backtest_pair(pair)
    pair_returns[pair] = returns
    pair_stats.append(stats)

# Multi-pair portfolio backtest
print("\n" + "="*80)
print("MULTI-PAIR PORTFOLIO")
print("="*80)

n_pairs = len(available_pairs)
allocation_per_pair = 1.0 / n_pairs

# Start with $10,000
equity = 10000.0
equity_curve = [equity]

# Get length (all pairs should have same length)
n_days = len(pair_returns[available_pairs[0]])

for day in range(n_days):
    daily_portfolio_return = 0.0

    for pair in available_pairs:
        pair_return = pair_returns[pair][day]
        daily_portfolio_return += allocation_per_pair * pair_return

    equity *= (1 + daily_portfolio_return)
    equity_curve.append(equity)

equity_curve = np.array(equity_curve)
portfolio_returns = np.diff(equity_curve) / equity_curve[:-1]

# Portfolio metrics
final_equity = equity_curve[-1]
total_return = (final_equity - 10000) / 10000 * 100
years = n_days / 250
annual_return = ((final_equity / 10000) ** (1 / years) - 1) * 100

# Sharpe
if portfolio_returns.std() > 0:
    sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
else:
    sharpe = 0

# Max drawdown
cummax = np.maximum.accumulate(equity_curve)
drawdowns = (equity_curve - cummax) / cummax * 100
max_dd = drawdowns.min()

print(f"\nPortfolio: {n_pairs} pairs @ {allocation_per_pair*100:.1f}% each")
print(f"Period: {n_days} trading days ({years:.2f} years)")
print()
print("PORTFOLIO PERFORMANCE (1x leverage):")
print(f"  Total Return: {total_return:+.2f}%")
print(f"  Annual Return: {annual_return:+.2f}%")
print(f"  Sharpe Ratio: {sharpe:.2f}")
print(f"  Max Drawdown: {max_dd:.2f}%")
print()
print("AT 2x LEVERAGE:")
print(f"  Annual Return: {annual_return * 2:+.2f}%")
print(f"  Max Drawdown: {max_dd * 2:.2f}%")
print()
print("AT 3.5x LEVERAGE:")
print(f"  Annual Return: {annual_return * 3.5:+.2f}%")
print(f"  Max Drawdown: {max_dd * 3.5:.2f}%")

print("\n" + "="*80)
print("COMPARISON TO XGBOOST")
print("="*80)
print("XGBoost 4-pair portfolio (4500 days):")
print("  Annual Return: ~24.95% @ 1x leverage")
print("  Annual Return: ~55.74% @ 2x leverage")
print()
print(f"ANN {n_pairs}-pair portfolio (4500 days):")
print(f"  Annual Return: {annual_return:+.2f}% @ 1x leverage")
print(f"  Annual Return: {annual_return * 2:+.2f}% @ 2x leverage")
print()
if annual_return > 24.95:
    diff = annual_return - 24.95
    pct_better = (diff / 24.95) * 100
    print(f"ANN is {diff:+.2f}% better ({pct_better:+.1f}% improvement)")
else:
    diff = 24.95 - annual_return
    pct_worse = (diff / 24.95) * 100
    print(f"ANN is {diff:.2f}% lower ({pct_worse:.1f}% below XGBoost)")

print("\n" + "="*80)
