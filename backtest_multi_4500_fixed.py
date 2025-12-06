"""
Backtest multi-pair 4500-day ANN checkpoints - FIXED VERSION
Uses same logic as working single-pair script
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
    """Backtest a single pair from checkpoint - uses FRESH price data"""
    print(f"\n{pair}:")
    print("-" * 80)

    # Load checkpoint
    with open(f'4500ANN/checkpoint_predictions_{pair}.pkl', 'rb') as f:
        checkpoint = pickle.load(f)

    predictions = checkpoint['predictions']
    test_indices = checkpoint['test_indices']

    # Load FRESH price data from CSV
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Get test data using indices
    test_df = df.iloc[test_indices].copy()

    print(f"  Predictions: {len(predictions)}")
    print(f"  Date range: {test_df.index[0].date()} to {test_df.index[-1].date()}")

    # Add predictions
    test_df['prediction'] = predictions

    # Load spread data
    try:
        spread_df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
        spread_df['date'] = pd.to_datetime(spread_df['date'])
        spread_df = spread_df.set_index('date')
        test_df = test_df.join(spread_df[['spread_pct']], how='left')
        mean_spread = test_df['spread_pct'].mean()
        print(f"  Mean spread: {mean_spread*100:.4f}%")
    except FileNotFoundError:
        test_df['spread_pct'] = 0.00025
        print(f"  Using default spread: 0.025%")

    # Backtest
    balance = 10000
    trades = []
    prediction_buffer = []

    for i in range(len(test_df)):
        current_pred = test_df.iloc[i]['prediction']

        if pd.isna(current_pred):
            continue

        # Update rolling buffer
        if i >= BUFFER_WARMUP:
            prediction_buffer.append(current_pred)
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer = prediction_buffer[-BUFFER_SIZE:]

        # Calculate thresholds
        if len(prediction_buffer) < BUFFER_WARMUP:
            continue

        buffer_array = np.array(prediction_buffer)
        lower_threshold = np.percentile(buffer_array, LOWER_PCT)
        upper_threshold = np.percentile(buffer_array, UPPER_PCT)

        # Generate signal
        if current_pred <= lower_threshold:
            direction = -1
        elif current_pred >= upper_threshold:
            direction = 1
        else:
            continue

        # Entry on next day's open
        if i + 1 >= len(test_df):
            break

        entry_price = test_df.iloc[i + 1]['open']
        entry_date = test_df.index[i + 1]
        spread_pct = test_df.iloc[i + 1]['spread_pct']
        if pd.isna(spread_pct):
            spread_pct = 0.00025

        # Exit on day after entry (1-day holding)
        if i + 2 >= len(test_df):
            break

        exit_row = test_df.iloc[i + 2]
        exit_date = test_df.index[i + 2]

        # Calculate P&L
        if direction == 1:  # LONG
            actual_entry = entry_price * (1 + spread_pct)

            if exit_row['low'] <= actual_entry * (1 - STOP_LOSS_PCT):
                exit_price = actual_entry * (1 - STOP_LOSS_PCT)
            elif exit_row['high'] >= actual_entry * (1 + TAKE_PROFIT_PCT):
                exit_price = actual_entry * (1 + TAKE_PROFIT_PCT)
            else:
                exit_price = exit_row['close']

            exit_price = exit_price * (1 - spread_pct)
            pnl_pct = (exit_price / actual_entry) - 1

        else:  # SHORT
            actual_entry = entry_price * (1 - spread_pct)

            if exit_row['high'] >= actual_entry * (1 + STOP_LOSS_PCT):
                exit_price = actual_entry * (1 + STOP_LOSS_PCT)
            elif exit_row['low'] <= actual_entry * (1 - TAKE_PROFIT_PCT):
                exit_price = actual_entry * (1 - TAKE_PROFIT_PCT)
            else:
                exit_price = exit_row['close']

            exit_price = exit_price * (1 + spread_pct)
            pnl_pct = (actual_entry / exit_price) - 1

        balance *= (1 + pnl_pct)

        trades.append({
            'date': exit_date,
            'pnl_pct': pnl_pct * 100,
            'balance': balance
        })

    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    total_return = (balance - 10000) / 10000 * 100
    years = len(test_df) / 250
    annual_return = ((balance / 10000) ** (1 / years) - 1) * 100

    returns = trades_df['pnl_pct'].values
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    peak = trades_df['balance'].expanding().max()
    drawdown = (trades_df['balance'] - peak) / peak * 100
    max_dd = drawdown.min()

    wins = len(trades_df[trades_df['pnl_pct'] > 0])
    win_rate = wins / len(trades_df) * 100 if len(trades_df) > 0 else 0

    print(f"  Total Return: {total_return:+.2f}%")
    print(f"  Annual Return: {annual_return:+.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"  Trades: {len(trades_df)}")
    print(f"  Win Rate: {win_rate:.1f}%")

    # Return daily equity curve for portfolio simulation
    # Create daily returns array aligned by date
    return trades_df.set_index('date')['pnl_pct'].reindex(test_df.index, fill_value=0.0).values / 100, {
        'pair': pair,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': len(trades_df),
        'win_rate': win_rate
    }

# Backtest each pair
pair_daily_returns = {}
pair_stats = []

for pair in available_pairs:
    returns, stats = backtest_pair(pair)
    pair_daily_returns[pair] = returns
    pair_stats.append(stats)

# Multi-pair portfolio
print("\n" + "="*80)
print("MULTI-PAIR PORTFOLIO")
print("="*80)

n_pairs = len(available_pairs)
allocation = 1.0 / n_pairs

# Combine returns (weighted average)
portfolio_daily_returns = np.zeros(len(returns))
for pair in available_pairs:
    portfolio_daily_returns += allocation * pair_daily_returns[pair]

# Calculate portfolio metrics
equity = 10000.0
equity_curve = [equity]

for daily_return in portfolio_daily_returns:
    equity *= (1 + daily_return)
    equity_curve.append(equity)

equity_curve = np.array(equity_curve[:-1])  # Remove last element
final_equity = equity_curve[-1]
total_return = (final_equity - 10000) / 10000 * 100
years = len(portfolio_daily_returns) / 250
annual_return = ((final_equity / 10000) ** (1 / years) - 1) * 100

# Sharpe
if portfolio_daily_returns.std() > 0:
    sharpe = (portfolio_daily_returns.mean() / portfolio_daily_returns.std()) * np.sqrt(252)
else:
    sharpe = 0

# Max drawdown
cummax = np.maximum.accumulate(equity_curve)
drawdowns = (equity_curve - cummax) / cummax * 100
max_dd = drawdowns.min()

print(f"\nPortfolio: {n_pairs} pairs @ {allocation*100:.1f}% each")
print(f"Period: {len(portfolio_daily_returns)} trading days ({years:.2f} years)")
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
