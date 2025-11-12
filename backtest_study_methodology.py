"""
Backtest using the STUDY'S METHODOLOGY:
- Entry: Q3 (top 25%) = BUY, Q1 (bottom 25%) = SELL
- Exit: Hold until next signal (no stops!)
- Position: Fixed 1 micro lot (1000 units)
- Transaction cost: 0.02%

This tests if our model achieves directional accuracy like the study did.
"""

import pandas as pd
import numpy as np
import pickle
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--predictions', type=str, required=True,
                    help='Path to predictions pickle file')
parser.add_argument('--pair', type=str, default='EURUSD',
                    help='Currency pair')
args = parser.parse_args()

CURRENCY_PAIR = args.pair.upper()
TRANSACTION_COST = 0.0002  # 0.02%
STARTING_CAPITAL = 1000
LOT_SIZE = 1000  # 1 micro lot = 1000 units
PIP_VALUE = 0.1  # $0.10 per pip for 1 micro lot

print("=" * 80)
print(f"BACKTEST - STUDY'S METHODOLOGY")
print("=" * 80)
print(f"Pair: {CURRENCY_PAIR}")
print(f"Entry: Q3 (top 25%%) = BUY, Q1 (bottom 25%%) = SELL")
print(f"Exit: Hold until next signal (NO STOPS)")
print(f"Position: Fixed {LOT_SIZE} units (1 micro lot)")
print(f"Pip value: ${PIP_VALUE} per pip")
print(f"Transaction cost: {TRANSACTION_COST*100}%")
print(f"Starting capital: ${STARTING_CAPITAL}")
print("=" * 80)

# Load predictions
print(f"\nLoading predictions from {args.predictions}...")
with open(args.predictions, 'rb') as f:
    results_df = pickle.load(f)

print(f"Predictions loaded: {len(results_df)} rows")
print(f"Date range: {results_df.index.min()} to {results_df.index.max()}")
print(f"Columns: {list(results_df.columns)}")

# Determine prediction column name
if 'predicted_return' in results_df.columns:
    pred_col = 'predicted_return'
elif 'predicted_class' in results_df.columns:
    pred_col = 'predicted_class'
else:
    print(f"ERROR: No prediction column found!")
    print(f"Available columns: {list(results_df.columns)}")
    exit(1)

print(f"\nUsing prediction column: {pred_col}")

# Calculate quartiles using ROLLING WINDOW (avoid lookahead bias)
# Use 252-bar window (42 days on 4H = ~6 weeks of data)
QUARTILE_WINDOW = 252

print(f"\nCalculating Q1/Q3 quartiles with {QUARTILE_WINDOW}-bar rolling window...")
results_df['q1'] = results_df[pred_col].rolling(window=QUARTILE_WINDOW, min_periods=QUARTILE_WINDOW).quantile(0.25)
results_df['q3'] = results_df[pred_col].rolling(window=QUARTILE_WINDOW, min_periods=QUARTILE_WINDOW).quantile(0.75)

# Generate signals
results_df['signal'] = 0
results_df.loc[results_df[pred_col] >= results_df['q3'], 'signal'] = 1  # BUY
results_df.loc[results_df[pred_col] <= results_df['q1'], 'signal'] = -1  # SELL

# Drop initial bars where we don't have quartiles yet
results_df = results_df.dropna(subset=['q1', 'q3'])

print(f"\nAfter quartile calculation: {len(results_df)} bars")
print(f"Date range: {results_df.index.min()} to {results_df.index.max()}")

# Signal distribution
signal_counts = results_df['signal'].value_counts().sort_index()
print(f"\nSignal distribution:")
for sig, count in signal_counts.items():
    sig_name = "SELL" if sig == -1 else ("HOLD" if sig == 0 else "BUY")
    print(f"  {sig_name:6s} ({sig:+2d}): {count:5d} bars ({count/len(results_df)*100:5.1f}%)")

# Implement HOLD UNTIL NEXT SIGNAL strategy
print("\n" + "=" * 80)
print("SIMULATING TRADES - HOLD UNTIL NEXT SIGNAL")
print("=" * 80)

trades = []
position = 0  # 0=flat, 1=long, -1=short
entry_price = 0
entry_date = None
entry_idx = None

for i, (date, row) in enumerate(results_df.iterrows()):
    signal = row['signal']
    current_price = row['close']

    # Check if we need to close current position and open new one
    if signal != 0 and signal != position:
        # Close existing position if we have one
        if position != 0:
            # Calculate P&L in pips
            if position == 1:  # Close long
                pips = (current_price - entry_price) * 10000
            else:  # Close short
                pips = (entry_price - current_price) * 10000

            # Calculate P&L in dollars
            gross_pnl = pips * PIP_VALUE

            # Calculate transaction costs (entry + exit)
            entry_cost = entry_price * LOT_SIZE * TRANSACTION_COST
            exit_cost = current_price * LOT_SIZE * TRANSACTION_COST
            total_cost = entry_cost + exit_cost

            net_pnl = gross_pnl - total_cost

            # Calculate return
            trade_return = net_pnl / STARTING_CAPITAL

            # Calculate bars held
            bars_held = i - entry_idx

            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'direction': 'LONG' if position == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': current_price,
                'pips': pips,
                'gross_pnl': gross_pnl,
                'transaction_cost': total_cost,
                'net_pnl': net_pnl,
                'return': trade_return,
                'bars_held': bars_held
            })

        # Open new position
        position = signal
        entry_price = current_price
        entry_date = date
        entry_idx = i

# Close final position if still open
if position != 0:
    final_row = results_df.iloc[-1]
    current_price = final_row['close']

    if position == 1:  # Close long
        pips = (current_price - entry_price) * 10000
    else:  # Close short
        pips = (entry_price - current_price) * 10000

    gross_pnl = pips * PIP_VALUE
    entry_cost = entry_price * LOT_SIZE * TRANSACTION_COST
    exit_cost = current_price * LOT_SIZE * TRANSACTION_COST
    total_cost = entry_cost + exit_cost
    net_pnl = gross_pnl - total_cost
    trade_return = net_pnl / STARTING_CAPITAL
    bars_held = len(results_df) - 1 - entry_idx

    trades.append({
        'entry_date': entry_date,
        'exit_date': results_df.index[-1],
        'direction': 'LONG' if position == 1 else 'SHORT',
        'entry_price': entry_price,
        'exit_price': current_price,
        'pips': pips,
        'gross_pnl': gross_pnl,
        'transaction_cost': total_cost,
        'net_pnl': net_pnl,
        'return': trade_return,
        'bars_held': bars_held
    })

# Convert to DataFrame
trades_df = pd.DataFrame(trades)

if len(trades_df) == 0:
    print("\nNo trades executed!")
    exit(0)

print(f"\nTotal trades: {len(trades_df)}")
print(f"Long trades:  {len(trades_df[trades_df['direction']=='LONG'])} ({len(trades_df[trades_df['direction']=='LONG'])/len(trades_df)*100:.1f}%)")
print(f"Short trades: {len(trades_df[trades_df['direction']=='SHORT'])} ({len(trades_df[trades_df['direction']=='SHORT'])/len(trades_df)*100:.1f}%)")

# Calculate metrics
trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
trades_df['cumulative_return'] = trades_df['return'].cumsum()
trades_df['equity'] = STARTING_CAPITAL + trades_df['cumulative_pnl']

# Overall statistics
print("\n" + "=" * 80)
print("PERFORMANCE STATISTICS")
print("=" * 80)

total_pnl = trades_df['net_pnl'].sum()
total_return = total_pnl / STARTING_CAPITAL
final_equity = STARTING_CAPITAL + total_pnl

print(f"\nOverall Performance:")
print(f"  Starting capital:  ${STARTING_CAPITAL:,.2f}")
print(f"  Final equity:      ${final_equity:,.2f}")
print(f"  Total P&L:         ${total_pnl:,.2f}")
print(f"  Total return:      {total_return*100:+.2f}%")

# Calculate time period
start_date = results_df.index.min()
end_date = results_df.index.max()
years = (end_date - start_date).days / 365.25

print(f"\nTime Period:")
print(f"  Start: {start_date}")
print(f"  End:   {end_date}")
print(f"  Duration: {years:.2f} years")

# Annualized return
annual_return = (((final_equity / STARTING_CAPITAL) ** (1/years)) - 1) * 100
print(f"  Annualized return: {annual_return:+.2f}%")

# Win rate
winning_trades = trades_df[trades_df['net_pnl'] > 0]
losing_trades = trades_df[trades_df['net_pnl'] < 0]
win_rate = len(winning_trades) / len(trades_df)

print(f"\nTrade Statistics:")
print(f"  Win rate:        {win_rate*100:.2f}% ({len(winning_trades)}/{len(trades_df)})")
print(f"  Avg win:         ${winning_trades['net_pnl'].mean():.2f} ({winning_trades['pips'].mean():.1f} pips)")
print(f"  Avg loss:        ${losing_trades['net_pnl'].mean():.2f} ({losing_trades['pips'].mean():.1f} pips)")
print(f"  Avg bars held:   {trades_df['bars_held'].mean():.1f} bars")

# Profit factor
gross_profit = winning_trades['net_pnl'].sum()
gross_loss = abs(losing_trades['net_pnl'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

print(f"  Profit factor:   {profit_factor:.2f}")

# Max drawdown
trades_df['peak'] = trades_df['equity'].cummax()
trades_df['drawdown'] = (trades_df['equity'] - trades_df['peak']) / trades_df['peak']
max_drawdown = trades_df['drawdown'].min()

print(f"  Max drawdown:    {max_drawdown*100:.2f}%")

# Sharpe ratio (approximate - using trade returns)
if len(trades_df) > 1:
    trade_returns = trades_df['return']
    sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(252/trades_df['bars_held'].mean()) if trade_returns.std() > 0 else 0
    print(f"  Sharpe ratio:    {sharpe:.2f}")

# Breakdown by direction
print("\n" + "=" * 80)
print("PERFORMANCE BY DIRECTION")
print("=" * 80)

for direction in ['LONG', 'SHORT']:
    dir_trades = trades_df[trades_df['direction'] == direction]
    if len(dir_trades) == 0:
        continue

    dir_pnl = dir_trades['net_pnl'].sum()
    dir_return = dir_pnl / STARTING_CAPITAL
    dir_wins = len(dir_trades[dir_trades['net_pnl'] > 0])
    dir_win_rate = dir_wins / len(dir_trades)

    print(f"\n{direction}:")
    print(f"  Trades:     {len(dir_trades)}")
    print(f"  Total P&L:  ${dir_pnl:,.2f}")
    print(f"  Return:     {dir_return*100:+.2f}%")
    print(f"  Win rate:   {dir_win_rate*100:.2f}%")
    print(f"  Avg P&L:    ${dir_trades['net_pnl'].mean():.2f}")
    print(f"  Avg pips:   {dir_trades['pips'].mean():.1f}")

# First 10 trades
print("\n" + "=" * 80)
print("FIRST 10 TRADES")
print("=" * 80)
print(trades_df.head(10)[['entry_date', 'exit_date', 'direction', 'pips', 'net_pnl', 'bars_held']].to_string(index=False))

# Last 10 trades
print("\n" + "=" * 80)
print("LAST 10 TRADES")
print("=" * 80)
print(trades_df.tail(10)[['entry_date', 'exit_date', 'direction', 'pips', 'net_pnl', 'bars_held']].to_string(index=False))

# Compare to study
print("\n" + "=" * 80)
print("COMPARISON TO STUDY")
print("=" * 80)
print("\nStudy (2000-2023, 23.5 years):")
print("  Start capital: $1,000")
print("  Final equity:  $12,879")
print("  Total return:  +1,187.9%")
print("  Annual return: ~11%")

print(f"\nOur results ({years:.2f} years):")
print(f"  Start capital: ${STARTING_CAPITAL:,.2f}")
print(f"  Final equity:  ${final_equity:,.2f}")
print(f"  Total return:  {total_return*100:+.2f}%")
print(f"  Annual return: {annual_return:+.2f}%")

# Save results
output_file = f'backtest_study_methodology_{CURRENCY_PAIR}.pkl'
with open(output_file, 'wb') as f:
    pickle.dump({
        'trades': trades_df,
        'summary': {
            'starting_capital': STARTING_CAPITAL,
            'final_equity': final_equity,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'annual_return': annual_return,
            'years': years,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe if 'sharpe' in locals() else 0
        }
    }, f)

print(f"\n\nResults saved to: {output_file}")
print("=" * 80)
