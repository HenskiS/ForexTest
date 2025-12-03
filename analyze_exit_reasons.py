"""
Analyze exit reasons from backtest trades.
"""
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='Backtest trades CSV file')
args = parser.parse_args()

# Load trades
trades_df = pd.read_csv(args.file)

print(f"\n{'='*80}")
print(f"EXIT REASON ANALYSIS")
print(f"{'='*80}")
print(f"File: {args.file}")
print(f"Total Trades: {len(trades_df)}")

# Count exit reasons
exit_counts = trades_df['exit_reason'].value_counts()
exit_percentages = trades_df['exit_reason'].value_counts(normalize=True) * 100

print(f"\n{'='*80}")
print("EXIT REASON BREAKDOWN")
print(f"{'='*80}")
for reason in exit_counts.index:
    count = exit_counts[reason]
    pct = exit_percentages[reason]
    print(f"{reason:20s}: {count:6d} trades ({pct:5.1f}%)")

# Breakdown by outcome
print(f"\n{'='*80}")
print("EXIT REASONS BY OUTCOME")
print(f"{'='*80}")

for reason in ['TAKE_PROFIT', 'STOP_LOSS', 'TIME_EXIT']:
    if reason not in exit_counts.index:
        continue

    reason_trades = trades_df[trades_df['exit_reason'] == reason]
    wins = reason_trades[reason_trades['outcome'] == 'WIN']
    losses = reason_trades[reason_trades['outcome'] == 'LOSS']

    win_rate = len(wins) / len(reason_trades) * 100 if len(reason_trades) > 0 else 0
    avg_return = reason_trades['net_return_pct'].mean()

    print(f"\n{reason}:")
    print(f"  Total: {len(reason_trades)} trades")
    print(f"  Wins: {len(wins)} ({len(wins)/len(reason_trades)*100:.1f}%)")
    print(f"  Losses: {len(losses)} ({len(losses)/len(reason_trades)*100:.1f}%)")
    print(f"  Avg Return: {avg_return:.2f}%")

# Statistics by exit type
print(f"\n{'='*80}")
print("RETURN STATISTICS BY EXIT TYPE")
print(f"{'='*80}")

for reason in exit_counts.index:
    reason_trades = trades_df[trades_df['exit_reason'] == reason]

    print(f"\n{reason}:")
    print(f"  Mean: {reason_trades['net_return_pct'].mean():.3f}%")
    print(f"  Median: {reason_trades['net_return_pct'].median():.3f}%")
    print(f"  Std Dev: {reason_trades['net_return_pct'].std():.3f}%")
    print(f"  Min: {reason_trades['net_return_pct'].min():.3f}%")
    print(f"  Max: {reason_trades['net_return_pct'].max():.3f}%")

# Direction analysis
print(f"\n{'='*80}")
print("EXIT REASONS BY DIRECTION")
print(f"{'='*80}")

for direction in ['LONG', 'SHORT']:
    dir_trades = trades_df[trades_df['direction'] == direction]
    print(f"\n{direction} Trades ({len(dir_trades)} total):")

    for reason in exit_counts.index:
        reason_dir = dir_trades[dir_trades['exit_reason'] == reason]
        if len(reason_dir) > 0:
            pct = len(reason_dir) / len(dir_trades) * 100
            avg_ret = reason_dir['net_return_pct'].mean()
            print(f"  {reason:20s}: {len(reason_dir):5d} ({pct:5.1f}%) | Avg: {avg_ret:6.2f}%")

print(f"\n{'='*80}")
