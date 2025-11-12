"""
Analyze return distribution by exit type across all currency pairs.
"""

import pandas as pd
import numpy as np
import glob

print("RETURN DISTRIBUTION BY EXIT TYPE")
print("="*80)

# Find all trade CSV files
trade_files = glob.glob('xgboost_trades_*_all.csv')

# Add the original EURUSD file if it exists
if 'xgboost_trades_all.csv' in glob.glob('xgboost_trades_all.csv'):
    trade_files.append('xgboost_trades_all.csv')

if not trade_files:
    print("No trade files found. Run export_trades scripts first.")
    exit(1)

print(f"Found {len(trade_files)} currency pair files\n")

# Load all trades
all_trades = []
for file in trade_files:
    df = pd.read_csv(file)
    all_trades.append(df)

# Combine all trades
combined_df = pd.concat(all_trades, ignore_index=True)

print(f"Total trades analyzed: {len(combined_df)}")
print(f"Date range: {combined_df['signal_date'].min()} to {combined_df['signal_date'].max()}")

# Define categories
target_wins = combined_df[(combined_df['exit_reason'] == 'Target') & (combined_df['outcome'] == 'WIN')]
stop_losses = combined_df[(combined_df['exit_reason'] == 'Stop Loss') & (combined_df['outcome'] == 'LOSS')]
time_wins = combined_df[(combined_df['exit_reason'] == 'Time Exit (5 days)') & (combined_df['outcome'] == 'WIN')]
time_losses = combined_df[(combined_df['exit_reason'] == 'Time Exit (5 days)') & (combined_df['outcome'] == 'LOSS')]

print("\n" + "="*80)
print("RETURN STATISTICS BY EXIT TYPE")
print("="*80)

def print_stats(name, trades):
    """Print statistics for a category of trades."""
    if len(trades) == 0:
        print(f"\n{name}: NO TRADES")
        return

    returns = trades['net_return_pct']

    print(f"\n{name}:")
    print(f"  Count:        {len(trades):6d} trades")
    print(f"  Min:          {returns.min():7.3f}%")
    print(f"  Mean:         {returns.mean():7.3f}%")
    print(f"  Median:       {returns.median():7.3f}%")
    print(f"  Max:          {returns.max():7.3f}%")
    print(f"  Std Dev:      {returns.std():7.3f}%")
    print(f"  25th %ile:    {returns.quantile(0.25):7.3f}%")
    print(f"  75th %ile:    {returns.quantile(0.75):7.3f}%")

# Print statistics for each category
print_stats("TARGET WINS (Hit Take-Profit)", target_wins)
print_stats("STOP LOSSES (Hit Stop-Loss)", stop_losses)
print_stats("TIME EXIT WINS (5-day profitable)", time_wins)
print_stats("TIME EXIT LOSSES (5-day unprofitable)", time_losses)

# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"\n{'Exit Type':<30} {'Count':>8} {'Min %':>9} {'Mean %':>9} {'Max %':>9}")
print("-" * 80)

categories = [
    ("Target Wins", target_wins),
    ("Stop Losses", stop_losses),
    ("Time Exit Wins (5d)", time_wins),
    ("Time Exit Losses (5d)", time_losses)
]

for name, trades in categories:
    if len(trades) > 0:
        returns = trades['net_return_pct']
        print(f"{name:<30} {len(trades):8d} {returns.min():9.3f} {returns.mean():9.3f} {returns.max():9.3f}")
    else:
        print(f"{name:<30} {0:8d} {'N/A':>9} {'N/A':>9} {'N/A':>9}")

# Key insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

if len(target_wins) > 0:
    target_mean = target_wins['net_return_pct'].mean()
    print(f"\n1. Target wins average {target_mean:.3f}% return")
    print(f"   (Close to theoretical 2.5:1 R/R after 0.02% transaction costs)")

if len(stop_losses) > 0:
    stop_mean = stop_losses['net_return_pct'].mean()
    print(f"\n2. Stop losses average {stop_mean:.3f}% loss")
    print(f"   (Close to -0.40% base stop after transaction costs)")

if len(time_wins) > 0:
    time_win_mean = time_wins['net_return_pct'].mean()
    time_win_min = time_wins['net_return_pct'].min()
    time_win_max = time_wins['net_return_pct'].max()
    print(f"\n3. Time exit wins average {time_win_mean:.3f}% return")
    print(f"   Range: {time_win_min:.3f}% to {time_win_max:.3f}%")
    print(f"   These are the 'small winners' that compound over time")

if len(time_losses) > 0:
    time_loss_mean = time_losses['net_return_pct'].mean()
    time_loss_min = time_losses['net_return_pct'].min()
    time_loss_max = time_losses['net_return_pct'].max()
    print(f"\n4. Time exit losses average {time_loss_mean:.3f}% loss")
    print(f"   Range: {time_loss_max:.3f}% to {time_loss_min:.3f}%")
    print(f"   These are positions that didn't hit stop but still lost money")

# Compare time wins vs time losses
if len(time_wins) > 0 and len(time_losses) > 0:
    time_win_mean = time_wins['net_return_pct'].mean()
    time_loss_mean = time_losses['net_return_pct'].mean()
    ratio = abs(time_win_mean / time_loss_mean)
    print(f"\n5. Time exit win/loss ratio: {ratio:.2f}:1")
    print(f"   Time wins average {abs(time_win_mean/time_loss_mean):.2f}x larger than time losses")

# Overall strategy profile
print("\n" + "="*80)
print("STRATEGY PROFILE")
print("="*80)

total_wins = len(target_wins) + len(time_wins)
total_losses = len(stop_losses) + len(time_losses)

print(f"\nTotal winning trades: {total_wins}")
print(f"  - Big wins (target hit): {len(target_wins)} ({len(target_wins)/total_wins*100:.1f}%)")
print(f"  - Small wins (time exit): {len(time_wins)} ({len(time_wins)/total_wins*100:.1f}%)")

print(f"\nTotal losing trades: {total_losses}")
print(f"  - Big losses (stop hit): {len(stop_losses)} ({len(stop_losses)/total_losses*100:.1f}%)")
print(f"  - Small losses (time exit): {len(time_losses)} ({len(time_losses)/total_losses*100:.1f}%)")

# Calculate weighted average returns
all_returns = combined_df['net_return_pct']
avg_return_per_trade = all_returns.mean()

print(f"\nOverall average return per trade: {avg_return_per_trade:.3f}%")
print(f"Win rate: {total_wins/(total_wins+total_losses)*100:.2f}%")

print("\n" + "="*80)
