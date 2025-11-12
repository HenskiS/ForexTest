"""
Analyze exit reasons for winning trades across all currency pairs.
"""

import pandas as pd
import glob

print("EXIT REASON ANALYSIS")
print("="*80)

# Find all trade CSV files
trade_files = glob.glob('xgboost_trades_*_all.csv')

if not trade_files:
    print("No trade files found. Run export_trades scripts first.")
    exit(1)

print(f"Found {len(trade_files)} currency pair files\n")

# Analyze each pair
all_results = []

for file in sorted(trade_files):
    # Extract pair name from filename
    if 'EURUSD' in file:
        pair = 'EURUSD'
    elif 'GBPUSD' in file:
        pair = 'GBPUSD'
    elif 'USDJPY' in file:
        pair = 'USDJPY'
    elif 'AUDUSD' in file:
        pair = 'AUDUSD'
    else:
        # This is the original EURUSD file without pair prefix
        pair = 'EURUSD'

    df = pd.read_csv(file)

    # Filter for wins only
    wins = df[df['outcome'] == 'WIN'].copy()
    losses = df[df['outcome'] == 'LOSS'].copy()

    # Count exit reasons for wins
    exit_reasons_wins = wins['exit_reason'].value_counts()

    # Count exit reasons for losses
    exit_reasons_losses = losses['exit_reason'].value_counts()

    result = {
        'pair': pair,
        'total_trades': len(df),
        'total_wins': len(wins),
        'total_losses': len(losses),
        'win_rate': len(wins) / len(df) * 100,
        'wins_target': exit_reasons_wins.get('Target', 0),
        'wins_time_exit': exit_reasons_wins.get('Time Exit (5 days)', 0),
        'wins_stop_loss': exit_reasons_wins.get('Stop Loss', 0),
        'losses_stop_loss': exit_reasons_losses.get('Stop Loss', 0),
        'losses_time_exit': exit_reasons_losses.get('Time Exit (5 days)', 0),
        'losses_target': exit_reasons_losses.get('Target', 0),
    }

    all_results.append(result)

# Print results for each pair
print("\n" + "="*80)
print("EXIT REASON BREAKDOWN BY PAIR")
print("="*80)

for r in all_results:
    print(f"\n{r['pair']}:")
    print(f"  Total trades: {r['total_trades']}")
    print(f"  Wins: {r['total_wins']} ({r['win_rate']:.2f}%)")
    print(f"  Losses: {r['total_losses']}")

    print(f"\n  WINNING TRADES:")
    if r['total_wins'] > 0:
        pct_target = r['wins_target'] / r['total_wins'] * 100
        pct_time = r['wins_time_exit'] / r['total_wins'] * 100
        pct_stop = r['wins_stop_loss'] / r['total_wins'] * 100

        print(f"    Hit Target:        {r['wins_target']:4d} ({pct_target:5.1f}%)")
        print(f"    Time Exit (5d):    {r['wins_time_exit']:4d} ({pct_time:5.1f}%)")
        print(f"    Stop Loss:         {r['wins_stop_loss']:4d} ({pct_stop:5.1f}%)")

    print(f"\n  LOSING TRADES:")
    if r['total_losses'] > 0:
        pct_stop = r['losses_stop_loss'] / r['total_losses'] * 100
        pct_time = r['losses_time_exit'] / r['total_losses'] * 100
        pct_target = r['losses_target'] / r['total_losses'] * 100

        print(f"    Stop Loss:         {r['losses_stop_loss']:4d} ({pct_stop:5.1f}%)")
        print(f"    Time Exit (5d):    {r['losses_time_exit']:4d} ({pct_time:5.1f}%)")
        print(f"    Target:            {r['losses_target']:4d} ({pct_target:5.1f}%)")

# Aggregate across all pairs
print("\n" + "="*80)
print("AGGREGATE ACROSS ALL PAIRS")
print("="*80)

total_wins = sum(r['total_wins'] for r in all_results)
total_losses = sum(r['total_losses'] for r in all_results)
total_trades = sum(r['total_trades'] for r in all_results)

wins_target = sum(r['wins_target'] for r in all_results)
wins_time = sum(r['wins_time_exit'] for r in all_results)
wins_stop = sum(r['wins_stop_loss'] for r in all_results)

losses_stop = sum(r['losses_stop_loss'] for r in all_results)
losses_time = sum(r['losses_time_exit'] for r in all_results)
losses_target = sum(r['losses_target'] for r in all_results)

print(f"\nTotal trades: {total_trades}")
print(f"Total wins: {total_wins} ({total_wins/total_trades*100:.2f}%)")
print(f"Total losses: {total_losses} ({total_losses/total_trades*100:.2f}%)")

print(f"\nWINNING TRADES ({total_wins} total):")
print(f"  Hit Target:        {wins_target:5d} ({wins_target/total_wins*100:5.1f}%)")
print(f"  Time Exit (5d):    {wins_time:5d} ({wins_time/total_wins*100:5.1f}%)")
print(f"  Stop Loss:         {wins_stop:5d} ({wins_stop/total_wins*100:5.1f}%)")

print(f"\nLOSING TRADES ({total_losses} total):")
print(f"  Stop Loss:         {losses_stop:5d} ({losses_stop/total_losses*100:5.1f}%)")
print(f"  Time Exit (5d):    {losses_time:5d} ({losses_time/total_losses*100:5.1f}%)")
print(f"  Target:            {losses_target:5d} ({losses_target/total_losses*100:5.1f}%)")

# Key insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

pct_wins_from_time = wins_time / total_wins * 100
pct_wins_from_target = wins_target / total_wins * 100

print(f"\n1. {pct_wins_from_time:.1f}% of wins come from Time Exit (5 days)")
print(f"2. {pct_wins_from_target:.1f}% of wins come from hitting Target")
print(f"3. {wins_stop} wins ({wins_stop/total_wins*100:.1f}%) came from Stop Loss being hit")
print(f"   (likely due to favorable price action after entry)")

print(f"\n4. Most losses ({losses_stop/total_losses*100:.1f}%) come from Stop Loss")
print(f"5. {losses_time/total_losses*100:.1f}% of losses come from Time Exit")
print(f"   (trades that didn't hit stop but didn't go in our favor)")

# Target hit rate
total_target_attempts = total_trades  # Every trade attempts to hit target
target_hit_rate = wins_target / total_target_attempts * 100
print(f"\n6. Target hit rate: {target_hit_rate:.2f}% of all trades hit the target")
print(f"   ({wins_target} targets hit out of {total_target_attempts} trades)")

print("\n" + "="*80)
