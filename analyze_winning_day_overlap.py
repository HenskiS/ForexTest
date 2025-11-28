"""
Analyze which days each pair wins and the overlap between them.

Key questions:
1. How often do pairs win on the same days?
2. How many days have at least 1 win? 2 wins? All 4?
3. Are winning days correlated or diversified?
"""
import pandas as pd
import numpy as np

PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']

print("Winning Day Overlap Analysis")
print("=" * 80)

# Load all trade data
pair_trades = {}
POSITION_SIZE = 250  # $250 per pair (2:1 leverage with $125 capital per pair)
for pair in PAIRS:
    df = pd.read_csv(f'{pair}_backtest_trades_1000days.csv')
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df['is_win'] = df['outcome'] == 'WIN'
    df['dollar_pnl'] = (df['net_return_pct'] / 100) * POSITION_SIZE  # Convert % return to $ P&L
    pair_trades[pair] = df
    print(f"Loaded {pair}: {len(df)} trades, {df['is_win'].sum()} wins ({df['is_win'].mean()*100:.1f}%)")

# Find common date range
print("\n" + "=" * 80)
print("Finding common date range...")

all_dates = None
for pair in PAIRS:
    dates = set(pair_trades[pair]['exit_date'].dt.date)
    if all_dates is None:
        all_dates = dates
    else:
        all_dates = all_dates.intersection(dates)

all_dates = sorted(list(all_dates))
print(f"Common dates: {len(all_dates)} days ({all_dates[0]} to {all_dates[-1]})")

# Create daily win matrix
print("\nBuilding daily win matrix...")

daily_wins = pd.DataFrame(index=all_dates)

for pair in PAIRS:
    df = pair_trades[pair].copy()
    df['exit_date_only'] = df['exit_date'].dt.date

    # Group by exit date and check if won that day
    daily_pair_wins = df.groupby('exit_date_only')['is_win'].apply(lambda x: x.values[0] if len(x) > 0 else False)
    daily_pair_pnl = df.groupby('exit_date_only')['dollar_pnl'].apply(lambda x: x.values[0] if len(x) > 0 else 0)

    daily_wins[pair] = daily_pair_wins.reindex(all_dates, fill_value=False)
    daily_wins[f'{pair}_pnl'] = daily_pair_pnl.reindex(all_dates, fill_value=0)

# Calculate statistics
print("\n" + "=" * 80)
print("WINNING DAY STATISTICS")
print("=" * 80)

# Count wins per day and calculate daily P&L
daily_wins['win_count'] = daily_wins[PAIRS].sum(axis=1)
daily_wins['has_any_win'] = daily_wins['win_count'] > 0
daily_wins['daily_pnl'] = sum(daily_wins[f'{pair}_pnl'] for pair in PAIRS)

print(f"\nTotal trading days analyzed: {len(daily_wins)}")
print(f"\nDays by number of wins:")
print("-" * 80)

for i in range(len(PAIRS) + 1):
    subset = daily_wins[daily_wins['win_count'] == i]
    count = len(subset)
    pct = count / len(daily_wins) * 100
    avg_pnl = subset['daily_pnl'].mean()
    print(f"  {i} wins: {count:>4} days ({pct:>5.1f}%)  Avg P&L: ${avg_pnl:>7.2f}")

print(f"\n{'*' * 80}")
winning_days = daily_wins[daily_wins['has_any_win']]
losing_days = daily_wins[~daily_wins['has_any_win']]
print(f"Days with at least 1 win: {len(winning_days)} ({daily_wins['has_any_win'].mean()*100:.1f}%)  Avg P&L: ${winning_days['daily_pnl'].mean():.2f}")
print(f"Days with 0 wins: {len(losing_days)} ({(~daily_wins['has_any_win']).mean()*100:.1f}%)  Avg P&L: ${losing_days['daily_pnl'].mean():.2f}")
print(f"{'*' * 80}")

# Pairwise correlation
print("\n" + "=" * 80)
print("PAIRWISE WIN CORRELATION")
print("=" * 80)

correlation_matrix = daily_wins[PAIRS].astype(int).corr()

print("\nCorrelation Matrix (1.0 = always win together, 0.0 = independent):")
print(correlation_matrix.round(3))

# Average correlation
avg_corr = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
print(f"\nAverage correlation: {avg_corr:.3f}")

if avg_corr < 0.3:
    print("-> LOW correlation: Pairs win independently (good diversification!)")
elif avg_corr < 0.6:
    print("-> MODERATE correlation: Some shared winning days")
else:
    print("-> HIGH correlation: Pairs often win together (limited diversification)")

# Specific pair combinations
print("\n" + "=" * 80)
print("SPECIFIC PAIR COMBINATIONS")
print("=" * 80)

print("\nHow often do pairs win TOGETHER on the same day:")
print("-" * 80)

for i, pair1 in enumerate(PAIRS):
    for pair2 in PAIRS[i+1:]:
        both_win = (daily_wins[pair1] & daily_wins[pair2]).sum()
        pair1_trades = daily_wins[pair1].sum()
        pair2_trades = daily_wins[pair2].sum()

        if pair1_trades > 0:
            overlap_pct1 = both_win / pair1_trades * 100
        else:
            overlap_pct1 = 0

        if pair2_trades > 0:
            overlap_pct2 = both_win / pair2_trades * 100
        else:
            overlap_pct2 = 0

        print(f"  {pair1} & {pair2}: {both_win} days ({overlap_pct1:.1f}% of {pair1} wins, {overlap_pct2:.1f}% of {pair2} wins)")

# Expected wins per week/month
print("\n" + "=" * 80)
print("EXPECTED WINNING FREQUENCY")
print("=" * 80)

winning_days = daily_wins['has_any_win'].sum()
total_days = len(daily_wins)

wins_per_week = (winning_days / total_days) * 5  # 5 trading days/week
wins_per_month = (winning_days / total_days) * 21  # ~21 trading days/month

print(f"\nExpected days with at least 1 win:")
print(f"  Per week: {wins_per_week:.1f} days")
print(f"  Per month: {wins_per_month:.1f} days")
print(f"\nExpected losing days (all 4 pairs lose):")
losing_days = (daily_wins['win_count'] == 0).sum()
losses_per_week = (losing_days / total_days) * 5
losses_per_month = (losing_days / total_days) * 21
print(f"  Per week: {losses_per_week:.1f} days")
print(f"  Per month: {losses_per_month:.1f} days")

# Best/worst streaks
print("\n" + "=" * 80)
print("WINNING/LOSING STREAKS")
print("=" * 80)

# Calculate streaks
has_win = daily_wins['has_any_win'].astype(int)
streak = has_win.ne(has_win.shift()).cumsum()
win_streaks = has_win.groupby(streak).sum()
win_streak_lengths = win_streaks[win_streaks > 0]

has_loss = (~daily_wins['has_any_win']).astype(int)
loss_streak = has_loss.ne(has_loss.shift()).cumsum()
loss_streaks = has_loss.groupby(loss_streak).sum()
loss_streak_lengths = loss_streaks[loss_streaks > 0]

if len(win_streak_lengths) > 0:
    print(f"\nLongest winning streak (days with at least 1 win): {win_streak_lengths.max()} days")
    print(f"Average winning streak: {win_streak_lengths.mean():.1f} days")

if len(loss_streak_lengths) > 0:
    print(f"\nLongest losing streak (all 4 pairs lose): {loss_streak_lengths.max()} days")
    print(f"Average losing streak: {loss_streak_lengths.mean():.1f} days")

# Daily win distribution visualization
print("\n" + "=" * 80)
print("DAILY WIN DISTRIBUTION (Visual)")
print("=" * 80)

for i in range(len(PAIRS) + 1):
    count = (daily_wins['win_count'] == i).sum()
    pct = count / len(daily_wins) * 100
    bar = '#' * int(pct / 2)  # Each # = 2%
    print(f"{i} wins: {bar} {pct:.1f}%")

print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

winning_day_rate = daily_wins['has_any_win'].mean() * 100
avg_wins_per_day = daily_wins['win_count'].mean()

avg_daily_pnl = daily_wins['daily_pnl'].mean()
winning_days_for_summary = daily_wins[daily_wins['has_any_win']]
losing_days_for_summary = daily_wins[~daily_wins['has_any_win']]
avg_winning_day_pnl = winning_days_for_summary['daily_pnl'].mean()
avg_losing_day_pnl = losing_days_for_summary['daily_pnl'].mean()

print(f"""
With 4-pair diversification ($250 position per pair):
- You'll have at least 1 winning pair on {winning_day_rate:.1f}% of days
- Average wins per day: {avg_wins_per_day:.2f}
- You'll have {wins_per_month:.0f} winning days per month
- Only {losses_per_month:.0f} days/month where all 4 pairs lose

Daily P&L Statistics:
- Average daily P&L: ${avg_daily_pnl:.2f}
- Average winning day: ${avg_winning_day_pnl:.2f}
- Average losing day: ${avg_losing_day_pnl:.2f}
- Best day: ${daily_wins['daily_pnl'].max():.2f}
- Worst day: ${daily_wins['daily_pnl'].min():.2f}

This means:
+ Frequent wins keep you motivated
+ Diversification cushions bad days
+ ~{100-losing_days/total_days*100:.0f}% of days have at least 1 win
""")

# Save detailed results
output_file = 'winning_day_overlap_analysis.csv'
daily_wins.to_csv(output_file)
print(f"\nDetailed daily win data saved to: {output_file}")
