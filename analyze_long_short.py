"""
Analyze long vs short performance for all commodities.
"""
import pandas as pd
import numpy as np
import sys
import io

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# All commodities including oil
COMMODITIES = ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'XCUUSD', 'BCOUSD']
NAMES = {
    'XAUUSD': 'Gold',
    'XAGUSD': 'Silver',
    'XPTUSD': 'Platinum',
    'XPDUSD': 'Palladium',
    'XCUUSD': 'Copper',
    'BCOUSD': 'Oil (Brent)'
}

def analyze_long_short(pair):
    """Analyze long vs short for a single commodity"""
    try:
        trades_file = f'{pair}_backtest_trades_750days.csv'
        df = pd.read_csv(trades_file)

        # Separate long and short trades
        long_trades = df[df['direction'] == 'LONG']
        short_trades = df[df['direction'] == 'SHORT']

        # Calculate stats for long trades
        if len(long_trades) > 0:
            long_wins = long_trades[long_trades['outcome'] == 'WIN']
            long_losses = long_trades[long_trades['outcome'] == 'LOSS']

            long_stats = {
                'total': len(long_trades),
                'wins': len(long_wins),
                'losses': len(long_losses),
                'win_rate': len(long_wins) / len(long_trades) * 100,
                'avg_return': long_trades['net_return_pct'].mean(),
                'total_return': long_trades['net_return_pct'].sum(),
                'avg_win': long_wins['net_return_pct'].mean() if len(long_wins) > 0 else 0,
                'avg_loss': long_losses['net_return_pct'].mean() if len(long_losses) > 0 else 0,
                'profit_factor': abs(long_wins['net_return_pct'].sum() / long_losses['net_return_pct'].sum()) if len(long_losses) > 0 and long_losses['net_return_pct'].sum() != 0 else 0
            }
        else:
            long_stats = None

        # Calculate stats for short trades
        if len(short_trades) > 0:
            short_wins = short_trades[short_trades['outcome'] == 'WIN']
            short_losses = short_trades[short_trades['outcome'] == 'LOSS']

            short_stats = {
                'total': len(short_trades),
                'wins': len(short_wins),
                'losses': len(short_losses),
                'win_rate': len(short_wins) / len(short_trades) * 100,
                'avg_return': short_trades['net_return_pct'].mean(),
                'total_return': short_trades['net_return_pct'].sum(),
                'avg_win': short_wins['net_return_pct'].mean() if len(short_wins) > 0 else 0,
                'avg_loss': short_losses['net_return_pct'].mean() if len(short_losses) > 0 else 0,
                'profit_factor': abs(short_wins['net_return_pct'].sum() / short_losses['net_return_pct'].sum()) if len(short_losses) > 0 and short_losses['net_return_pct'].sum() != 0 else 0
            }
        else:
            short_stats = None

        return {
            'pair': pair,
            'name': NAMES[pair],
            'long': long_stats,
            'short': short_stats
        }

    except Exception as e:
        print(f"Error analyzing {pair}: {e}")
        return None

print("="*80)
print("LONG vs SHORT PERFORMANCE ANALYSIS")
print("="*80)
print()

all_results = []
for pair in COMMODITIES:
    result = analyze_long_short(pair)
    if result:
        all_results.append(result)

# Detailed breakdown for each commodity
for res in all_results:
    print(f"\n{'='*80}")
    print(f"{res['name']} ({res['pair']})")
    print(f"{'='*80}")

    if res['long']:
        print(f"\nLONG Trades:")
        print(f"  Total Trades: {res['long']['total']}")
        print(f"  Win Rate: {res['long']['win_rate']:.1f}% ({res['long']['wins']}/{res['long']['total']})")
        print(f"  Avg Return per Trade: {res['long']['avg_return']:.2f}%")
        print(f"  Total Return: {res['long']['total_return']:.2f}%")
        print(f"  Avg Win: {res['long']['avg_win']:.2f}%")
        print(f"  Avg Loss: {res['long']['avg_loss']:.2f}%")
        print(f"  Profit Factor: {res['long']['profit_factor']:.2f}")
    else:
        print(f"\nLONG Trades: None")

    if res['short']:
        print(f"\nSHORT Trades:")
        print(f"  Total Trades: {res['short']['total']}")
        print(f"  Win Rate: {res['short']['win_rate']:.1f}% ({res['short']['wins']}/{res['short']['total']})")
        print(f"  Avg Return per Trade: {res['short']['avg_return']:.2f}%")
        print(f"  Total Return: {res['short']['total_return']:.2f}%")
        print(f"  Avg Win: {res['short']['avg_win']:.2f}%")
        print(f"  Avg Loss: {res['short']['avg_loss']:.2f}%")
        print(f"  Profit Factor: {res['short']['profit_factor']:.2f}")
    else:
        print(f"\nSHORT Trades: None")

    # Compare
    if res['long'] and res['short']:
        print(f"\nCOMPARISON:")
        print(f"  Long/Short Ratio: {res['long']['total']}/{res['short']['total']} = {res['long']['total']/res['short']['total']:.2f}:1")
        if res['long']['total_return'] > res['short']['total_return']:
            print(f"  LONG performing better: {res['long']['total_return']:.1f}% vs {res['short']['total_return']:.1f}%")
        else:
            print(f"  SHORT performing better: {res['short']['total_return']:.1f}% vs {res['long']['total_return']:.1f}%")

# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print()

summary_data = []
for res in all_results:
    row = {'Commodity': res['name']}

    if res['long']:
        row['Long Trades'] = res['long']['total']
        row['Long WR'] = f"{res['long']['win_rate']:.1f}%"
        row['Long Return'] = f"{res['long']['total_return']:.0f}%"
        row['Long PF'] = f"{res['long']['profit_factor']:.2f}"
    else:
        row['Long Trades'] = 0
        row['Long WR'] = 'N/A'
        row['Long Return'] = 'N/A'
        row['Long PF'] = 'N/A'

    if res['short']:
        row['Short Trades'] = res['short']['total']
        row['Short WR'] = f"{res['short']['win_rate']:.1f}%"
        row['Short Return'] = f"{res['short']['total_return']:.0f}%"
        row['Short PF'] = f"{res['short']['profit_factor']:.2f}"
    else:
        row['Short Trades'] = 0
        row['Short WR'] = 'N/A'
        row['Short Return'] = 'N/A'
        row['Short PF'] = 'N/A'

    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Overall stats
print("\n" + "="*80)
print("AGGREGATE STATS (ALL COMMODITIES)")
print("="*80)

total_long_trades = sum(res['long']['total'] for res in all_results if res['long'])
total_short_trades = sum(res['short']['total'] for res in all_results if res['short'])
total_long_wins = sum(res['long']['wins'] for res in all_results if res['long'])
total_short_wins = sum(res['short']['wins'] for res in all_results if res['short'])
total_long_return = sum(res['long']['total_return'] for res in all_results if res['long'])
total_short_return = sum(res['short']['total_return'] for res in all_results if res['short'])

print(f"\nOverall Long Performance:")
print(f"  Total Trades: {total_long_trades}")
print(f"  Win Rate: {total_long_wins/total_long_trades*100:.1f}%")
print(f"  Total Return: {total_long_return:.0f}%")
print(f"  Avg Return per Trade: {total_long_return/total_long_trades:.2f}%")

print(f"\nOverall Short Performance:")
print(f"  Total Trades: {total_short_trades}")
print(f"  Win Rate: {total_short_wins/total_short_trades*100:.1f}%")
print(f"  Total Return: {total_short_return:.0f}%")
print(f"  Avg Return per Trade: {total_short_return/total_short_trades:.2f}%")

print(f"\nLong vs Short:")
print(f"  Trade Ratio: {total_long_trades}L / {total_short_trades}S = {total_long_trades/total_short_trades:.2f}:1")
print(f"  Return Ratio: {total_long_return:.0f}% / {total_short_return:.0f}% = {total_long_return/total_short_return:.2f}:1")

if total_long_return > total_short_return:
    print(f"\n→ LONG trades dominate: {total_long_return/total_short_return:.1f}x more profitable")
else:
    print(f"\n→ SHORT trades dominate: {total_short_return/total_long_return:.1f}x more profitable")

print("="*80)
