"""
Analyze the complete 10-asset portfolio performance
Shows unleveraged stats for all assets that will be traded
"""
import pandas as pd
import numpy as np
import sys
import io

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 10-Asset Portfolio
ASSETS = [
    # Forex (4)
    'EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY',
    # Metals (5)
    'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'XCUUSD',
    # Commodities/Indices (5)
    'SUGARUSD', 'SPX500USD', 'DE30EUR', 'WTICOUSD', 'BCOUSD'
]

ASSET_NAMES = {
    'EURUSD': 'EUR/USD',
    'GBPUSD': 'GBP/USD',
    'AUDUSD': 'AUD/USD',
    'USDJPY': 'USD/JPY',
    'XAUUSD': 'Gold',
    'XAGUSD': 'Silver',
    'XPTUSD': 'Platinum',
    'XPDUSD': 'Palladium',
    'XCUUSD': 'Copper',
    'SUGARUSD': 'Sugar',
    'SPX500USD': 'S&P 500',
    'DE30EUR': 'DAX',
    'WTICOUSD': 'WTI Oil',
    'BCOUSD': 'Brent Oil'
}

ASSET_CATEGORIES = {
    'EURUSD': 'Forex', 'GBPUSD': 'Forex', 'AUDUSD': 'Forex', 'USDJPY': 'Forex',
    'XAUUSD': 'Metals', 'XAGUSD': 'Metals', 'XPTUSD': 'Metals', 'XPDUSD': 'Metals', 'XCUUSD': 'Metals',
    'SUGARUSD': 'Commodities', 'SPX500USD': 'Indices', 'DE30EUR': 'Indices', 'WTICOUSD': 'Energy', 'BCOUSD': 'Energy'
}

def analyze_asset(asset):
    """Analyze a single asset from backtest data"""
    try:
        trades_file = f'{asset}_backtest_trades_750days.csv'
        df = pd.read_csv(trades_file)

        if len(df) == 0:
            return None

        # Calculate stats
        wins = df[df['outcome'] == 'WIN']
        losses = df[df['outcome'] == 'LOSS']

        # Max drawdown calculation
        df['cumulative_return'] = (1 + df['net_return_pct'] / 100).cumprod()
        df['peak'] = df['cumulative_return'].expanding().max()
        df['drawdown'] = (df['cumulative_return'] - df['peak']) / df['peak'] * 100
        max_dd = df['drawdown'].min()

        # Daily win rate (what % of days have at least 1 position)
        if 'entry_date' in df.columns:
            df['entry_day'] = pd.to_datetime(df['entry_date']).dt.date
            days_with_trades = df['entry_day'].nunique()
            total_days = 750
            trade_frequency = days_with_trades / total_days * 100
        else:
            trade_frequency = 0

        # Weekly analysis
        if 'entry_date' in df.columns:
            df['entry_week'] = pd.to_datetime(df['entry_date']).dt.to_period('W')
            weekly_returns = df.groupby('entry_week')['net_return_pct'].sum()
            weeks_with_wins = (weekly_returns > 0).sum()
            total_weeks = len(weekly_returns)
            weekly_win_pct = weeks_with_wins / total_weeks * 100 if total_weeks > 0 else 0
            avg_weekly_return = weekly_returns.mean()
        else:
            weekly_win_pct = 0
            avg_weekly_return = 0

        stats = {
            'asset': asset,
            'name': ASSET_NAMES[asset],
            'category': ASSET_CATEGORIES[asset],
            'total_trades': len(df),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(df) * 100,
            'total_return': df['net_return_pct'].sum(),
            'annual_return': df['net_return_pct'].sum() / 2.98,  # ~750 days = 2.98 years
            'avg_win': wins['net_return_pct'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['net_return_pct'].mean() if len(losses) > 0 else 0,
            'max_dd': max_dd,
            'profit_factor': abs(wins['net_return_pct'].sum() / losses['net_return_pct'].sum()) if len(losses) > 0 and losses['net_return_pct'].sum() != 0 else 0,
            'trade_frequency': trade_frequency,
            'weekly_win_pct': weekly_win_pct,
            'avg_weekly_return': avg_weekly_return
        }

        return stats

    except Exception as e:
        print(f"Error analyzing {asset}: {e}")
        return None

print("="*100)
print("10-ASSET PORTFOLIO ANALYSIS - 750 DAY BACKTEST (UNLEVERAGED)")
print("="*100)
print()

# Collect all stats
all_stats = []
for asset in ASSETS:
    stats = analyze_asset(asset)
    if stats:
        all_stats.append(stats)

if not all_stats:
    print("No backtest data found!")
    sys.exit(1)

# Create DataFrame
df_stats = pd.DataFrame(all_stats)

# Summary by category
print("PERFORMANCE BY ASSET CLASS")
print("-"*100)
for category in ['Forex', 'Metals', 'Indices', 'Energy', 'Commodities']:
    cat_df = df_stats[df_stats['category'] == category]
    if len(cat_df) > 0:
        print(f"\n{category}:")
        print(f"  Assets: {len(cat_df)}")
        print(f"  Avg Annual Return: {cat_df['annual_return'].mean():.1f}%")
        print(f"  Avg Win Rate: {cat_df['win_rate'].mean():.1f}%")
        print(f"  Avg Max DD: {cat_df['max_dd'].mean():.1f}%")

print("\n")
print("="*100)
print("INDIVIDUAL ASSET PERFORMANCE")
print("="*100)

for _, row in df_stats.iterrows():
    print(f"\n{row['name']} ({row['asset']})")
    print("-"*100)
    print(f"  Category: {row['category']}")
    print(f"  Total Trades: {row['total_trades']}")
    print(f"  Win Rate: {row['win_rate']:.1f}% ({row['wins']}/{row['total_trades']})")
    print(f"  Total Return (750 days): {row['total_return']:.1f}%")
    print(f"  Annual Return: {row['annual_return']:.1f}%")
    print(f"  Avg Win: {row['avg_win']:.2f}%")
    print(f"  Avg Loss: {row['avg_loss']:.2f}%")
    print(f"  Profit Factor: {row['profit_factor']:.2f}")
    print(f"  Max Drawdown: {row['max_dd']:.1f}%")
    print(f"  Trade Frequency: {row['trade_frequency']:.1f}% of days")
    print(f"  Weekly Win Rate: {row['weekly_win_pct']:.1f}%")
    print(f"  Avg Weekly Return: {row['avg_weekly_return']:.2f}%")

print("\n")
print("="*100)
print("PORTFOLIO SUMMARY TABLE")
print("="*100)

summary_df = df_stats[[
    'name', 'category', 'total_trades', 'win_rate',
    'annual_return', 'max_dd', 'profit_factor', 'weekly_win_pct'
]].copy()
summary_df.columns = ['Asset', 'Class', 'Trades', 'WR%', 'Annual%', 'MaxDD%', 'PF', 'Weekly WR%']
summary_df['WR%'] = summary_df['WR%'].apply(lambda x: f"{x:.1f}")
summary_df['Annual%'] = summary_df['Annual%'].apply(lambda x: f"{x:.0f}")
summary_df['MaxDD%'] = summary_df['MaxDD%'].apply(lambda x: f"{x:.1f}")
summary_df['PF'] = summary_df['PF'].apply(lambda x: f"{x:.2f}")
summary_df['Weekly WR%'] = summary_df['Weekly WR%'].apply(lambda x: f"{x:.0f}")

print()
print(summary_df.to_string(index=False))

print("\n")
print("="*100)
print("PORTFOLIO METRICS (EQUAL 10% ALLOCATION PER ASSET)")
print("="*100)

# Portfolio-level metrics
total_annual = df_stats['annual_return'].sum() / 10  # Equal 10% allocation
avg_win_rate = df_stats['win_rate'].mean()
avg_max_dd = df_stats['max_dd'].mean()
avg_profit_factor = df_stats['profit_factor'].mean()
total_trades = df_stats['total_trades'].sum()
avg_weekly_wr = df_stats['weekly_win_pct'].mean()

print(f"\nPortfolio Annual Return (1x leverage): {total_annual:.1f}%")
print(f"Portfolio Annual Return (2x leverage): {total_annual * 2:.1f}%")
print(f"Average Win Rate: {avg_win_rate:.1f}%")
print(f"Average Max Drawdown: {avg_max_dd:.1f}%")
print(f"Average Profit Factor: {avg_profit_factor:.2f}")
print(f"Total Trades (sum across all 750-day backtests): {total_trades}")
print(f"Avg Trades Per Asset Per Day: {total_trades / 750 / 10:.2f}")
print(f"Expected Daily Trades (10 assets): ~{len(ASSETS)} trades/day")
print(f"Average Weekly Win Rate: {avg_weekly_wr:.1f}%")

# Daily metrics
days_with_trades = []
for asset in ASSETS:
    try:
        df = pd.read_csv(f'{asset}_backtest_trades_750days.csv')
        if 'entry_date' in df.columns:
            df['entry_day'] = pd.to_datetime(df['entry_date']).dt.date
            days_with_trades.extend(df['entry_day'].unique())
    except:
        pass

unique_days = len(set(days_with_trades))
print(f"\nDays with at least 1 trade: {unique_days}/750 ({unique_days/750*100:.1f}%)")

print("\n")
print("="*100)
print("LEVERAGE PROJECTIONS")
print("="*100)
print(f"\n1x Leverage: {total_annual:.0f}% annual, Max DD ~{avg_max_dd:.1f}%")
print(f"2x Leverage: {total_annual * 2:.0f}% annual, Max DD ~{avg_max_dd * 2:.1f}%")
print(f"3x Leverage: {total_annual * 3:.0f}% annual, Max DD ~{avg_max_dd * 3:.1f}%")

print("\n")
print("="*100)
print("BEST PERFORMING ASSETS")
print("="*100)

top_5 = df_stats.nlargest(5, 'annual_return')[['name', 'annual_return', 'win_rate', 'max_dd']]
print()
for idx, row in top_5.iterrows():
    print(f"{row['name']:20} {row['annual_return']:6.0f}% annual | {row['win_rate']:4.1f}% WR | {row['max_dd']:5.1f}% DD")

print("\n")
print("="*100)
