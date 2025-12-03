"""
Recalculate backtest results with REAL OANDA spreads
"""
import pandas as pd
import numpy as np
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Realistic OANDA spreads at 9 AM EST (in percentage)
REAL_SPREADS = {
    # Forex - tight spreads
    'EURUSD': 0.0073,
    'GBPUSD': 0.0092,
    'AUDUSD': 0.0154,
    'USDJPY': 0.0067,

    # Metals - wider spreads
    'XAUUSD': 0.0125,  # Gold $0.25 on $2000
    'XAGUSD': 0.0120,  # Silver $0.003 on $25
    'XPTUSD': 0.2500,  # Platinum $3 on $1200
    'XPDUSD': 0.3000,  # Palladium $3 on $1000
    'XCUUSD': 0.0375,  # Copper $0.0015 on $4

    # Commodities/Indices
    'SUGARUSD': 0.2000,  # Sugar
    'SPX500USD': 0.0111,  # S&P 500
    'DE30EUR': 0.0125,  # DAX
    'WTICOUSD': 0.0400,  # WTI Oil
    'BCOUSD': 0.0625,  # Brent Oil
}

# Current backtest uses 0.02% transaction cost
# We need to ADD the real spread cost
CURRENT_TRANSACTION_COST = 0.02

print('='*100)
print('RECALCULATING BACKTEST RESULTS WITH REAL SPREADS')
print('='*100)
print()

results = []

for asset, real_spread_pct in REAL_SPREADS.items():
    try:
        # Load backtest trades
        df = pd.read_csv(f'{asset}_backtest_trades_750days.csv')

        # Current net returns already include 0.02% transaction cost
        # We need to subtract the ADDITIONAL spread cost
        additional_spread_cost = real_spread_pct  # Applied on both entry and exit

        # Adjust all returns for real spreads
        # Spread hits you twice: entering and exiting
        # So total spread cost = 2 * spread_pct
        total_spread_impact = additional_spread_cost

        # Recalculate net returns
        df['adjusted_return_pct'] = df['net_return_pct'] - total_spread_impact
        df['adjusted_outcome'] = df['adjusted_return_pct'].apply(lambda x: 'WIN' if x > 0 else 'LOSS')

        # Calculate new statistics
        wins = df[df['adjusted_outcome'] == 'WIN']
        losses = df[df['adjusted_outcome'] == 'LOSS']

        old_total_return = df['net_return_pct'].sum()
        new_total_return = df['adjusted_return_pct'].sum()
        old_annual_return = old_total_return / 2.98
        new_annual_return = new_total_return / 2.98

        results.append({
            'asset': asset,
            'spread': real_spread_pct,
            'old_annual': old_annual_return,
            'new_annual': new_annual_return,
            'annual_diff': new_annual_return - old_annual_return,
            'old_win_rate': len(df[df['outcome'] == 'WIN']) / len(df) * 100,
            'new_win_rate': len(wins) / len(df) * 100,
            'old_avg_win': df[df['outcome'] == 'WIN']['net_return_pct'].mean(),
            'new_avg_win': wins['net_return_pct'].mean() if len(wins) > 0 else 0,
            'old_avg_loss': df[df['outcome'] == 'LOSS']['net_return_pct'].mean(),
            'new_avg_loss': losses['net_return_pct'].mean() if len(losses) > 0 else 0,
            'trades': len(df)
        })

    except Exception as e:
        print(f"Error processing {asset}: {e}")

results_df = pd.DataFrame(results)

print('PART 1: IMPACT OF REAL SPREADS ON ANNUAL RETURNS')
print('-'*100)
print()
print(f"{'Asset':<12} {'Spread':<10} {'Old Annual':<12} {'New Annual':<12} {'Impact':<12}")
print('-'*100)

for _, row in results_df.iterrows():
    impact_str = f"{row['annual_diff']:+.1f}%"
    print(f"{row['asset']:<12} {row['spread']:>8.2f}%  {row['old_annual']:>10.1f}%  {row['new_annual']:>10.1f}%  {impact_str:>10}")

print()
print('='*100)
print('PART 2: WIN RATE CHANGES')
print('-'*100)
print()
print(f"{'Asset':<12} {'Old WR':<10} {'New WR':<10} {'Change':<10}")
print('-'*100)

for _, row in results_df.iterrows():
    change = row['new_win_rate'] - row['old_win_rate']
    print(f"{row['asset']:<12} {row['old_win_rate']:>8.1f}%  {row['new_win_rate']:>8.1f}%  {change:+>8.1f}%")

print()
print('='*100)
print('PART 3: PORTFOLIO IMPACT')
print('-'*100)
print()

# Calculate portfolio-level impacts (10% allocation per asset, so sum and divide by 10)
old_portfolio_annual = results_df['old_annual'].sum() / 10
new_portfolio_annual = results_df['new_annual'].sum() / 10

print(f"OLD Portfolio Annual Return (1x): {old_portfolio_annual:.1f}%")
print(f"NEW Portfolio Annual Return (1x): {new_portfolio_annual:.1f}%")
print(f"Impact: {new_portfolio_annual - old_portfolio_annual:+.1f}%")
print()
print(f"OLD Portfolio Annual Return (2x): {old_portfolio_annual * 2:.1f}%")
print(f"NEW Portfolio Annual Return (2x): {new_portfolio_annual * 2:.1f}%")
print(f"Impact: {(new_portfolio_annual - old_portfolio_annual) * 2:+.1f}%")
print()

# Show which assets are hurt most
print('='*100)
print('ASSETS MOST AFFECTED BY REAL SPREADS')
print('-'*100)
print()

top_losers = results_df.nlargest(5, 'spread')[['asset', 'spread', 'old_annual', 'new_annual', 'annual_diff']]
for _, row in top_losers.iterrows():
    print(f"{row['asset']}:")
    print(f"  Spread: {row['spread']:.2f}%")
    print(f"  {row['old_annual']:.1f}% → {row['new_annual']:.1f}% ({row['annual_diff']:+.1f}%)")

    if row['new_annual'] < 0:
        print(f"  ⚠️  TURNS NEGATIVE - not profitable with real spreads!")
    elif row['new_annual'] < row['old_annual'] * 0.5:
        print(f"  ⚠️  Loses >50% of returns to spreads")
    print()

print('='*100)
print('RECOMMENDATION')
print('='*100)
print()

# Check if any assets turn unprofitable
unprofitable = results_df[results_df['new_annual'] < 0]

if len(unprofitable) > 0:
    print('⚠️  WARNING: Some assets become UNPROFITABLE with real spreads!')
    print()
    print('Assets to REMOVE from portfolio:')
    for _, row in unprofitable.iterrows():
        print(f"  • {row['asset']}: {row['old_annual']:.1f}% → {row['new_annual']:.1f}%")
    print()

    # Recalculate portfolio without unprofitable assets
    profitable = results_df[results_df['new_annual'] > 0]
    new_portfolio = profitable['new_annual'].sum() / len(profitable) * len(profitable)

    print(f"Revised Portfolio ({len(profitable)} assets):")
    print(f"  Annual Return (1x): {new_portfolio:.1f}%")
    print(f"  Annual Return (2x): {new_portfolio * 2:.1f}%")
    print()

    print('Suggested actions:')
    print('  1. Remove unprofitable assets from trading')
    print('  2. Or use wider stops for high-spread assets')
    print('  3. Or trade only during tightest spread times')
else:
    print('✓ All assets remain profitable with real spreads!')
    print()
    print(f"Portfolio still delivers {new_portfolio_annual:.1f}% annual (1x) / {new_portfolio_annual*2:.1f}% (2x)")
    print()
    print('The spread impact is manageable. Your strategy is robust!')

print()
print('='*100)
