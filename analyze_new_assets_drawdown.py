"""
Analyze maximum drawdown for new assets (Sugar, S&P 500, DAX, WTI Oil) to determine safe leverage levels.
"""
import pandas as pd
import numpy as np
import sys
import io

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# New assets to analyze
NEW_ASSETS = ['SUGARUSD', 'SPX500USD', 'DE30EUR', 'WTICOUSD']
NAMES = {
    'SUGARUSD': 'Sugar',
    'SPX500USD': 'S&P 500',
    'DE30EUR': 'DAX',
    'WTICOUSD': 'WTI Oil'
}

def calculate_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve"""
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak * 100
    max_drawdown = drawdown.min()

    # Find the peak and trough for max drawdown
    max_dd_idx = drawdown.idxmin()
    peak_idx = equity_curve[:max_dd_idx].idxmax()

    return max_drawdown, peak_idx, max_dd_idx

def analyze_asset_drawdown(pair):
    """Analyze drawdown for a single asset"""
    try:
        trades_file = f'{pair}_backtest_trades_750days.csv'
        df = pd.read_csv(trades_file)

        # Parse dates
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df = df.sort_values('exit_date')

        # Calculate cumulative equity at 1x leverage (starting with $1000)
        df['cumulative_return'] = (1 + df['net_return_pct'] / 100).cumprod()
        df['equity_1x'] = 1000 * df['cumulative_return']

        # Calculate max drawdown at 1x
        max_dd_1x, peak_idx, trough_idx = calculate_drawdown(df['equity_1x'])

        # Calculate what happens at different leverage levels
        leverage_analysis = []
        for lev in [1, 2, 3, 4, 5, 6]:
            # Apply leverage to returns
            df[f'equity_{lev}x'] = 1000.0
            for i in range(len(df)):
                if i == 0:
                    df.loc[df.index[i], f'equity_{lev}x'] = 1000 * (1 + df.iloc[i]['net_return_pct'] / 100 * lev)
                else:
                    prev_equity = df.iloc[i-1][f'equity_{lev}x']
                    df.loc[df.index[i], f'equity_{lev}x'] = prev_equity * (1 + df.iloc[i]['net_return_pct'] / 100 * lev)

            # Calculate max drawdown at this leverage
            max_dd, _, _ = calculate_drawdown(df[f'equity_{lev}x'])
            final_equity = df[f'equity_{lev}x'].iloc[-1]
            total_return = (final_equity - 1000) / 1000 * 100

            # Check if account went bust (equity < $100)
            min_equity = df[f'equity_{lev}x'].min()
            went_bust = min_equity < 100

            leverage_analysis.append({
                'leverage': f'{lev}x',
                'max_dd': max_dd,
                'final_equity': final_equity,
                'total_return': total_return,
                'min_equity': min_equity,
                'bust': went_bust
            })

        # Calculate statistics
        final_equity = df['equity_1x'].iloc[-1]
        total_return = (final_equity - 1000) / 1000 * 100
        num_trades = len(df)
        years = (df['exit_date'].iloc[-1] - df['exit_date'].iloc[0]).days / 365.25
        annual_return = ((final_equity / 1000) ** (1 / years) - 1) * 100

        return {
            'pair': pair,
            'name': NAMES[pair],
            'max_dd_1x': max_dd_1x,
            'final_equity': final_equity,
            'total_return': total_return,
            'annual_return': annual_return,
            'num_trades': num_trades,
            'years': years,
            'peak_date': df.iloc[peak_idx]['exit_date'],
            'trough_date': df.iloc[trough_idx]['exit_date'],
            'leverage_analysis': leverage_analysis
        }

    except Exception as e:
        print(f"Error analyzing {pair}: {e}")
        import traceback
        traceback.print_exc()
        return None

print("="*80)
print("MAXIMUM DRAWDOWN ANALYSIS FOR NEW ASSETS")
print("="*80)
print()

all_results = []
for pair in NEW_ASSETS:
    print(f"Analyzing {NAMES[pair]}...")
    result = analyze_asset_drawdown(pair)
    if result:
        all_results.append(result)

# Summary table
print("\n" + "="*80)
print("DRAWDOWN SUMMARY AT 1X LEVERAGE")
print("="*80)
print()

summary_data = []
for res in all_results:
    summary_data.append({
        'Asset': res['name'],
        'Annual Return': f"{res['annual_return']:.1f}%",
        'Max DD (1x)': f"{res['max_dd_1x']:.1f}%",
        'Peak Date': res['peak_date'].strftime('%Y-%m-%d'),
        'Trough Date': res['trough_date'].strftime('%Y-%m-%d'),
        'Safe Leverage': 'TBD'
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Leverage analysis for each asset
print("\n" + "="*80)
print("LEVERAGE ANALYSIS - MAX DRAWDOWN AT DIFFERENT LEVERAGE LEVELS")
print("="*80)

for res in all_results:
    print(f"\n{res['name']} ({res['pair']}):")
    print("-" * 60)

    lev_data = []
    for lev_res in res['leverage_analysis']:
        status = "✗ BUST" if lev_res['bust'] else "✓ OK"
        lev_data.append({
            'Leverage': lev_res['leverage'],
            'Max DD': f"{lev_res['max_dd']:.1f}%",
            'Total Return': f"{lev_res['total_return']:.0f}%",
            'Final Equity': f"${lev_res['final_equity']:,.0f}",
            'Min Equity': f"${lev_res['min_equity']:,.0f}",
            'Status': status
        })

    lev_df = pd.DataFrame(lev_data)
    print(lev_df.to_string(index=False))

    # Determine safe leverage (max DD < 50%)
    safe_leverages = [l for l in res['leverage_analysis'] if not l['bust'] and l['max_dd'] > -50]
    if safe_leverages:
        max_safe_lev = len([l for l in res['leverage_analysis'] if not l['bust'] and l['max_dd'] > -50])
        print(f"\n  → Safe leverage: Up to {max_safe_lev}x (keeps max DD < 50%)")
    else:
        print(f"\n  → Safe leverage: 1x only (very high volatility)")

# Final recommendations
print("\n" + "="*80)
print("LEVERAGE RECOMMENDATIONS")
print("="*80)
print()
print("Based on max drawdown analysis:")
print()

for res in all_results:
    # Find max safe leverage (keeps DD < 50% and doesn't bust)
    safe_lev = 1
    for lev_res in res['leverage_analysis']:
        if not lev_res['bust'] and lev_res['max_dd'] > -50:
            safe_lev = int(lev_res['leverage'].replace('x', ''))

    annual_at_safe_lev = res['annual_return'] * safe_lev
    print(f"{res['name']:15} → {safe_lev}x leverage (Max DD: {res['leverage_analysis'][safe_lev-1]['max_dd']:.1f}%, "
          f"Annual: ~{annual_at_safe_lev:.0f}%)")

print()
print("Note: Conservative leverage keeps max drawdown under 50% and avoids account bust.")
print("Compare to commodities: 6x leverage with max DD 25-47%")
print("="*80)

# Comparison with existing commodities
print("\n" + "="*80)
print("COMPARISON: NEW ASSETS vs EXISTING COMMODITIES")
print("="*80)
print()

comparison_data = [
    {'Asset': 'Palladium', 'Type': 'Metal', 'Annual (1x)': '356%', 'Max DD (1x)': '-4.8%', 'Safe Lev': '6x', 'Annual (Safe)': '~2135%'},
    {'Asset': 'Platinum', 'Type': 'Metal', 'Annual (1x)': '192%', 'Max DD (1x)': '-4.4%', 'Safe Lev': '6x', 'Annual (Safe)': '~1151%'},
    {'Asset': 'Copper', 'Type': 'Metal', 'Annual (1x)': '147%', 'Max DD (1x)': '-6.2%', 'Safe Lev': '6x', 'Annual (Safe)': '~884%'},
    {'Asset': 'Silver', 'Type': 'Metal', 'Annual (1x)': '138%', 'Max DD (1x)': '-9.1%', 'Safe Lev': '6x', 'Annual (Safe)': '~830%'},
    {'Asset': 'Gold', 'Type': 'Metal', 'Annual (1x)': '120%', 'Max DD (1x)': '-4.6%', 'Safe Lev': '6x', 'Annual (Safe)': '~719%'},
]

for res in all_results:
    safe_lev = 1
    for lev_res in res['leverage_analysis']:
        if not lev_res['bust'] and lev_res['max_dd'] > -50:
            safe_lev = int(lev_res['leverage'].replace('x', ''))

    comparison_data.append({
        'Asset': res['name'],
        'Type': 'New',
        'Annual (1x)': f"{res['annual_return']:.0f}%",
        'Max DD (1x)': f"{res['max_dd_1x']:.1f}%",
        'Safe Lev': f"{safe_lev}x",
        'Annual (Safe)': f"~{res['annual_return'] * safe_lev:.0f}%"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))
print()
print("="*80)
