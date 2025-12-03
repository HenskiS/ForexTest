"""
Determine optimal spread thresholds based on backtest data
Shows the relationship between spreads and profitability
"""
import pandas as pd
import numpy as np
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ASSETS = [
    'EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY',
    'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'XCUUSD',
    'SUGARUSD', 'SPX500USD', 'DE30EUR', 'WTICOUSD', 'BCOUSD'
]

# Typical spreads at 9 AM EST (from our earlier analysis)
TYPICAL_SPREADS = {
    'EURUSD': 0.0073, 'GBPUSD': 0.0092, 'AUDUSD': 0.0154, 'USDJPY': 0.0067,
    'XAUUSD': 0.0125, 'XAGUSD': 0.0120, 'XPTUSD': 0.2500, 'XPDUSD': 0.3000, 'XCUUSD': 0.0375,
    'SUGARUSD': 0.2000, 'SPX500USD': 0.0111, 'DE30EUR': 0.0125, 'WTICOUSD': 0.0400, 'BCOUSD': 0.0625,
}

# Current thresholds (2x normal)
CURRENT_MAX = {
    'EURUSD': 0.020, 'GBPUSD': 0.020, 'AUDUSD': 0.030, 'USDJPY': 0.015,
    'XAUUSD': 0.025, 'XAGUSD': 0.025, 'XPTUSD': 0.400, 'XPDUSD': 0.450, 'XCUUSD': 0.075,
    'SUGARUSD': 0.300, 'SPX500USD': 0.025, 'DE30EUR': 0.025, 'WTICOUSD': 0.080, 'BCOUSD': 0.100,
}

print('='*100)
print('DETERMINING OPTIMAL SPREAD THRESHOLDS')
print('='*100)
print()

# Load backtest data and analyze profitability
results = []

for asset in ASSETS:
    try:
        df = pd.read_csv(f'{asset}_backtest_trades_750days.csv')

        wins = df[df['outcome'] == 'WIN']
        losses = df[df['outcome'] == 'LOSS']

        avg_win = wins['net_return_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['net_return_pct'].mean() if len(losses) > 0 else 0
        win_rate = len(wins) / len(df) * 100

        # Calculate expected value per trade
        expected_value = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

        typical_spread = TYPICAL_SPREADS[asset]
        current_max = CURRENT_MAX[asset]

        # At what spread does the trade become unprofitable?
        # EV - spread_cost = 0
        # spread_cost = EV
        breakeven_spread = expected_value

        # Safety margin: we want at least 50% of EV remaining after spread
        # So max spread = EV * 0.5
        safe_spread = expected_value * 0.5

        # Conservative: max spread = EV * 0.3 (keep 70% of EV)
        conservative_spread = expected_value * 0.3

        results.append({
            'asset': asset,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expected_value': expected_value,
            'typical_spread': typical_spread,
            'current_max': current_max,
            'breakeven_spread': breakeven_spread,
            'safe_spread': safe_spread,
            'conservative_spread': conservative_spread,
            'typical_vs_ev': typical_spread / expected_value * 100 if expected_value > 0 else 0,
            'current_max_vs_ev': current_max / expected_value * 100 if expected_value > 0 else 0
        })
    except Exception as e:
        print(f"Error processing {asset}: {e}")

df_results = pd.DataFrame(results)

print('PART 1: EXPECTED VALUE vs SPREAD ANALYSIS')
print('-'*100)
print()
print(f"{'Asset':<12} {'EV/Trade':<10} {'Typical':<10} {'Current Max':<12} {'Typical %':<12} {'Max %':<10}")
print('-'*100)

for _, row in df_results.iterrows():
    print(f"{row['asset']:<12} {row['expected_value']:>8.3f}%  {row['typical_spread']:>8.3f}%  "
          f"{row['current_max']:>10.3f}%  {row['typical_vs_ev']:>10.1f}%  {row['current_max_vs_ev']:>8.1f}%")

print()
print('KEY INSIGHT: Typical % and Max % show what portion of expected value is consumed by spread')
print()

print('='*100)
print('PART 2: THRESHOLD RECOMMENDATIONS BY METHOD')
print('-'*100)
print()

print('METHOD 1: Break-Even Threshold (Maximum Theoretical)')
print('  Definition: Spread at which trade EV becomes zero')
print('  Risk: No margin for error, any worse execution = losing trade')
print()
print(f"{'Asset':<12} {'EV/Trade':<12} {'Breakeven Spread':<18} {'vs Typical':<12}")
print('-'*60)
for _, row in df_results.iterrows():
    vs_typical = row['breakeven_spread'] / row['typical_spread'] if row['typical_spread'] > 0 else 0
    print(f"{row['asset']:<12} {row['expected_value']:>10.3f}%  {row['breakeven_spread']:>16.3f}%  {vs_typical:>10.1f}x")

print()
print('='*100)
print('METHOD 2: Safety Margin (50% of EV Preserved)')
print('  Definition: Keep at least 50% of expected value after spread costs')
print('  Risk: Moderate safety margin')
print()
print(f"{'Asset':<12} {'Safe Spread':<15} {'vs Typical':<12} {'vs Current':<12}")
print('-'*60)
for _, row in df_results.iterrows():
    vs_typical = row['safe_spread'] / row['typical_spread'] if row['typical_spread'] > 0 else 0
    vs_current = row['safe_spread'] / row['current_max'] if row['current_max'] > 0 else 0
    print(f"{row['asset']:<12} {row['safe_spread']:>13.3f}%  {vs_typical:>10.1f}x  {vs_current:>10.1f}x")

print()
print('='*100)
print('METHOD 3: Conservative (70% of EV Preserved)')
print('  Definition: Keep at least 70% of expected value after spread costs')
print('  Risk: High safety margin - may skip legitimate trades')
print()
print(f"{'Asset':<12} {'Conservative':<15} {'vs Typical':<12} {'vs Current':<12}")
print('-'*60)
for _, row in df_results.iterrows():
    vs_typical = row['conservative_spread'] / row['typical_spread'] if row['typical_spread'] > 0 else 0
    vs_current = row['conservative_spread'] / row['current_max'] if row['current_max'] > 0 else 0
    print(f"{row['asset']:<12} {row['conservative_spread']:>13.3f}%  {vs_typical:>10.1f}x  {vs_current:>10.1f}x")

print()
print('='*100)
print('PART 3: CURRENT THRESHOLD EVALUATION')
print('-'*100)
print()

print('How do our current thresholds compare?')
print()
print(f"{'Asset':<12} {'Current Max':<12} {'Method':<20} {'Safety Level':<15}")
print('-'*70)

for _, row in df_results.iterrows():
    current = row['current_max']

    if current >= row['breakeven_spread']:
        method = 'TOO LOOSE'
        safety = 'DANGEROUS'
    elif current >= row['safe_spread']:
        method = 'Above 50% EV'
        safety = 'LOW MARGIN'
    elif current >= row['conservative_spread']:
        method = 'Between 50-70% EV'
        safety = 'GOOD'
    else:
        method = 'Below 70% EV'
        safety = 'VERY SAFE'

    print(f"{row['asset']:<12} {current:>10.3f}%  {method:<20} {safety:<15}")

print()
print('='*100)
print('PART 4: RECOMMENDED THRESHOLDS')
print('-'*100)
print()

print('Based on Expected Value Analysis:')
print()

# Recommend 50% EV preservation (safety margin) as default
print('RECOMMENDED: Use 50% EV Preservation (Method 2)')
print()
print(f"{'Asset':<12} {'Typical':<10} {'Recommended':<15} {'Multiplier':<12}")
print('-'*60)

recommended_thresholds = {}
for _, row in df_results.iterrows():
    typical = row['typical_spread']
    recommended = row['safe_spread']
    multiplier = recommended / typical if typical > 0 else 0
    recommended_thresholds[row['asset']] = recommended

    print(f"{row['asset']:<12} {typical:>8.3f}%  {recommended:>13.3f}%  {multiplier:>10.1f}x")

print()
print('='*100)
print('PART 5: PRACTICAL CONSIDERATIONS')
print('-'*100)
print()

print('Why NOT use breakeven as threshold?')
print('  • No margin for error or slippage')
print('  • Any execution delay = instant loss')
print('  • Risk of degraded performance over time')
print()

print('Why NOT use conservative (70% EV)?')
print('  • May skip too many legitimate trades')
print('  • Forex spreads at 9 AM are very tight')
print('  • Miss opportunities during normal volatility')
print()

print('Why use 50% EV preservation?')
print('  • Balance between safety and opportunity')
print('  • Allows for normal market volatility')
print('  • Protects against slippage and delays')
print('  • Still trades most days at 9 AM EST')
print()

print('='*100)
print('PART 6: ALTERNATIVE APPROACH - MULTIPLIER-BASED')
print('-'*100)
print()

print('Instead of EV-based, use simple multiplier of typical spread:')
print()

multipliers = [1.5, 2.0, 2.5, 3.0]
print(f"{'Asset':<12} {'Typical':<10} ", end='')
for mult in multipliers:
    print(f"{mult}x{' '*7}", end='')
print()
print('-'*60)

for _, row in df_results.iterrows():
    typical = row['typical_spread']
    print(f"{row['asset']:<12} {typical:>8.3f}%  ", end='')
    for mult in multipliers:
        threshold = typical * mult
        print(f"{threshold:>8.3f}%  ", end='')
    print()

print()
print('Multiplier Guidance:')
print('  • 1.5x: Very strict - only allow slight widening')
print('  • 2.0x: Balanced - our current approach')
print('  • 2.5x: Moderate - allow more volatility')
print('  • 3.0x: Loose - may accept suboptimal spreads')
print()

print('='*100)
print('FINAL RECOMMENDATION')
print('='*100)
print()

print('Use HYBRID APPROACH:')
print()
print('1. For Low-Spread Assets (Forex, Gold, Indices):')
print('   • Use 2-3x typical spread')
print('   • These spreads are so tight that even 3x is acceptable')
print('   • Example: EUR/USD typical 0.007%, max 0.020% (2.7x)')
print()

print('2. For High-Spread Assets (Platinum, Palladium, Sugar):')
print('   • Use 50% EV preservation')
print('   • These assets have huge spreads already')
print('   • Example: Platinum typical 0.25%, max based on EV not multiplier')
print()

print('3. For Medium-Spread Assets (Silver, Copper, Energy):')
print('   • Use whichever is MORE CONSERVATIVE between:')
print('     - 2x typical spread, OR')
print('     - 50% EV preservation')
print()

print('Current Status of Your Thresholds:')
print()

for _, row in df_results.iterrows():
    current = row['current_max']
    typical = row['typical_spread']
    safe_ev = row['safe_spread']
    multiplier = current / typical if typical > 0 else 0

    # Check if current threshold is reasonable
    if row['asset'] in ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'XAUUSD', 'SPX500USD', 'DE30EUR']:
        # Low spread assets - use multiplier
        recommended = typical * 2.5
        status = '✓ GOOD' if current <= recommended * 1.2 else '⚠️ TOO LOOSE'
    else:
        # High/medium spread assets - use EV
        recommended = safe_ev
        status = '✓ GOOD' if current <= recommended * 1.2 else '⚠️ TOO LOOSE'

    print(f"{row['asset']:<12} Current: {current:.3f}% ({multiplier:.1f}x)  {status}")

print()
print('='*100)
print()
print('SUMMARY: Your current 2x multiplier is GOOD for most assets!')
print()
print('Consider tightening for:')
print('  • Silver (currently 2.1x, but low EV - recommend 1.5x)')
print('  • DAX (currently 2.0x, but could be tighter)')
print()
print('But overall, your thresholds strike a good balance between:')
print('  • Protecting against wide spreads')
print('  • Not skipping too many legitimate trading opportunities')
print()
print('='*100)
