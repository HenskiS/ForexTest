"""Test optimized strategy on all available currency pairs"""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'strategies')

from src.data_fetcher import FMPDataFetcher
from src.backtester import Backtester
from strategies.volume_divergence import VolumeDivergence
import pandas as pd

fetcher = FMPDataFetcher()
pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

# Test both optimized strategies
strategies = [
    ('Aggressive (1.0x stop, 2.0x target)', VolumeDivergence()),
    ('Conservative (2.5x stop, 2.0x target)',
     VolumeDivergence(stop_atr_multiplier=2.5, target_atr_multiplier=2.0)),
]

results_summary = []

for strat_name, strategy in strategies:
    print('\n' + '='*70)
    print(f'TESTING: {strat_name}')
    print('='*70)

    for pair in pairs:
        data = fetcher.load_data(f'{pair}_1day.csv')

        if data.empty:
            print(f'\n{pair}: No data available')
            continue

        backtester = Backtester(10000, 0.02, 0.0001)
        result = backtester.run(data, strategy)

        print(f'\n{pair}:')
        print(f'  Return: {result.metrics["total_return_pct"]:>7.2f}%')
        print(f'  Trades: {result.metrics["total_trades"]:>7}')
        print(f'  Win Rate: {result.metrics["win_rate"]:>5.1f}%')
        print(f'  Sharpe: {result.metrics["sharpe_ratio"]:>7.2f}')
        print(f'  Max DD: {result.metrics["max_drawdown"]:>7.2f}%')
        print(f'  Profit Factor: {result.metrics["profit_factor"]:>4.2f}')

        results_summary.append({
            'Strategy': strat_name,
            'Pair': pair,
            'Return': result.metrics["total_return_pct"],
            'Trades': result.metrics["total_trades"],
            'Win_Rate': result.metrics["win_rate"],
            'Sharpe': result.metrics["sharpe_ratio"],
            'Max_DD': result.metrics["max_drawdown"],
            'Profit_Factor': result.metrics["profit_factor"]
        })

# Summary comparison
print('\n' + '='*70)
print('SUMMARY: AGGRESSIVE STRATEGY ACROSS PAIRS')
print('='*70)
df = pd.DataFrame([r for r in results_summary if 'Aggressive' in r['Strategy']])
if not df.empty:
    print(df[['Pair', 'Return', 'Trades', 'Win_Rate', 'Sharpe', 'Max_DD']].to_string(index=False))
    print(f'\nAverage Return: {df["Return"].mean():.2f}%')
    print(f'Best Pair: {df.loc[df["Return"].idxmax(), "Pair"]} ({df["Return"].max():.2f}%)')
    print(f'Worst Pair: {df.loc[df["Return"].idxmin(), "Pair"]} ({df["Return"].min():.2f}%)')
    print(f'Consistent?: {"Yes - all positive" if (df["Return"] > 0).all() else "No - some losses"}')

print('\n' + '='*70)
print('SUMMARY: CONSERVATIVE STRATEGY ACROSS PAIRS')
print('='*70)
df_cons = pd.DataFrame([r for r in results_summary if 'Conservative' in r['Strategy']])
if not df_cons.empty:
    print(df_cons[['Pair', 'Return', 'Trades', 'Win_Rate', 'Sharpe', 'Max_DD']].to_string(index=False))
    print(f'\nAverage Return: {df_cons["Return"].mean():.2f}%')
    print(f'Best Pair: {df_cons.loc[df_cons["Return"].idxmax(), "Pair"]} ({df_cons["Return"].max():.2f}%)')
    print(f'Worst Pair: {df_cons.loc[df_cons["Return"].idxmin(), "Pair"]} ({df_cons["Return"].min():.2f}%)')
    print(f'Consistent?: {"Yes - all positive" if (df_cons["Return"] > 0).all() else "No - some losses"}')

print('\n' + '='*70)
print('RECOMMENDATION')
print('='*70)
if (df["Return"] > 0).all():
    print('✓ Aggressive strategy is ROBUST - profitable on all pairs!')
    print(f'  Average return: {df["Return"].mean():.2f}% over 10 years')
    print(f'  Range: {df["Return"].min():.2f}% to {df["Return"].max():.2f}%')
else:
    print('⚠ Aggressive strategy shows inconsistent results across pairs')
    print('  Consider testing on more data or using conservative approach')

if (df_cons["Return"] > 0).all():
    print('\n✓ Conservative strategy is ROBUST - profitable on all pairs!')
    print(f'  Average return: {df_cons["Return"].mean():.2f}% over 10 years')
    print(f'  Range: {df_cons["Return"].min():.2f}% to {df_cons["Return"].max():.2f}%')

print('\n' + '='*70)
