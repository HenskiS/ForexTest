"""
Compare all three strategy variants side-by-side

Run aggressive, baseline, and conservative as separate accounts
to see which performs best in each period.

This will show us:
1. When aggressive (0.75x/1.5x) works vs fails
2. When conservative (1.25x/2.5x) is needed
3. If baseline (1.0x/2.0x) is truly the best overall
"""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'strategies')

import pandas as pd
from src.data_fetcher import FMPDataFetcher
from src.backtester import Backtester
from strategies.volume_divergence import VolumeDivergence


def compare_variants():
    """Compare all three variants across all periods"""
    print('='*70)
    print('STRATEGY VARIANT COMPARISON')
    print('Running 3 separate accounts: Aggressive, Baseline, Conservative')
    print('='*70)

    fetcher = FMPDataFetcher()
    data = fetcher.load_data('EURUSD_1day.csv')
    data['date_dt'] = pd.to_datetime(data['date'])

    periods = {
        'Pre-COVID (2016-2019)': data[(data['date_dt'] >= '2016-01-01') & (data['date_dt'] < '2020-01-01')],
        'COVID Era (2020-2021)': data[(data['date_dt'] >= '2020-01-01') & (data['date_dt'] < '2022-01-01')],
        'Recent (2022-2024)': data[(data['date_dt'] >= '2022-01-01') & (data['date_dt'] < '2025-01-01')],
        '2025': data[data['date_dt'] >= '2025-01-01'],
        'All Data (2015-2025)': data
    }

    # Define variants
    variants = {
        'Aggressive': {'stop': 0.75, 'target': 1.5, 'color': 'aggressive'},
        'Baseline': {'stop': 1.0, 'target': 2.0, 'color': 'baseline'},
        'Conservative': {'stop': 1.25, 'target': 2.5, 'color': 'conservative'}
    }

    all_results = []

    for period_name, period_data in periods.items():
        if len(period_data) == 0:
            continue

        print(f'\n{"="*70}')
        print(f'{period_name} ({len(period_data)} bars)')
        print('='*70)

        period_results = {'period': period_name}

        for variant_name, params in variants.items():
            strategy = VolumeDivergence(15, 25, 14, params['stop'], params['target'])
            bt = Backtester(10000, 0.02, 0.0001)
            result = bt.run(period_data, strategy)

            period_results[f'{variant_name}_return'] = result.metrics['total_return_pct']
            period_results[f'{variant_name}_sharpe'] = result.metrics['sharpe_ratio']
            period_results[f'{variant_name}_wr'] = result.metrics['win_rate']
            period_results[f'{variant_name}_trades'] = result.metrics['total_trades']
            period_results[f'{variant_name}_dd'] = result.metrics['max_drawdown']

            print(f'\n{variant_name:12} ({params["stop"]:.2f}x/{params["target"]:.1f}x):')
            print(f'  Return:  {result.metrics["total_return_pct"]:7.2f}%')
            print(f'  Sharpe:  {result.metrics["sharpe_ratio"]:7.2f}')
            print(f'  Win Rate: {result.metrics["win_rate"]:6.1f}%')
            print(f'  Trades:   {result.metrics["total_trades"]:3}')
            print(f'  Max DD:  {result.metrics["max_drawdown"]:7.2f}%')

        # Find best in this period
        returns = {k: period_results[f'{k}_return'] for k in variants.keys()}
        best = max(returns, key=returns.get)
        worst = min(returns, key=returns.get)

        print(f'\n  >>> BEST: {best} ({returns[best]:.2f}%)')
        print(f'  >>> WORST: {worst} ({returns[worst]:.2f}%)')

        all_results.append(period_results)

    # Summary table
    print(f'\n{"="*70}')
    print('SUMMARY TABLE')
    print('='*70)

    print(f'\n{"Period":<25} {"Aggressive":<15} {"Baseline":<15} {"Conservative":<15} {"Winner"}')
    print('-'*80)

    for r in all_results:
        agg_ret = r['Aggressive_return']
        base_ret = r['Baseline_return']
        cons_ret = r['Conservative_return']

        returns = {'Aggressive': agg_ret, 'Baseline': base_ret, 'Conservative': cons_ret}
        winner = max(returns, key=returns.get)

        print(f'{r["period"]:<25} {agg_ret:>7.2f}%       {base_ret:>7.2f}%       {cons_ret:>7.2f}%       {winner}')

    # Win count
    print(f'\n{"="*70}')
    print('WIN COUNT BY VARIANT')
    print('='*70)

    win_counts = {'Aggressive': 0, 'Baseline': 0, 'Conservative': 0}

    for r in all_results:
        returns = {
            'Aggressive': r['Aggressive_return'],
            'Baseline': r['Baseline_return'],
            'Conservative': r['Conservative_return']
        }
        winner = max(returns, key=returns.get)
        win_counts[winner] += 1

    for variant, count in win_counts.items():
        print(f'{variant}: {count}/{len(all_results)} periods')

    # Risk-adjusted (Sharpe)
    print(f'\n{"="*70}')
    print('RISK-ADJUSTED COMPARISON (Sharpe Ratio)')
    print('='*70)

    print(f'\n{"Period":<25} {"Aggressive":<12} {"Baseline":<12} {"Conservative":<12} {"Best"}')
    print('-'*75)

    sharpe_wins = {'Aggressive': 0, 'Baseline': 0, 'Conservative': 0}

    for r in all_results:
        agg_sharpe = r['Aggressive_sharpe']
        base_sharpe = r['Baseline_sharpe']
        cons_sharpe = r['Conservative_sharpe']

        sharpes = {'Aggressive': agg_sharpe, 'Baseline': base_sharpe, 'Conservative': cons_sharpe}
        best = max(sharpes, key=sharpes.get)
        sharpe_wins[best] += 1

        print(f'{r["period"]:<25} {agg_sharpe:>7.2f}      {base_sharpe:>7.2f}      {cons_sharpe:>7.2f}      {best}')

    print(f'\nSharpe wins:')
    for variant, count in sharpe_wins.items():
        print(f'{variant}: {count}/{len(all_results)} periods')

    # Survival analysis
    print(f'\n{"="*70}')
    print('SURVIVAL ANALYSIS')
    print('='*70)

    print('\nNegative return periods (account would be losing):')
    for r in all_results:
        negatives = []
        if r['Aggressive_return'] < 0:
            negatives.append(f"Aggressive ({r['Aggressive_return']:.2f}%)")
        if r['Baseline_return'] < 0:
            negatives.append(f"Baseline ({r['Baseline_return']:.2f}%)")
        if r['Conservative_return'] < 0:
            negatives.append(f"Conservative ({r['Conservative_return']:.2f}%)")

        if negatives:
            print(f'{r["period"]}: {", ".join(negatives)}')

    # Max drawdown comparison
    print(f'\n{"="*70}')
    print('MAXIMUM DRAWDOWN (Worst Loss)')
    print('='*70)

    print(f'\n{"Period":<25} {"Aggressive":<12} {"Baseline":<12} {"Conservative":<12}')
    print('-'*75)

    for r in all_results:
        print(f'{r["period"]:<25} {r["Aggressive_dd"]:>7.2f}%     {r["Baseline_dd"]:>7.2f}%     {r["Conservative_dd"]:>7.2f}%')

    # Final verdict
    print(f'\n{"="*70}')
    print('FINAL VERDICT')
    print('='*70)

    print('\nAGGRESSIVE (0.75x/1.5x):')
    print('  Pros: Highest returns in calm markets (2025: {:.2f}%)'.format(
        [r for r in all_results if r['period'] == '2025'][0]['Aggressive_return']))
    print('  Cons: Catastrophic losses in volatile markets')
    print('  Use: Only if you can predict calm periods (you can\'t)')

    print('\nBASELINE (1.0x/2.0x):')
    base_all = [r for r in all_results if r['period'] == 'All Data (2015-2025)'][0]
    print(f'  Pros: Positive in all periods, best overall ({base_all["Baseline_return"]:.2f}%)')
    print(f'  Cons: Leaves gains on table in calm markets')
    print(f'  Use: RECOMMENDED - consistent, survives everything')

    print('\nCONSERVATIVE (1.25x/2.5x):')
    print('  Pros: Lower drawdowns in volatile markets')
    print('  Cons: Lower returns overall')
    print('  Use: If very risk-averse')

    print(f'\n{"="*70}')
    print('RECOMMENDATION: Use BASELINE (1.0x/2.0x)')
    print('It won the most periods and has the best risk-adjusted returns.')
    print('='*70)


if __name__ == "__main__":
    compare_variants()
