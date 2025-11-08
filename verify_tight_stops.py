"""
Verify that tight stops (0.5x/1.5x) work on training data too
Not just curve-fitted to 2025

Test on:
1. Pre-COVID training (2016-2019)
2. 2025 validation
3. Full historical (2015-2025) to see consistency
"""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'strategies')

import pandas as pd
from src.data_fetcher import FMPDataFetcher
from src.backtester import Backtester
from strategies.volume_divergence import VolumeDivergence


def test_stops_on_periods():
    """Test different stop/target combos across time periods"""
    print('='*70)
    print('VERIFY TIGHT STOPS ACROSS TIME PERIODS')
    print('='*70)

    # Load data
    fetcher = FMPDataFetcher()
    data = fetcher.load_data('EURUSD_1day.csv')
    data['date_dt'] = pd.to_datetime(data['date'])

    # Define periods
    periods = {
        'Pre-COVID (2016-2019)': data[(data['date_dt'] >= '2016-01-01') & (data['date_dt'] < '2020-01-01')],
        'COVID Era (2020-2021)': data[(data['date_dt'] >= '2020-01-01') & (data['date_dt'] < '2022-01-01')],
        'Recent (2022-2024)': data[(data['date_dt'] >= '2022-01-01') & (data['date_dt'] < '2025-01-01')],
        '2025': data[data['date_dt'] >= '2025-01-01'],
        'All Data (2015-2025)': data
    }

    # Test configurations
    configs = [
        ('Baseline 1.0x/2.0x', 1.0, 2.0),
        ('Tight 0.5x/1.5x', 0.5, 1.5),
        ('Moderate 0.75x/1.5x', 0.75, 1.5),
        ('Moderate 0.75x/2.0x', 0.75, 2.0),
    ]

    results = []

    for period_name, period_data in periods.items():
        if len(period_data) == 0:
            continue

        print(f'\n{"="*70}')
        print(f'{period_name} ({len(period_data)} bars)')
        print('='*70)

        for config_name, stop, target in configs:
            strategy = VolumeDivergence(15, 25, 14, stop, target)
            backtester = Backtester(10000, 0.02, 0.0001)
            result = backtester.run(period_data, strategy)

            results.append({
                'period': period_name,
                'config': config_name,
                'stop': stop,
                'target': target,
                'return': result.metrics['total_return_pct'],
                'sharpe': result.metrics['sharpe_ratio'],
                'win_rate': result.metrics['win_rate'],
                'trades': result.metrics['total_trades'],
                'max_dd': result.metrics['max_drawdown']
            })

            print(f'\n{config_name}:')
            print(f'  Return: {result.metrics["total_return_pct"]:.2f}%')
            print(f'  Sharpe: {result.metrics["sharpe_ratio"]:.2f}')
            print(f'  Win Rate: {result.metrics["win_rate"]:.1f}%')
            print(f'  Trades: {result.metrics["total_trades"]}')
            print(f'  Max DD: {result.metrics["max_drawdown"]:.2f}%')

    # Summary comparison
    print(f'\n{"="*70}')
    print('SUMMARY: TIGHT STOPS vs BASELINE')
    print('='*70)

    results_df = pd.DataFrame(results)

    for period_name in periods.keys():
        period_results = results_df[results_df['period'] == period_name]
        if len(period_results) == 0:
            continue

        baseline = period_results[period_results['config'] == 'Baseline 1.0x/2.0x'].iloc[0]
        tight = period_results[period_results['config'] == 'Tight 0.5x/1.5x'].iloc[0]

        print(f'\n{period_name}:')
        print(f'  Baseline: {baseline["return"]:.2f}% return, {baseline["sharpe"]:.2f} Sharpe, {baseline["win_rate"]:.1f}% WR')
        print(f'  Tight:    {tight["return"]:.2f}% return, {tight["sharpe"]:.2f} Sharpe, {tight["win_rate"]:.1f}% WR')
        print(f'  Improvement: {tight["return"] - baseline["return"]:+.2f}% return, {tight["sharpe"] - baseline["sharpe"]:+.2f} Sharpe')

    # Check consistency
    print(f'\n{"="*70}')
    print('CONSISTENCY CHECK')
    print('='*70)

    tight_results = results_df[results_df['config'] == 'Tight 0.5x/1.5x']
    baseline_results = results_df[results_df['config'] == 'Baseline 1.0x/2.0x']

    tight_wins = sum(1 for _, row in tight_results.iterrows()
                     if row['return'] > baseline_results[baseline_results['period'] == row['period']].iloc[0]['return'])

    print(f'\nTight stops outperformed baseline in {tight_wins}/{len(tight_results)} periods')

    if tight_wins >= len(tight_results) * 0.8:
        print('CONSISTENT! Tight stops work across time periods.')
    elif tight_wins >= len(tight_results) * 0.6:
        print('MODERATE. Tight stops work in most periods.')
    else:
        print('INCONSISTENT. Tight stops may be curve-fitted to 2025.')

    # Risk-adjusted comparison
    print(f'\n{"="*70}')
    print('RISK-ADJUSTED COMPARISON (Sharpe Ratio)')
    print('='*70)

    tight_sharpes = tight_results['sharpe'].tolist()
    baseline_sharpes = baseline_results['sharpe'].tolist()

    print(f'\nTight 0.5x/1.5x Sharpe by period:')
    for period, sharpe in zip(tight_results['period'], tight_sharpes):
        print(f'  {period}: {sharpe:.2f}')

    print(f'\nBaseline 1.0x/2.0x Sharpe by period:')
    for period, sharpe in zip(baseline_results['period'], baseline_sharpes):
        print(f'  {period}: {sharpe:.2f}')

    avg_tight_sharpe = sum(tight_sharpes) / len(tight_sharpes)
    avg_baseline_sharpe = sum(baseline_sharpes) / len(baseline_sharpes)

    print(f'\nAverage Sharpe:')
    print(f'  Tight: {avg_tight_sharpe:.2f}')
    print(f'  Baseline: {avg_baseline_sharpe:.2f}')
    print(f'  Improvement: {avg_tight_sharpe - avg_baseline_sharpe:+.2f}')

    print(f'\n{"="*70}')
    print('RECOMMENDATION')
    print('='*70)

    if tight_wins >= len(tight_results) * 0.8 and avg_tight_sharpe > avg_baseline_sharpe:
        print('STRONG RECOMMENDATION: Switch to 0.5x/1.5x stops!')
        print('Benefits:')
        print('  - Higher returns across time periods')
        print('  - Better risk-adjusted performance')
        print('  - More consistent wins')
        print('  - Tighter risk control')
    elif tight_wins >= len(tight_results) * 0.6:
        print('MODERATE RECOMMENDATION: Consider 0.5x/1.5x stops')
        print('Works well in most periods but not all.')
    else:
        print('CAUTION: Stick with baseline 1.0x/2.0x')
        print('Tight stops may be overfitted to specific market conditions.')

    return results_df


if __name__ == "__main__":
    results = test_stops_on_periods()
