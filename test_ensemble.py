"""Test ensemble adaptive strategy"""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'strategies')

import pandas as pd
from src.data_fetcher import FMPDataFetcher
from src.backtester import Backtester
from strategies.volume_divergence import VolumeDivergence
from strategies.ensemble_adaptive import EnsembleAdaptive


def test_ensemble():
    """Test ensemble adaptive vs baseline"""
    print('='*70)
    print('ENSEMBLE ADAPTIVE TEST')
    print('='*70)

    fetcher = FMPDataFetcher()
    data = fetcher.load_data('EURUSD_1day.csv')
    data['date_dt'] = pd.to_datetime(data['date'])

    periods = {
        'Pre-COVID (2016-2019)': data[(data['date_dt'] >= '2016-01-01') & (data['date_dt'] < '2020-01-01')],
        'COVID (2020-2021)': data[(data['date_dt'] >= '2020-01-01') & (data['date_dt'] < '2022-01-01')],
        'Recent (2022-2024)': data[(data['date_dt'] >= '2022-01-01') & (data['date_dt'] < '2025-01-01')],
        '2025': data[data['date_dt'] >= '2025-01-01'],
        'All Data (2015-2025)': data
    }

    results = []

    for period_name, period_data in periods.items():
        if len(period_data) == 0:
            continue

        print(f'\n{"="*70}')
        print(f'{period_name}')
        print('='*70)

        # Baseline
        baseline = VolumeDivergence(15, 25, 14, 1.0, 2.0)
        bt1 = Backtester(10000, 0.02, 0.0001)
        r1 = bt1.run(period_data, baseline)

        # Ensemble
        ensemble = EnsembleAdaptive(15, 25, 14, eval_window=10, switch_threshold=5.0)
        bt2 = Backtester(10000, 0.02, 0.0001)
        r2 = bt2.run(period_data, ensemble)

        imp = r2.metrics["total_return_pct"] - r1.metrics["total_return_pct"]

        print(f'\nBaseline:  {r1.metrics["total_return_pct"]:7.2f}% return, {r1.metrics["win_rate"]:4.1f}% WR, {r1.metrics["total_trades"]:3} trades, Sharpe {r1.metrics["sharpe_ratio"]:.2f}')
        print(f'Ensemble:  {r2.metrics["total_return_pct"]:7.2f}% return, {r2.metrics["win_rate"]:4.1f}% WR, {r2.metrics["total_trades"]:3} trades, Sharpe {r2.metrics["sharpe_ratio"]:.2f}')
        print(f'Improvement: {imp:+7.2f}%')
        print(f'\nEnsemble stats: {ensemble.get_stats()}')

        results.append({
            'period': period_name,
            'baseline': r1.metrics["total_return_pct"],
            'ensemble': r2.metrics["total_return_pct"],
            'improvement': imp
        })

    # Summary
    print(f'\n{"="*70}')
    print('SUMMARY')
    print('='*70)

    for r in results:
        symbol = '[+]' if r['improvement'] > 0 else '[-]'
        print(f'{symbol} {r["period"]:25} {r["improvement"]:+7.2f}%  ({r["baseline"]:6.2f}% -> {r["ensemble"]:6.2f}%)')

    wins = sum(1 for r in results if r['improvement'] > 0)
    print(f'\nEnsemble outperformed in {wins}/{len(results)} periods')

    # Key metrics
    y2025 = [r for r in results if r['period'] == '2025'][0]
    all_data = [r for r in results if r['period'] == 'All Data (2015-2025)'][0]

    print(f'\n{"="*70}')
    print('KEY RESULTS')
    print('='*70)
    print(f'\n2025 Performance:')
    print(f'  Baseline: {y2025["baseline"]:.2f}%')
    print(f'  Ensemble: {y2025["ensemble"]:.2f}%')
    print(f'  Target: ~40% (tight stops benchmark)')

    print(f'\nOverall (2015-2025):')
    print(f'  Baseline: {all_data["baseline"]:.2f}%')
    print(f'  Ensemble: {all_data["ensemble"]:.2f}%')

    if y2025['ensemble'] > 20:
        print(f'\n[+] SUCCESS! Ensemble captured significant 2025 gains')
    elif y2025['ensemble'] > y2025['baseline']:
        print(f'\n[~] Ensemble improved 2025 but not dramatically')
    else:
        print(f'\n[-] Ensemble did not improve 2025 performance')


if __name__ == "__main__":
    test_ensemble()
