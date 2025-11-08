"""Test volatility-based adaptive strategy"""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'strategies')

import pandas as pd
from src.data_fetcher import FMPDataFetcher
from src.backtester import Backtester
from strategies.volume_divergence import VolumeDivergence
from strategies.volatility_adaptive import VolatilityAdaptive


def test_vol_adaptive():
    """Test volatility adaptive vs baseline"""
    print('='*70)
    print('VOLATILITY ADAPTIVE TEST')
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

        # Vol Adaptive
        vol_adaptive = VolatilityAdaptive(15, 25, 14, vol_lookback=20, vol_historical=100)
        bt2 = Backtester(10000, 0.02, 0.0001)
        r2 = bt2.run(period_data, vol_adaptive)

        imp = r2.metrics["total_return_pct"] - r1.metrics["total_return_pct"]

        print(f'\nBaseline:      {r1.metrics["total_return_pct"]:7.2f}% return, {r1.metrics["win_rate"]:4.1f}% WR, {r1.metrics["total_trades"]:3} trades, Sharpe {r1.metrics["sharpe_ratio"]:.2f}')
        print(f'Vol Adaptive:  {r2.metrics["total_return_pct"]:7.2f}% return, {r2.metrics["win_rate"]:4.1f}% WR, {r2.metrics["total_trades"]:3} trades, Sharpe {r2.metrics["sharpe_ratio"]:.2f}')
        print(f'Improvement:   {imp:+7.2f}%')
        print(f'\nRegime: {vol_adaptive.get_stats()}')

        results.append({
            'period': period_name,
            'baseline': r1.metrics["total_return_pct"],
            'adaptive': r2.metrics["total_return_pct"],
            'improvement': imp
        })

    # Summary
    print(f'\n{"="*70}')
    print('SUMMARY')
    print('='*70)

    for r in results:
        symbol = '[+]' if r['improvement'] > 0 else '[-]'
        print(f'{symbol} {r["period"]:25} {r["improvement"]:+7.2f}%  (Baseline: {r["baseline"]:6.2f}% -> Adaptive: {r["adaptive"]:6.2f}%)')

    wins = sum(1 for r in results if r['improvement'] > 0)
    print(f'\nAdaptive outperformed in {wins}/{len(results)} periods')

    # Key question: Did it capture 2025 gains AND survive 2020-2024?
    y2025 = [r for r in results if r['period'] == '2025'][0]
    covid = [r for r in results if r['period'] == 'COVID (2020-2021)'][0]
    recent = [r for r in results if r['period'] == 'Recent (2022-2024)'][0]

    print(f'\n{"="*70}')
    print('KEY GOALS')
    print('='*70)

    print(f'\n1. CAPTURE 2025 GAINS (target: close to 40%):')
    print(f'   Baseline: {y2025["baseline"]:.2f}%')
    print(f'   Adaptive: {y2025["adaptive"]:.2f}%')
    if y2025['adaptive'] > 20:
        print(f'   [+] SUCCESS - Captured significant gains!')
    elif y2025['adaptive'] > y2025['baseline']:
        print(f'   [~] Improved but not close to 40%')
    else:
        print(f'   [-] FAILED - Did not improve')

    print(f'\n2. SURVIVE VOLATILE PERIODS (stay positive):')
    print(f'   COVID: Baseline {covid["baseline"]:.2f}% -> Adaptive {covid["adaptive"]:.2f}%')
    print(f'   Recent: Baseline {recent["baseline"]:.2f}% -> Adaptive {recent["adaptive"]:.2f}%')
    if covid['adaptive'] > 0 and recent['adaptive'] > 0:
        print(f'   [+] SUCCESS - Stayed positive in both')
    elif covid['adaptive'] > 0 or recent['adaptive'] > 0:
        print(f'   [~] PARTIAL - Positive in one period')
    else:
        print(f'   [-] FAILED - Lost money in volatile periods')

    print(f'\n{"="*70}')
    print('FINAL VERDICT')
    print('='*70)

    if y2025['adaptive'] > 20 and covid['adaptive'] > 0 and recent['adaptive'] > 0:
        print('SUCCESS! Vol-adaptive strategy:')
        print('  - Captures gains in calm markets (2025)')
        print('  - Survives volatile markets (2020-2024)')
        print('  -> RECOMMEND using volatility-adaptive approach')
    elif y2025['adaptive'] > y2025['baseline'] and covid['adaptive'] > covid['baseline']:
        print('PARTIAL SUCCESS:')
        print('  - Improves on baseline in key periods')
        print('  -> Consider using, but monitor closely')
    else:
        print('Did not meet goals. Baseline may be better.')
        print('Adaptation rules need further refinement.')


if __name__ == "__main__":
    test_vol_adaptive()
