"""Test simple adaptive strategy"""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'strategies')

import pandas as pd
from src.data_fetcher import FMPDataFetcher
from src.backtester import Backtester
from strategies.volume_divergence import VolumeDivergence
from strategies.simple_adaptive import SimpleAdaptive


def test_simple_adaptive():
    """Test simple adaptive vs baseline"""
    print('='*70)
    print('SIMPLE ADAPTIVE TEST')
    print('='*70)

    fetcher = FMPDataFetcher()
    data = fetcher.load_data('EURUSD_1day.csv')
    data['date_dt'] = pd.to_datetime(data['date'])

    periods = {
        'Pre-COVID (2016-2019)': data[(data['date_dt'] >= '2016-01-01') & (data['date_dt'] < '2020-01-01')],
        'COVID (2020-2021)': data[(data['date_dt'] >= '2020-01-01') & (data['date_dt'] < '2022-01-01')],
        'Recent (2022-2024)': data[(data['date_dt'] >= '2022-01-01') & (data['date_dt'] < '2025-01-01')],
        '2025': data[data['date_dt'] >= '2025-01-01']
    }

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

        # Adaptive
        adaptive = SimpleAdaptive(15, 25, 14)
        bt2 = Backtester(10000, 0.02, 0.0001)
        r2 = bt2.run(period_data, adaptive)

        print(f'\nBaseline:  {r1.metrics["total_return_pct"]:.2f}% return, {r1.metrics["win_rate"]:.1f}% WR, {r1.metrics["total_trades"]} trades')
        print(f'Adaptive:  {r2.metrics["total_return_pct"]:.2f}% return, {r2.metrics["win_rate"]:.1f}% WR, {r2.metrics["total_trades"]} trades')
        print(f'Improvement: {r2.metrics["total_return_pct"] - r1.metrics["total_return_pct"]:+.2f}%')

        print(f'\nAdaptive stats: {adaptive.get_stats()}')

    print(f'\n{"="*70}')
    print('DONE')
    print('='*70)


if __name__ == "__main__":
    test_simple_adaptive()
