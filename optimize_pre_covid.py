"""
Optimize strategy on pre-COVID data (2016-2019), test on 2025

Goal: See if pre-COVID market conditions are more similar to 2025
      than the turbulent 2020-2024 period
"""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'strategies')

import pandas as pd
import numpy as np
from src.data_fetcher import FMPDataFetcher
from src.backtester import Backtester
from strategies.volume_divergence import VolumeDivergence


def split_data_pre_covid(data):
    """Split: 2016-2019 for optimization, 2025 for testing"""
    data['date_dt'] = pd.to_datetime(data['date'])

    train = data[(data['date_dt'] >= '2016-01-01') & (data['date_dt'] < '2020-01-01')].copy()
    test = data[data['date_dt'] >= '2025-01-01'].copy()

    return train, test


def optimize_on_pre_covid():
    """Optimize parameters on 2016-2019, test on 2025"""
    print('='*70)
    print('OPTIMIZE ON PRE-COVID DATA (2016-2019)')
    print('='*70)

    # Load data
    fetcher = FMPDataFetcher()
    data = fetcher.load_data('EURUSD_1day.csv')

    if data.empty:
        print("No data found!")
        return

    print(f'\nTotal data: {len(data)} bars ({data["date"].min()} to {data["date"].max()})')

    # Split data
    train_data, test_data = split_data_pre_covid(data)

    print(f'\nData splits:')
    print(f'  Train: {len(train_data)} bars ({train_data["date"].min()} to {train_data["date"].max()}) - pre-COVID optimization')
    print(f'  Test:  {len(test_data)} bars ({test_data["date"].min()} to {test_data["date"].max()}) - 2025 validation')

    # Parameter grid
    lookback_periods = [15, 20, 30]
    volume_periods = [20, 25, 30]
    atr_periods = [10, 14, 20]
    stop_multipliers = [1.0, 1.5, 2.0, 2.5]
    target_multipliers = [1.5, 2.0, 2.5, 3.0]

    total_combos = len(lookback_periods) * len(volume_periods) * len(atr_periods) * \
                   len(stop_multipliers) * len(target_multipliers)

    print(f'\nTesting {total_combos} parameter combinations on pre-COVID data...')
    print('='*70)

    results = []
    count = 0

    for lookback in lookback_periods:
        for volume in volume_periods:
            for atr in atr_periods:
                for stop in stop_multipliers:
                    for target in target_multipliers:
                        count += 1
                        if count % 50 == 0:
                            print(f'Progress: {count}/{total_combos}...')

                        strategy = VolumeDivergence(
                            lookback_period=lookback,
                            volume_period=volume,
                            atr_period=atr,
                            stop_atr_multiplier=stop,
                            target_atr_multiplier=target
                        )

                        backtester = Backtester(10000, 0.02, 0.0001)
                        result = backtester.run(train_data, strategy)

                        # Skip if too few trades
                        if result.metrics['total_trades'] < 10:
                            continue

                        results.append({
                            'lookback': lookback,
                            'volume': volume,
                            'atr': atr,
                            'stop': stop,
                            'target': target,
                            'return': result.metrics['total_return_pct'],
                            'sharpe': result.metrics['sharpe_ratio'],
                            'win_rate': result.metrics['win_rate'],
                            'trades': result.metrics['total_trades'],
                            'max_dd': result.metrics['max_drawdown']
                        })

    print(f'\nCompleted! Found {len(results)} valid parameter combinations.')

    # Sort by return
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return', ascending=False)

    print(f'\n{"="*70}')
    print('TOP 10 STRATEGIES (2016-2019 PRE-COVID OPTIMIZATION)')
    print('='*70)
    print(results_df.head(10).to_string(index=False))

    # Get best strategy
    best = results_df.iloc[0]

    print(f'\n{"="*70}')
    print('BEST STRATEGY ON PRE-COVID DATA (2016-2019)')
    print('='*70)
    print(f'Parameters:')
    print(f'  Lookback: {int(best["lookback"])}')
    print(f'  Volume period: {int(best["volume"])}')
    print(f'  ATR period: {int(best["atr"])}')
    print(f'  Stop: {best["stop"]:.1f}x ATR')
    print(f'  Target: {best["target"]:.1f}x ATR')
    print(f'\n2016-2019 Performance:')
    print(f'  Return: {best["return"]:.2f}%')
    print(f'  Sharpe: {best["sharpe"]:.2f}')
    print(f'  Win Rate: {best["win_rate"]:.1f}%')
    print(f'  Trades: {int(best["trades"])}')
    print(f'  Max DD: {best["max_dd"]:.2f}%')

    # Test best on 2025
    print(f'\n{"="*70}')
    print('TESTING PRE-COVID PARAMETERS ON 2025')
    print('='*70)

    best_strategy = VolumeDivergence(
        lookback_period=int(best["lookback"]),
        volume_period=int(best["volume"]),
        atr_period=int(best["atr"]),
        stop_atr_multiplier=best["stop"],
        target_atr_multiplier=best["target"]
    )

    backtester_test = Backtester(10000, 0.02, 0.0001)
    result_test = backtester_test.run(test_data, best_strategy)

    print(f'2025 Performance (Out-of-Sample):')
    print(f'  Return: {result_test.metrics["total_return_pct"]:.2f}%')
    print(f'  Sharpe: {result_test.metrics["sharpe_ratio"]:.2f}')
    print(f'  Win Rate: {result_test.metrics["win_rate"]:.1f}%')
    print(f'  Trades: {result_test.metrics["total_trades"]}')
    print(f'  Max DD: {result_test.metrics["max_drawdown"]:.2f}%')

    # Compare to other optimizations
    print(f'\n{"="*70}')
    print('COMPARISON: PRE-COVID vs POST-COVID vs OLD PARAMETERS')
    print('='*70)

    # Old params (2015-2021 optimization)
    old_strategy = VolumeDivergence(20, 25, 14, 1.0, 2.0)
    backtester_old = Backtester(10000, 0.02, 0.0001)
    result_old = backtester_old.run(test_data, old_strategy)

    # Recent params (2021-2024 optimization)
    recent_strategy = VolumeDivergence(15, 20, 14, 1.0, 2.5)
    backtester_recent = Backtester(10000, 0.02, 0.0001)
    result_recent = backtester_recent.run(test_data, recent_strategy)

    print(f'\nOLD PARAMS (2015-2021 optimization):')
    print(f'  Params: 20/25/14, 1.0x stop, 2.0x target')
    print(f'  2025 Return: {result_old.metrics["total_return_pct"]:.2f}%')
    print(f'  2025 Win Rate: {result_old.metrics["win_rate"]:.1f}%')
    print(f'  2025 Trades: {result_old.metrics["total_trades"]}')

    print(f'\nRECENT PARAMS (2021-2024 optimization):')
    print(f'  Params: 15/20/14, 1.0x stop, 2.5x target')
    print(f'  2025 Return: {result_recent.metrics["total_return_pct"]:.2f}%')
    print(f'  2025 Win Rate: {result_recent.metrics["win_rate"]:.1f}%')
    print(f'  2025 Trades: {result_recent.metrics["total_trades"]}')

    print(f'\nPRE-COVID PARAMS (2016-2019 optimization):')
    print(f'  Params: {int(best["lookback"])}/{int(best["volume"])}/{int(best["atr"])}, {best["stop"]:.1f}x stop, {best["target"]:.1f}x target')
    print(f'  2025 Return: {result_test.metrics["total_return_pct"]:.2f}%')
    print(f'  2025 Win Rate: {result_test.metrics["win_rate"]:.1f}%')
    print(f'  2025 Trades: {result_test.metrics["total_trades"]}')

    print(f'\n{"="*70}')
    print('RANKING (2025 PERFORMANCE)')
    print('='*70)

    ranking = [
        ('Pre-COVID (2016-2019)', result_test.metrics["total_return_pct"]),
        ('Old (2015-2021)', result_old.metrics["total_return_pct"]),
        ('Recent (2021-2024)', result_recent.metrics["total_return_pct"])
    ]
    ranking.sort(key=lambda x: x[1], reverse=True)

    for i, (name, ret) in enumerate(ranking, 1):
        print(f'{i}. {name}: {ret:.2f}%')

    print(f'\n{"="*70}')
    print('CONCLUSION')
    print('='*70)

    best_approach = ranking[0]
    if result_test.metrics["total_return_pct"] > 5:
        print(f'Pre-COVID parameters work! {result_test.metrics["total_return_pct"]:.2f}% return in 2025.')
        print('Hypothesis: Pre-COVID and 2025 markets share similar characteristics.')
        print('The 2020-2024 period was anomalous (COVID, rate changes, etc.)')
    elif best_approach[0].startswith('Pre-COVID'):
        print(f'Pre-COVID parameters are best, but still weak ({best_approach[1]:.2f}%).')
        print('Markets have structurally changed - no historical period helps.')
    else:
        print(f'{best_approach[0]} parameters work best on 2025.')
        print(f'Pre-COVID optimization did not help ({result_test.metrics["total_return_pct"]:.2f}%).')
        print('No period of optimization produces good 2025 results.')


if __name__ == "__main__":
    optimize_on_pre_covid()
