"""
Test Adaptive Volume Divergence Strategy

Compare:
1. Baseline (1.0x/2.0x always)
2. Adaptive (adjusts based on performance)

Test across all periods to ensure it:
- Captures more gains in 2025 (closer to 40%)
- Doesn't blow up in 2020-2024 (stay positive)
"""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'strategies')

import pandas as pd
from src.data_fetcher import FMPDataFetcher
from src.backtester import Backtester
from strategies.volume_divergence import VolumeDivergence
from strategies.adaptive_volume_divergence import AdaptiveVolumeDivergence


def test_adaptive_strategy():
    """Test adaptive strategy across time periods"""
    print('='*70)
    print('ADAPTIVE STRATEGY TEST')
    print('='*70)

    # Load data
    fetcher = FMPDataFetcher()
    data = fetcher.load_data('EURUSD_1day.csv')
    data['date_dt'] = pd.to_datetime(data['date'])

    # Define test periods
    periods = {
        'Pre-COVID (2016-2019)': data[(data['date_dt'] >= '2016-01-01') & (data['date_dt'] < '2020-01-01')],
        'COVID Era (2020-2021)': data[(data['date_dt'] >= '2020-01-01') & (data['date_dt'] < '2022-01-01')],
        'Recent (2022-2024)': data[(data['date_dt'] >= '2022-01-01') & (data['date_dt'] < '2025-01-01')],
        '2025': data[data['date_dt'] >= '2025-01-01'],
        'All Data (2015-2025)': data
    }

    results = []

    for period_name, period_data in periods.items():
        if len(period_data) == 0:
            continue

        print(f'\n{"="*70}')
        print(f'{period_name} ({len(period_data)} bars)')
        print('='*70)

        # Test baseline
        baseline = VolumeDivergence(15, 25, 14, 1.0, 2.0)
        bt_baseline = Backtester(10000, 0.02, 0.0001)
        result_baseline = bt_baseline.run(period_data, baseline)

        # Test adaptive
        adaptive = AdaptiveVolumeDivergence(
            lookback_period=15,
            volume_period=25,
            atr_period=14,
            base_stop=1.0,
            base_target=2.0,
            enable_streak_adaptation=True,
            enable_volatility_adaptation=True,
            win_streak_threshold=3,
            loss_streak_threshold=2,
            lookback_trades=10
        )
        bt_adaptive = Backtester(10000, 0.02, 0.0001)
        result_adaptive = bt_adaptive.run(period_data, adaptive)

        print(f'\nBaseline (1.0x/2.0x always):')
        print(f'  Return: {result_baseline.metrics["total_return_pct"]:.2f}%')
        print(f'  Sharpe: {result_baseline.metrics["sharpe_ratio"]:.2f}')
        print(f'  Win Rate: {result_baseline.metrics["win_rate"]:.1f}%')
        print(f'  Trades: {result_baseline.metrics["total_trades"]}')
        print(f'  Max DD: {result_baseline.metrics["max_drawdown"]:.2f}%')

        print(f'\nAdaptive (adjusts parameters):')
        print(f'  Return: {result_adaptive.metrics["total_return_pct"]:.2f}%')
        print(f'  Sharpe: {result_adaptive.metrics["sharpe_ratio"]:.2f}')
        print(f'  Win Rate: {result_adaptive.metrics["win_rate"]:.1f}%')
        print(f'  Trades: {result_adaptive.metrics["total_trades"]}')
        print(f'  Max DD: {result_adaptive.metrics["max_drawdown"]:.2f}%')

        print(f'\nImprovement:')
        improvement = result_adaptive.metrics["total_return_pct"] - result_baseline.metrics["total_return_pct"]
        sharpe_imp = result_adaptive.metrics["sharpe_ratio"] - result_baseline.metrics["sharpe_ratio"]
        print(f'  Return: {improvement:+.2f}%')
        print(f'  Sharpe: {sharpe_imp:+.2f}')

        if improvement > 0:
            print(f'  [+] Adaptive is BETTER')
        else:
            print(f'  [-] Baseline is better')

        results.append({
            'period': period_name,
            'baseline_return': result_baseline.metrics["total_return_pct"],
            'adaptive_return': result_adaptive.metrics["total_return_pct"],
            'baseline_sharpe': result_baseline.metrics["sharpe_ratio"],
            'adaptive_sharpe': result_adaptive.metrics["sharpe_ratio"],
            'improvement': improvement
        })

        # Show regime changes for 2025
        if period_name == '2025':
            print(f'\n{"="*70}')
            print('2025 REGIME ANALYSIS')
            print('='*70)

            # Check signals dataframe for regime info
            signals = adaptive.generate_signals(period_data)
            regimes = signals[signals['signal'] != 0][['date', 'regime', 'stop_mult', 'target_mult']]

            if len(regimes) > 0:
                print(f'\nRegime per signal:')
                for idx, row in regimes.iterrows():
                    print(f'  {row["date"]}: {row["regime"]} (stop={row["stop_mult"]:.2f}x, target={row["target_mult"]:.1f}x)')

    # Summary
    print(f'\n{"="*70}')
    print('SUMMARY')
    print('='*70)

    results_df = pd.DataFrame(results)

    adaptive_wins = sum(1 for _, row in results_df.iterrows() if row['improvement'] > 0)
    print(f'\nAdaptive outperformed in {adaptive_wins}/{len(results_df)} periods')

    print(f'\nPeriod-by-period:')
    for _, row in results_df.iterrows():
        symbol = '✓' if row['improvement'] > 0 else '✗'
        print(f'  {symbol} {row["period"]}: {row["improvement"]:+.2f}% ({row["adaptive_return"]:.2f}% vs {row["baseline_return"]:.2f}%)')

    # Key question: Does it capture 2025 gains without blowing up 2020-2024?
    print(f'\n{"="*70}')
    print('KEY METRICS')
    print('='*70)

    covid_row = results_df[results_df['period'] == 'COVID Era (2020-2021)']
    recent_row = results_df[results_df['period'] == 'Recent (2022-2024)']
    y2025_row = results_df[results_df['period'] == '2025']
    all_row = results_df[results_df['period'] == 'All Data (2015-2025)']

    if len(covid_row) > 0:
        print(f'\nCOVID (2020-2021) - Survival Test:')
        print(f'  Baseline: {covid_row.iloc[0]["baseline_return"]:.2f}%')
        print(f'  Adaptive: {covid_row.iloc[0]["adaptive_return"]:.2f}%')
        if covid_row.iloc[0]["adaptive_return"] > 0:
            print(f'  ✓ Adaptive survived COVID')
        else:
            print(f'  ✗ Adaptive lost money in COVID')

    if len(recent_row) > 0:
        print(f'\nRecent (2022-2024) - Stability Test:')
        print(f'  Baseline: {recent_row.iloc[0]["baseline_return"]:.2f}%')
        print(f'  Adaptive: {recent_row.iloc[0]["adaptive_return"]:.2f}%')
        if recent_row.iloc[0]["adaptive_return"] > 0:
            print(f'  ✓ Adaptive stayed positive')
        else:
            print(f'  ✗ Adaptive lost money')

    if len(y2025_row) > 0:
        print(f'\n2025 - Opportunity Test:')
        print(f'  Baseline: {y2025_row.iloc[0]["baseline_return"]:.2f}%')
        print(f'  Adaptive: {y2025_row.iloc[0]["adaptive_return"]:.2f}%')
        print(f'  Target: ~40% (tight stops benchmark)')
        if y2025_row.iloc[0]["adaptive_return"] > 20:
            print(f'  ✓ Adaptive captured more gains')
        elif y2025_row.iloc[0]["adaptive_return"] > y2025_row.iloc[0]["baseline_return"]:
            print(f'  ~ Adaptive better but not close to 40%')
        else:
            print(f'  ✗ Adaptive missed opportunity')

    if len(all_row) > 0:
        print(f'\nAll Data (2015-2025) - Overall:')
        print(f'  Baseline: {all_row.iloc[0]["baseline_return"]:.2f}%')
        print(f'  Adaptive: {all_row.iloc[0]["adaptive_return"]:.2f}%')
        if all_row.iloc[0]["adaptive_return"] > all_row.iloc[0]["baseline_return"]:
            print(f'  ✓ Adaptive improves overall returns')
        else:
            print(f'  ✗ Baseline still better overall')

    print(f'\n{"="*70}')
    print('CONCLUSION')
    print('='*70)

    all_positive = all(row['adaptive_return'] > 0 for _, row in results_df.iterrows() if 'All Data' not in row['period'])
    improved_2025 = len(y2025_row) > 0 and y2025_row.iloc[0]['adaptive_return'] > y2025_row.iloc[0]['baseline_return']

    if all_positive and improved_2025:
        print('SUCCESS! Adaptive strategy:')
        print('  ✓ Stayed positive in all periods')
        print('  ✓ Improved 2025 performance')
        print('  → Use adaptive strategy going forward')
    elif improved_2025 and adaptive_wins >= len(results_df) * 0.6:
        print('PARTIAL SUCCESS:')
        print('  ✓ Improved 2025 performance')
        print('  ~ Mixed results in other periods')
        print('  → Consider adaptive, monitor closely')
    else:
        print('Adaptive did not meet goals:')
        if not improved_2025:
            print('  ✗ Did not improve 2025 performance')
        if not all_positive:
            print('  ✗ Lost money in some periods')
        print('  → Stick with baseline or refine adaptation rules')

    return results_df


if __name__ == "__main__":
    results = test_adaptive_strategy()
