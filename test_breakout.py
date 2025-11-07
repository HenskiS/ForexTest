"""
Test the breakout strategy
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'strategies'))

from src.data_fetcher import FMPDataFetcher
from src.backtester import Backtester
from src.performance_analysis import PerformanceAnalyzer
from strategies.breakout_atr import BreakoutATR

print("="*80)
print("TESTING BREAKOUT STRATEGY")
print("="*80)

# Load EUR/USD daily data
fetcher = FMPDataFetcher()
data = fetcher.load_data('EURUSD_1day.csv')

if data.empty:
    print("No data found. Run fetch_3year_data.py first.")
    exit(1)

print(f"\nData: {len(data)} bars from {data['date'].min()} to {data['date'].max()}")

# Test different parameter combinations
configs = [
    # (lookback, stop_atr, target_atr)
    (48, 2.0, 3.0),  # Original: 1.5:1 R/R
    (48, 1.5, 3.0),  # Tighter stop: 2:1 R/R
    (48, 2.0, 4.0),  # Wider target: 2:1 R/R
    (30, 2.0, 3.0),  # Shorter lookback
    (60, 2.0, 3.0),  # Longer lookback
]

print("\n" + "="*80)
print("Testing different parameter combinations:")
print("="*80)

results = {}

for lookback, stop_mult, target_mult in configs:
    strategy = BreakoutATR(
        lookback_period=lookback,
        stop_atr_multiplier=stop_mult,
        target_atr_multiplier=target_mult
    )

    print(f"\n\nTesting: {strategy.name}")
    print("-" * 80)

    # Run backtest
    backtester = Backtester(
        initial_capital=10000,
        risk_per_trade=0.02,  # 2% of capital
        commission=0.0001
    )

    result = backtester.run(data, strategy)

    # Print summary
    if result.trades:
        print(f"Trades: {result.metrics['total_trades']}")
        print(f"Win Rate: {result.metrics['win_rate']:.1f}%")
        print(f"Return: {result.metrics['total_return_pct']:.2f}%")
        print(f"Max DD: {result.metrics['max_drawdown']:.2f}%")
        print(f"Sharpe: {result.metrics['sharpe_ratio']:.2f}")
        print(f"Profit Factor: {result.metrics['profit_factor']:.2f}")

        results[strategy.name] = result
    else:
        print("No trades generated")

# Show best performer
if results:
    print("\n" + "="*80)
    print("BEST PERFORMER:")
    print("="*80)

    best_name = max(results.keys(),
                   key=lambda k: results[k].metrics['total_return_pct'])
    best_result = results[best_name]

    analyzer = PerformanceAnalyzer(best_result, best_name)
    analyzer.generate_full_report()

print("\n" + "="*80)
print("Testing complete!")
print("="*80)
