"""
Main script to run forex strategy backtests
"""
import sys
import os

# Add src and strategies to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'strategies'))

from src.data_fetcher import FMPDataFetcher
from src.backtester import Backtester
from src.performance_analysis import PerformanceAnalyzer, StrategyComparison
from strategies.ma_crossover_atr import MACrossoverATR
from strategies.volume_divergence import VolumeDivergence


def test_single_strategy(data, strategy, initial_capital=10000, risk_per_trade=0.02):
    """Test a single strategy"""
    print(f"\n{'='*70}")
    print(f"Testing: {strategy.name}")
    print(f"{'='*70}")
    print(strategy.get_description())

    # Run backtest
    backtester = Backtester(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        commission=0.0001  # 1 pip spread
    )

    result = backtester.run(data, strategy)

    # Analyze results
    analyzer = PerformanceAnalyzer(result, strategy.name)
    analyzer.generate_full_report()

    return result


def run_all_strategies(symbol='EURUSD', timeframe='1hour', from_date=None, to_date=None, use_cache=True):
    """Run all strategies and compare results"""
    print("="*70)
    print("FOREX STRATEGY BACKTESTING SYSTEM")
    print("="*70)

    # Fetch or load cached data
    fetcher = FMPDataFetcher()
    cache_filename = f"{symbol}_{timeframe}.csv"

    if use_cache:
        data = fetcher.load_data(cache_filename)
        if not data.empty:
            print(f"\nLoaded cached {symbol} data ({timeframe}): {len(data)} candles")
        else:
            print(f"\nNo cached data found. Fetching {symbol} data ({timeframe})...")
            data = fetcher.get_historical_data(symbol, from_date=from_date, to_date=to_date, timeframe=timeframe)
            if not data.empty:
                fetcher.save_data(data, cache_filename)
    else:
        print(f"\nFetching fresh {symbol} data ({timeframe})...")
        data = fetcher.get_historical_data(symbol, from_date=from_date, to_date=to_date, timeframe=timeframe)
        if not data.empty:
            fetcher.save_data(data, cache_filename)

    if data.empty:
        print("Failed to fetch data. Please check your API key and connection.")
        return

    print(f"Data range: {data['date'].min()} to {data['date'].max()}")

    # Define strategies to test
    strategies = [
        # OPTIMIZED: Volume Divergence (EUR/USD 1-day optimized)
        # 81.18% return over 10 years, Sharpe 2.77, 42.3% win rate
        VolumeDivergence(),  # Uses optimized defaults: 20/25/14, 1.0x stop, 2.0x target

        # Conservative option: Higher win rate, lower returns
        # 44.27% return over 10 years, Sharpe 3.65, 67.3% win rate
        VolumeDivergence(lookback_period=20, volume_period=25, atr_period=14,
                        stop_atr_multiplier=2.5, target_atr_multiplier=2.0),
    ]

    # Test all strategies
    results = {}
    for strategy in strategies:
        result = test_single_strategy(data, strategy)
        results[strategy.name] = result

    # Compare strategies
    print("\n" + "="*70)
    print("COMPARING ALL STRATEGIES")
    print("="*70)

    comparison = StrategyComparison(results)
    comparison.compare_metrics()
    comparison.plot_comparison()

    print("\nAll results saved to 'results' folder")
    print("="*70)


def quick_test(use_cache=True):
    """Quick test with a single strategy"""
    print("Running quick test with EUR/USD...")

    fetcher = FMPDataFetcher()
    cache_filename = "EURUSD_1hour.csv"

    if use_cache:
        data = fetcher.load_data(cache_filename)
        if data.empty:
            data = fetcher.get_historical_data('EURUSD', timeframe='1hour')
            if not data.empty:
                fetcher.save_data(data, cache_filename)
    else:
        data = fetcher.get_historical_data('EURUSD', timeframe='1hour')
        if not data.empty:
            fetcher.save_data(data, cache_filename)

    if data.empty:
        print("Failed to fetch data")
        return

    # Test MA Crossover strategy
    strategy = MACrossoverATR(fast_period=20, slow_period=50)
    test_single_strategy(data, strategy)


def test_multiple_pairs(use_cache=True):
    """Test best strategies on multiple currency pairs"""
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    fetcher = FMPDataFetcher()

    # Use best performing strategy from initial tests
    strategy = MACrossoverATR(fast_period=20, slow_period=50, atr_stop_multiplier=2.0)

    print("\n" + "="*70)
    print(f"Testing {strategy.name} across multiple pairs")
    print("="*70)

    results = {}
    for pair in pairs:
        print(f"\n\nTesting {pair}...")
        cache_filename = f"{pair}_1hour.csv"

        if use_cache:
            data = fetcher.load_data(cache_filename)
            if data.empty:
                data = fetcher.get_historical_data(pair, timeframe='1hour')
                if not data.empty:
                    fetcher.save_data(data, cache_filename)
        else:
            data = fetcher.get_historical_data(pair, timeframe='1hour')
            if not data.empty:
                fetcher.save_data(data, cache_filename)

        if not data.empty:
            backtester = Backtester(initial_capital=10000, risk_per_trade=0.02, commission=0.0001)
            result = backtester.run(data, strategy)

            analyzer = PerformanceAnalyzer(result, f"{strategy.name}_{pair}")
            analyzer.print_summary()

            results[pair] = result

    # Compare across pairs
    if results:
        comparison = StrategyComparison(results)
        comparison.compare_metrics()


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Choose what to run:

    # Option 1: Quick test with single strategy
    # quick_test()

    # Option 2: Test optimized strategies on EUR/USD (recommended)
    # Volume Divergence optimized for EUR/USD 1-day
    run_all_strategies(symbol='EURUSD', timeframe='1day')

    # Option 3: Test on other pairs to check robustness
    # run_all_strategies(symbol='GBPUSD', timeframe='1day')
    # run_all_strategies(symbol='USDJPY', timeframe='1day')

    # Option 4: Test best strategy on multiple pairs
    # test_multiple_pairs()

    # Option 5: Custom test
    # fetcher = FMPDataFetcher()
    # data = fetcher.get_historical_data('GBPUSD', timeframe='1day')
    # strategy = VolumeDivergence()  # Uses optimized defaults
    # test_single_strategy(data, strategy, initial_capital=10000, risk_per_trade=0.02)
