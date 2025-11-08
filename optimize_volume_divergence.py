"""
Parameter Optimization for Volume Divergence Strategy
Tests multiple parameter combinations to find optimal settings
"""
import sys
import os
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime

# Add src and strategies to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'strategies'))

from src.data_fetcher import FMPDataFetcher
from src.backtester import Backtester
from strategies.volume_divergence import VolumeDivergence


def optimize_strategy(data, param_grid, initial_capital=10000, risk_per_trade=0.02):
    """
    Test all parameter combinations and return results

    Args:
        data: DataFrame with OHLCV data
        param_grid: Dictionary with parameter ranges
        initial_capital: Starting capital
        risk_per_trade: Risk per trade as fraction

    Returns:
        DataFrame with results for all parameter combinations
    """
    results = []

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    total = len(combinations)
    print(f"\nTesting {total} parameter combinations...")
    print("="*70)

    for i, params in enumerate(combinations, 1):
        # Create parameter dictionary
        param_dict = dict(zip(param_names, params))

        # Create strategy with these parameters
        strategy = VolumeDivergence(**param_dict)

        # Run backtest
        backtester = Backtester(
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            commission=0.0001  # 1 pip spread
        )

        result = backtester.run(data, strategy)

        # Extract key metrics
        metrics = result.metrics

        # Add to results
        result_row = {
            **param_dict,
            'total_trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_return_pct': metrics.get('total_return_pct', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'final_capital': metrics.get('final_capital', initial_capital),
            'avg_win': metrics.get('avg_win', 0),
            'avg_loss': metrics.get('avg_loss', 0)
        }
        results.append(result_row)

        # Progress update
        if i % 10 == 0 or i == total:
            print(f"Progress: {i}/{total} ({i/total*100:.1f}%)")
            if result_row['total_trades'] > 0:
                print(f"  Latest: {param_dict} -> Return: {result_row['total_return_pct']:.2f}%, "
                      f"Win Rate: {result_row['win_rate']:.1f}%, Trades: {result_row['total_trades']}")

    return pd.DataFrame(results)


def analyze_results(results_df, min_trades=10):
    """
    Analyze optimization results and identify best parameters

    Args:
        results_df: DataFrame with optimization results
        min_trades: Minimum number of trades to consider valid
    """
    # Filter results with enough trades
    valid_results = results_df[results_df['total_trades'] >= min_trades].copy()

    if valid_results.empty:
        print(f"\nNo results with at least {min_trades} trades found!")
        return

    print("\n" + "="*70)
    print(f"OPTIMIZATION RESULTS (min {min_trades} trades)")
    print("="*70)
    print(f"\nTotal combinations tested: {len(results_df)}")
    print(f"Valid combinations (>={min_trades} trades): {len(valid_results)}")

    # Sort by different metrics
    print("\n" + "="*70)
    print("TOP 10 BY TOTAL RETURN")
    print("="*70)
    top_return = valid_results.nlargest(10, 'total_return_pct')
    print(top_return[['lookback_period', 'volume_period', 'atr_period',
                      'stop_atr_multiplier', 'target_atr_multiplier',
                      'total_return_pct', 'win_rate', 'total_trades',
                      'sharpe_ratio', 'max_drawdown']].to_string(index=False))

    print("\n" + "="*70)
    print("TOP 10 BY SHARPE RATIO")
    print("="*70)
    top_sharpe = valid_results.nlargest(10, 'sharpe_ratio')
    print(top_sharpe[['lookback_period', 'volume_period', 'atr_period',
                      'stop_atr_multiplier', 'target_atr_multiplier',
                      'sharpe_ratio', 'total_return_pct', 'win_rate',
                      'total_trades', 'max_drawdown']].to_string(index=False))

    print("\n" + "="*70)
    print("TOP 10 BY WIN RATE")
    print("="*70)
    top_winrate = valid_results.nlargest(10, 'win_rate')
    print(top_winrate[['lookback_period', 'volume_period', 'atr_period',
                       'stop_atr_multiplier', 'target_atr_multiplier',
                       'win_rate', 'total_return_pct', 'total_trades',
                       'sharpe_ratio', 'max_drawdown']].to_string(index=False))

    print("\n" + "="*70)
    print("TOP 10 BY PROFIT FACTOR")
    print("="*70)
    top_pf = valid_results.nlargest(10, 'profit_factor')
    print(top_pf[['lookback_period', 'volume_period', 'atr_period',
                  'stop_atr_multiplier', 'target_atr_multiplier',
                  'profit_factor', 'total_return_pct', 'win_rate',
                  'total_trades', 'sharpe_ratio']].to_string(index=False))

    # Calculate composite score
    # Normalize metrics and create weighted score
    valid_results['return_score'] = (valid_results['total_return_pct'] - valid_results['total_return_pct'].min()) / \
                                     (valid_results['total_return_pct'].max() - valid_results['total_return_pct'].min() + 1e-10)
    valid_results['sharpe_score'] = (valid_results['sharpe_ratio'] - valid_results['sharpe_ratio'].min()) / \
                                     (valid_results['sharpe_ratio'].max() - valid_results['sharpe_ratio'].min() + 1e-10)
    valid_results['winrate_score'] = valid_results['win_rate'] / 100
    valid_results['drawdown_score'] = 1 - (abs(valid_results['max_drawdown']) / 100)

    # Weighted composite score (return 40%, sharpe 30%, win rate 20%, drawdown 10%)
    valid_results['composite_score'] = (
        valid_results['return_score'] * 0.4 +
        valid_results['sharpe_score'] * 0.3 +
        valid_results['winrate_score'] * 0.2 +
        valid_results['drawdown_score'] * 0.1
    )

    print("\n" + "="*70)
    print("TOP 10 BY COMPOSITE SCORE (Return 40%, Sharpe 30%, WinRate 20%, Drawdown 10%)")
    print("="*70)
    top_composite = valid_results.nlargest(10, 'composite_score')
    print(top_composite[['lookback_period', 'volume_period', 'atr_period',
                         'stop_atr_multiplier', 'target_atr_multiplier',
                         'composite_score', 'total_return_pct', 'sharpe_ratio',
                         'win_rate', 'total_trades', 'max_drawdown']].to_string(index=False))

    # Best overall
    best = valid_results.loc[valid_results['composite_score'].idxmax()]
    print("\n" + "="*70)
    print("RECOMMENDED PARAMETERS (Best Composite Score)")
    print("="*70)
    print(f"Lookback Period: {int(best['lookback_period'])}")
    print(f"Volume Period: {int(best['volume_period'])}")
    print(f"ATR Period: {int(best['atr_period'])}")
    print(f"Stop ATR Multiplier: {best['stop_atr_multiplier']:.1f}")
    print(f"Target ATR Multiplier: {best['target_atr_multiplier']:.1f}")
    print(f"\nPerformance:")
    print(f"  Total Return: {best['total_return_pct']:.2f}%")
    print(f"  Win Rate: {best['win_rate']:.1f}%")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Max Drawdown: {best['max_drawdown']:.2f}%")
    print(f"  Total Trades: {int(best['total_trades'])}")
    print(f"  Composite Score: {best['composite_score']:.3f}")

    return valid_results


def run_optimization(symbol='EURUSD', timeframe='1day', min_trades=10):
    """
    Run full parameter optimization

    Args:
        symbol: Currency pair to test
        timeframe: Timeframe to use
        min_trades: Minimum trades to consider valid
    """
    print("="*70)
    print("VOLUME DIVERGENCE STRATEGY - PARAMETER OPTIMIZATION")
    print("="*70)

    # Load data
    fetcher = FMPDataFetcher()
    cache_filename = f"{symbol}_{timeframe}.csv"
    data = fetcher.load_data(cache_filename)

    if data.empty:
        print(f"No data found for {symbol} {timeframe}")
        print("Please ensure data files exist in the data/ directory")
        return

    print(f"\nSymbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Data range: {data['date'].min()} to {data['date'].max()}")
    print(f"Total candles: {len(data)}")

    # Define parameter grid to test
    param_grid = {
        'lookback_period': [20, 30, 40, 48, 60],  # Period for high/low detection
        'volume_period': [10, 15, 20, 25, 30],     # Period for volume average
        'atr_period': [10, 14, 20],                # ATR calculation period
        'stop_atr_multiplier': [1.0, 1.5, 2.0, 2.5],  # Stop loss distance
        'target_atr_multiplier': [2.0, 3.0, 4.0, 4.5, 5.0, 6.0]  # Take profit distance
    }

    print("\nParameter ranges:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Run optimization
    results_df = optimize_strategy(data, param_grid)

    # Save raw results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"results/optimization_{symbol}_{timeframe}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nRaw results saved to: {results_file}")

    # Analyze results
    valid_results = analyze_results(results_df, min_trades=min_trades)

    # Save filtered results
    if valid_results is not None and not valid_results.empty:
        filtered_file = f"results/optimization_{symbol}_{timeframe}_{timestamp}_filtered.csv"
        valid_results.to_csv(filtered_file, index=False)
        print(f"\nFiltered results saved to: {filtered_file}")

    print("\n" + "="*70)


def quick_optimization():
    """Quick optimization with reduced parameter space"""
    print("="*70)
    print("QUICK PARAMETER OPTIMIZATION")
    print("="*70)

    # Load data
    fetcher = FMPDataFetcher()
    data = fetcher.load_data("EURUSD_1day.csv")

    if data.empty:
        print("No data found!")
        return

    print(f"Testing EUR/USD 1-day data: {len(data)} candles")

    # Smaller parameter grid for quick test
    param_grid = {
        'lookback_period': [40, 48, 60],
        'volume_period': [15, 20, 25],
        'atr_period': [14],
        'stop_atr_multiplier': [1.5, 2.0],
        'target_atr_multiplier': [3.0, 4.0, 4.5]
    }

    results_df = optimize_strategy(data, param_grid)
    analyze_results(results_df, min_trades=5)


if __name__ == "__main__":
    # Create results directory if needed
    os.makedirs('results', exist_ok=True)

    # Choose optimization type:

    # Option 1: Full optimization on single pair
    # This will test 5*5*3*4*6 = 1800 combinations
    run_optimization(symbol='EURUSD', timeframe='1day', min_trades=10)

    # Option 2: Test on multiple pairs
    # for symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']:
    #     run_optimization(symbol=symbol, timeframe='1day', min_trades=10)

    # Option 3: Quick test with fewer parameters
    # quick_optimization()
