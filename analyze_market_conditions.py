"""
Analyze market conditions to understand why hourly vs daily differ
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_fetcher import FMPDataFetcher
import pandas as pd
import numpy as np

fetcher = FMPDataFetcher()

pairs = ['EURUSD', 'GBPUSD']

print("="*80)
print("MARKET CONDITION ANALYSIS")
print("="*80)

for pair in pairs:
    print(f"\n{'='*80}")
    print(f"{pair} ANALYSIS")
    print(f"{'='*80}")

    # Load hourly data
    hourly = fetcher.load_data(f'{pair}_1hour.csv')

    if not hourly.empty:
        print(f"\nHOURLY ({len(hourly)} bars from {hourly['date'].min()} to {hourly['date'].max()})")

        # Calculate basic stats
        returns = hourly['close'].pct_change()

        # Trend strength: compare close to moving average
        hourly['sma20'] = hourly['close'].rolling(20).mean()
        hourly['sma50'] = hourly['close'].rolling(50).mean()

        # Calculate how often price is trending (above/below MA)
        above_sma20 = (hourly['close'] > hourly['sma20']).sum() / len(hourly) * 100
        above_sma50 = (hourly['close'] > hourly['sma50']).sum() / len(hourly) * 100

        # Calculate volatility
        volatility = returns.std() * np.sqrt(252 * 24)  # Annualized hourly volatility

        # Calculate max drawdown
        cummax = hourly['close'].cummax()
        drawdown = (hourly['close'] - cummax) / cummax * 100
        max_dd = drawdown.min()

        # Overall trend
        total_return = (hourly['close'].iloc[-1] / hourly['close'].iloc[0] - 1) * 100

        # Check for ranging vs trending
        # If price oscillates around MA, it's ranging
        crosses = ((hourly['close'] > hourly['sma20']) != (hourly['close'].shift(1) > hourly['sma20'])).sum()

        print(f"  Total Return: {total_return:+.2f}%")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        print(f"  Annualized Vol: {volatility:.2f}%")
        print(f"  Price above SMA20: {above_sma20:.1f}% of time")
        print(f"  Price above SMA50: {above_sma50:.1f}% of time")
        print(f"  SMA20 crosses: {crosses} (high = choppy, low = trending)")

        # Determine market type
        if crosses > 30:
            market_type = "CHOPPY/RANGING"
        elif above_sma20 > 60 or above_sma20 < 40:
            market_type = "TRENDING"
        else:
            market_type = "MIXED"

        print(f"  Market Type: {market_type}")

    # Load daily data
    daily = fetcher.load_data(f'{pair}_1day.csv')

    if not daily.empty:
        print(f"\nDAILY ({len(daily)} bars from {daily['date'].min()} to {daily['date'].max()})")

        # Same analysis for daily
        returns = daily['close'].pct_change()
        daily['sma20'] = daily['close'].rolling(20).mean()
        daily['sma50'] = daily['close'].rolling(50).mean()

        above_sma20 = (daily['close'] > daily['sma20']).sum() / len(daily) * 100
        above_sma50 = (daily['close'] > daily['sma50']).sum() / len(daily) * 100

        volatility = returns.std() * np.sqrt(252)

        cummax = daily['close'].cummax()
        drawdown = (daily['close'] - cummax) / cummax * 100
        max_dd = drawdown.min()

        total_return = (daily['close'].iloc[-1] / daily['close'].iloc[0] - 1) * 100

        crosses = ((daily['close'] > daily['sma20']) != (daily['close'].shift(1) > daily['sma20'])).sum()

        print(f"  Total Return: {total_return:+.2f}%")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        print(f"  Annualized Vol: {volatility:.2f}%")
        print(f"  Price above SMA20: {above_sma20:.1f}% of time")
        print(f"  Price above SMA50: {above_sma50:.1f}% of time")
        print(f"  SMA20 crosses: {crosses}")

        if crosses > 50:
            market_type = "CHOPPY/RANGING"
        elif above_sma20 > 60 or above_sma20 < 40:
            market_type = "TRENDING"
        else:
            market_type = "MIXED"

        print(f"  Market Type: {market_type}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("Choppy/ranging markets hurt trend-following strategies")
print("Trending markets favor MA crossover strategies")
print("="*80)
