"""
Enhanced Moving Average Crossover with ADX Filter

Improvement over basic MA crossover:
- Only trades when ADX > threshold (confirms trending market)
- Filters out choppy/ranging periods that cause whipsaws
- Should reduce losing trades and improve win rate

Expected Improvement: +1-2% return, +5-10% win rate
"""
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange


class MACrossoverADX:
    """Moving Average Crossover with ADX trend filter"""

    def __init__(self, fast_period=20, slow_period=50, atr_period=14,
                 atr_stop_multiplier=2.0, risk_reward_ratio=2.5,
                 adx_period=14, adx_threshold=25, use_ema=True):
        """
        Initialize strategy parameters

        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period
            atr_period: ATR calculation period
            atr_stop_multiplier: Stop loss distance in ATR units
            risk_reward_ratio: Take profit distance ratio
            adx_period: ADX calculation period (default 14)
            adx_threshold: Minimum ADX to take trades (default 25)
            use_ema: Use EMA instead of SMA
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.use_ema = use_ema
        self.name = f"MA_ADX_{fast_period}_{slow_period}_ADX{adx_threshold}"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with ADX filter

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with signals and indicators
        """
        df = data.copy()

        # Calculate Moving Averages
        if self.use_ema:
            df['fast_ma'] = EMAIndicator(close=df['close'], window=self.fast_period).ema_indicator()
            df['slow_ma'] = EMAIndicator(close=df['close'], window=self.slow_period).ema_indicator()
        else:
            df['fast_ma'] = SMAIndicator(close=df['close'], window=self.fast_period).sma_indicator()
            df['slow_ma'] = SMAIndicator(close=df['close'], window=self.slow_period).sma_indicator()

        # Calculate ATR
        df['atr'] = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.atr_period
        ).average_true_range()

        # Calculate ADX
        adx_indicator = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.adx_period
        )
        df['adx'] = adx_indicator.adx()

        # Generate crossover signals
        df['ma_diff'] = df['fast_ma'] - df['slow_ma']
        df['prev_ma_diff'] = df['ma_diff'].shift(1)

        # Initialize signal column
        df['signal'] = 0

        # Long signal: fast MA crosses above slow MA AND ADX > threshold
        long_condition = (
            (df['ma_diff'] > 0) &
            (df['prev_ma_diff'] <= 0) &
            (df['adx'] > self.adx_threshold)  # ADX FILTER
        )
        df.loc[long_condition, 'signal'] = 1

        # Short signal: fast MA crosses below slow MA AND ADX > threshold
        short_condition = (
            (df['ma_diff'] < 0) &
            (df['prev_ma_diff'] >= 0) &
            (df['adx'] > self.adx_threshold)  # ADX FILTER
        )
        df.loc[short_condition, 'signal'] = -1

        # Calculate stop loss and take profit levels
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan

        # For long positions
        long_signals = df['signal'] == 1
        df.loc[long_signals, 'stop_loss'] = (
            df.loc[long_signals, 'close'] -
            df.loc[long_signals, 'atr'] * self.atr_stop_multiplier
        )
        stop_distance_long = df.loc[long_signals, 'close'] - df.loc[long_signals, 'stop_loss']
        df.loc[long_signals, 'take_profit'] = (
            df.loc[long_signals, 'close'] +
            stop_distance_long * self.risk_reward_ratio
        )

        # For short positions
        short_signals = df['signal'] == -1
        df.loc[short_signals, 'stop_loss'] = (
            df.loc[short_signals, 'close'] +
            df.loc[short_signals, 'atr'] * self.atr_stop_multiplier
        )
        stop_distance_short = df.loc[short_signals, 'stop_loss'] - df.loc[short_signals, 'close']
        df.loc[short_signals, 'take_profit'] = (
            df.loc[short_signals, 'close'] -
            stop_distance_short * self.risk_reward_ratio
        )

        # Forward fill stop loss and take profit for open positions
        in_position = False
        current_stop = None
        current_tp = None
        current_direction = None

        for i in range(len(df)):
            if df.iloc[i]['signal'] != 0:
                in_position = True
                current_stop = df.iloc[i]['stop_loss']
                current_tp = df.iloc[i]['take_profit']
                current_direction = df.iloc[i]['signal']
            elif in_position:
                # Check if opposite signal (exit)
                if current_direction == 1 and df.iloc[i-1]['ma_diff'] < 0:
                    in_position = False
                    df.at[df.index[i], 'signal'] = 0  # Exit signal
                elif current_direction == -1 and df.iloc[i-1]['ma_diff'] > 0:
                    in_position = False
                    df.at[df.index[i], 'signal'] = 0  # Exit signal
                else:
                    # Keep stop loss and take profit active
                    df.at[df.index[i], 'stop_loss'] = current_stop
                    df.at[df.index[i], 'take_profit'] = current_tp

        return df

    def get_description(self):
        """Return strategy description"""
        ma_type = "EMA" if self.use_ema else "SMA"
        return f"""
Enhanced Moving Average Crossover with ADX Filter

Parameters:
- Fast MA: {ma_type}{self.fast_period}
- Slow MA: {ma_type}{self.slow_period}
- ATR Period: {self.atr_period}
- Stop Loss: {self.atr_stop_multiplier}x ATR
- Risk/Reward: {self.risk_reward_ratio}:1
- ADX Period: {self.adx_period}
- ADX Threshold: {self.adx_threshold}

Logic:
- Long when fast MA crosses above slow MA AND ADX > {self.adx_threshold}
- Short when fast MA crosses below slow MA AND ADX > {self.adx_threshold}
- ADX filter ensures we only trade in trending markets
- Skips trades during choppy/ranging periods (ADX < {self.adx_threshold})

Improvement over basic MA crossover:
- Filters out low-probability whipsaw trades
- Higher win rate, fewer losing trades
- Better risk-adjusted returns
"""


if __name__ == "__main__":
    # Test the strategy
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

    from src.data_fetcher import FMPDataFetcher
    from src.backtester import Backtester
    from src.performance_analysis import PerformanceAnalyzer

    print("="*80)
    print("Testing Enhanced MA Crossover with ADX Filter")
    print("="*80)

    fetcher = FMPDataFetcher()
    data = fetcher.load_data('EURUSD_1day.csv')

    if not data.empty:
        # Test original vs ADX-filtered
        print("\n1. ORIGINAL MA 20/50 (No ADX filter):")
        print("-" * 80)

        from strategies.ma_crossover_atr import MACrossoverATR
        original = MACrossoverATR(fast_period=20, slow_period=50, atr_stop_multiplier=2.0, risk_reward_ratio=2.5)
        backtester = Backtester(initial_capital=10000, risk_per_trade=0.02, commission=0.0001)
        result_original = backtester.run(data, original)

        print(f"Trades: {result_original.metrics['total_trades']}")
        print(f"Win Rate: {result_original.metrics['win_rate']:.1f}%")
        print(f"Return: {result_original.metrics['total_return_pct']:.2f}%")
        print(f"Sharpe: {result_original.metrics['sharpe_ratio']:.2f}")

        print("\n2. ENHANCED MA 20/50 (With ADX > 25 filter):")
        print("-" * 80)

        enhanced = MACrossoverADX(fast_period=20, slow_period=50, atr_stop_multiplier=2.0,
                                  risk_reward_ratio=2.5, adx_threshold=25)
        result_enhanced = backtester.run(data, enhanced)

        print(f"Trades: {result_enhanced.metrics['total_trades']}")
        print(f"Win Rate: {result_enhanced.metrics['win_rate']:.1f}%")
        print(f"Return: {result_enhanced.metrics['total_return_pct']:.2f}%")
        print(f"Sharpe: {result_enhanced.metrics['sharpe_ratio']:.2f}")

        print("\n" + "="*80)
        print("COMPARISON:")
        print("="*80)
        print(f"Trades Reduced: {result_original.metrics['total_trades'] - result_enhanced.metrics['total_trades']}")
        print(f"Win Rate Change: {result_enhanced.metrics['win_rate'] - result_original.metrics['win_rate']:+.1f}%")
        print(f"Return Change: {result_enhanced.metrics['total_return_pct'] - result_original.metrics['total_return_pct']:+.2f}%")
        print(f"Sharpe Change: {result_enhanced.metrics['sharpe_ratio'] - result_original.metrics['sharpe_ratio']:+.2f}")
    else:
        print("No data found. Run fetch_3year_data.py first.")
