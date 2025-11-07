"""
Moving Average Crossover Strategy with ATR-based Position Sizing and Stops

Strategy Logic:
- Uses fast and slow moving averages for trend identification
- ATR (Average True Range) for volatility-based stops and position sizing
- Entry: When fast MA crosses above slow MA (long) or below (short)
- Exit: When MAs cross in opposite direction or hit stop loss/take profit
- Stop Loss: ATR-based (1.5x ATR from entry)
- Take Profit: Risk/reward ratio based (2:1 or 3:1)
"""
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


class MACrossoverATR:
    """Moving Average Crossover with ATR risk management"""

    def __init__(self, fast_period=20, slow_period=50, atr_period=14,
                 atr_stop_multiplier=1.5, risk_reward_ratio=2.0,
                 use_ema=True):
        """
        Initialize strategy parameters

        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period
            atr_period: ATR calculation period
            atr_stop_multiplier: Stop loss distance in ATR units
            risk_reward_ratio: Take profit distance ratio
            use_ema: Use EMA instead of SMA
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.use_ema = use_ema
        self.name = f"MA_Crossover_ATR_{fast_period}_{slow_period}"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

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

        # Generate crossover signals
        df['ma_diff'] = df['fast_ma'] - df['slow_ma']
        df['prev_ma_diff'] = df['ma_diff'].shift(1)

        # Initialize signal column
        df['signal'] = 0

        # Long signal: fast MA crosses above slow MA
        long_condition = (df['ma_diff'] > 0) & (df['prev_ma_diff'] <= 0)
        df.loc[long_condition, 'signal'] = 1

        # Short signal: fast MA crosses below slow MA
        short_condition = (df['ma_diff'] < 0) & (df['prev_ma_diff'] >= 0)
        df.loc[short_condition, 'signal'] = -1

        # Calculate stop loss and take profit levels
        # These are set at signal generation and used by the backtester
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
        # This allows the backtester to check them on every bar
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
Moving Average Crossover with ATR Risk Management

Parameters:
- Fast MA: {ma_type}{self.fast_period}
- Slow MA: {ma_type}{self.slow_period}
- ATR Period: {self.atr_period}
- Stop Loss: {self.atr_stop_multiplier}x ATR
- Risk/Reward: {self.risk_reward_ratio}:1

Logic:
- Long when fast MA crosses above slow MA
- Short when fast MA crosses below slow MA
- Dynamic stops based on ATR for volatility adaptation
- Fixed risk/reward ratio for consistent risk management
"""


if __name__ == "__main__":
    # Test the strategy
    from src.data_fetcher import FMPDataFetcher

    fetcher = FMPDataFetcher()
    data = fetcher.get_historical_data('EURUSD', timeframe='1hour')

    if not data.empty:
        strategy = MACrossoverATR()
        signals = strategy.generate_signals(data)

        print(strategy.get_description())
        print(f"\nGenerated {len(signals[signals['signal'] != 0])} signals")
        print(f"Long signals: {len(signals[signals['signal'] == 1])}")
        print(f"Short signals: {len(signals[signals['signal'] == -1])}")
