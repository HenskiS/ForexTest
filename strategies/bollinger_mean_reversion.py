"""
Bollinger Bands Mean Reversion Strategy

Strategy Logic:
- Uses Bollinger Bands to identify overbought/oversold conditions
- RSI filter to confirm momentum
- Entry: When price touches outer band with RSI confirmation
- Exit: When price returns to middle band or opposite signal
- Risk management based on band width
"""
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator


class BollingerMeanReversion:
    """Bollinger Bands Mean Reversion Strategy"""

    def __init__(self, bb_period=20, bb_std=2.0, rsi_period=14,
                 rsi_oversold=30, rsi_overbought=70,
                 use_rsi_filter=True, risk_reward_ratio=2.0):
        """
        Initialize strategy parameters

        Args:
            bb_period: Bollinger Bands period
            bb_std: Number of standard deviations for bands
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            use_rsi_filter: Whether to use RSI confirmation
            risk_reward_ratio: Risk/reward ratio for exits
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.use_rsi_filter = use_rsi_filter
        self.risk_reward_ratio = risk_reward_ratio
        self.name = f"BB_MeanReversion_{bb_period}_{bb_std}_RR{risk_reward_ratio}"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with signals and indicators
        """
        df = data.copy()

        # Calculate Bollinger Bands
        bb = BollingerBands(
            close=df['close'],
            window=self.bb_period,
            window_dev=self.bb_std
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Calculate RSI
        df['rsi'] = RSIIndicator(
            close=df['close'],
            window=self.rsi_period
        ).rsi()

        # Calculate percentage of price position within bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Initialize signal column
        df['signal'] = 0

        # Long signal: Price touches/crosses lower band (oversold)
        # Price at or below lower band
        long_price_condition = df['low'] <= df['bb_lower']

        if self.use_rsi_filter:
            # RSI also shows oversold
            long_condition = long_price_condition & (df['rsi'] < self.rsi_oversold)
        else:
            long_condition = long_price_condition

        df.loc[long_condition, 'signal'] = 1

        # Short signal: Price touches/crosses upper band (overbought)
        # Price at or above upper band
        short_price_condition = df['high'] >= df['bb_upper']

        if self.use_rsi_filter:
            # RSI also shows overbought
            short_condition = short_price_condition & (df['rsi'] > self.rsi_overbought)
        else:
            short_condition = short_price_condition

        df.loc[short_condition, 'signal'] = -1

        # Calculate stop loss and take profit levels
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan

        # For long positions
        # Stop: Below lower band, Target: Based on risk/reward ratio
        long_signals = df['signal'] == 1
        band_range = df.loc[long_signals, 'bb_upper'] - df.loc[long_signals, 'bb_lower']
        df.loc[long_signals, 'stop_loss'] = (
            df.loc[long_signals, 'close'] - band_range * 0.2  # Stop 20% of band width below entry
        )
        # Calculate stop distance and apply risk/reward ratio
        stop_distance_long = df.loc[long_signals, 'close'] - df.loc[long_signals, 'stop_loss']
        df.loc[long_signals, 'take_profit'] = (
            df.loc[long_signals, 'close'] + stop_distance_long * self.risk_reward_ratio
        )

        # For short positions
        # Stop: Above upper band, Target: Based on risk/reward ratio
        short_signals = df['signal'] == -1
        band_range_short = df.loc[short_signals, 'bb_upper'] - df.loc[short_signals, 'bb_lower']
        df.loc[short_signals, 'stop_loss'] = (
            df.loc[short_signals, 'close'] + band_range_short * 0.2  # Stop 20% of band width above entry
        )
        # Calculate stop distance and apply risk/reward ratio
        stop_distance_short = df.loc[short_signals, 'stop_loss'] - df.loc[short_signals, 'close']
        df.loc[short_signals, 'take_profit'] = (
            df.loc[short_signals, 'close'] - stop_distance_short * self.risk_reward_ratio
        )

        # Forward fill stop loss and take profit for backtester
        # The backtester will handle exits based on stop/target
        in_position = False
        current_stop = None
        current_tp = None

        for i in range(len(df)):
            if df.iloc[i]['signal'] != 0:
                in_position = True
                current_stop = df.iloc[i]['stop_loss']
                current_tp = df.iloc[i]['take_profit']
            elif in_position:
                # Keep stop loss and take profit active
                df.at[df.index[i], 'stop_loss'] = current_stop
                df.at[df.index[i], 'take_profit'] = current_tp

        return df

    def get_description(self):
        """Return strategy description"""
        return f"""
Bollinger Bands Mean Reversion Strategy

Parameters:
- BB Period: {self.bb_period}
- BB Std Dev: {self.bb_std}
- RSI Period: {self.rsi_period}
- RSI Oversold: {self.rsi_oversold}
- RSI Overbought: {self.rsi_overbought}
- Use RSI Filter: {self.use_rsi_filter}
- Risk/Reward: {self.risk_reward_ratio}:1

Logic:
- Long when price touches lower band (oversold) with RSI < {self.rsi_oversold}
- Short when price touches upper band (overbought) with RSI > {self.rsi_overbought}
- Exit at stop loss or take profit based on {self.risk_reward_ratio}:1 ratio
- Stop loss: 20% of band width from entry
- Works best in ranging/sideways markets
"""


class BollingerBounce:
    """Alternative Bollinger strategy - trade bounces off bands"""

    def __init__(self, bb_period=20, bb_std=2.0, atr_period=14,
                 atr_stop_multiplier=2.0):
        """
        Initialize strategy - trades bounces off Bollinger Bands

        Args:
            bb_period: Bollinger Bands period
            bb_std: Standard deviation multiplier
            atr_period: ATR period for stops
            atr_stop_multiplier: Stop loss distance in ATR
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.name = f"BB_Bounce_{bb_period}_{bb_std}"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for bounce strategy"""
        from ta.volatility import AverageTrueRange

        df = data.copy()

        # Calculate Bollinger Bands
        bb = BollingerBands(close=df['close'], window=self.bb_period, window_dev=self.bb_std)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()

        # Calculate ATR
        df['atr'] = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.atr_period
        ).average_true_range()

        df['signal'] = 0
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan

        # Look for bounces: price touches band then reverses
        for i in range(2, len(df)):
            prev_close = df.iloc[i-1]['close']
            curr_close = df.iloc[i]['close']
            prev_low = df.iloc[i-1]['low']
            curr_high = df.iloc[i]['high']

            bb_lower = df.iloc[i-1]['bb_lower']
            bb_upper = df.iloc[i-1]['bb_upper']
            bb_middle = df.iloc[i]['bb_middle']

            # Long: touched lower band and now bouncing up
            if prev_low <= bb_lower and curr_close > prev_close:
                df.at[df.index[i], 'signal'] = 1
                df.at[df.index[i], 'stop_loss'] = curr_close - df.iloc[i]['atr'] * self.atr_stop_multiplier
                df.at[df.index[i], 'take_profit'] = bb_middle

            # Short: touched upper band and now bouncing down
            elif curr_high >= bb_upper and curr_close < prev_close:
                df.at[df.index[i], 'signal'] = -1
                df.at[df.index[i], 'stop_loss'] = curr_close + df.iloc[i]['atr'] * self.atr_stop_multiplier
                df.at[df.index[i], 'take_profit'] = bb_middle

        return df

    def get_description(self):
        """Return strategy description"""
        return f"""
Bollinger Bands Bounce Strategy

Parameters:
- BB Period: {self.bb_period}
- BB Std Dev: {self.bb_std}
- ATR Period: {self.atr_period}
- Stop Loss: {self.atr_stop_multiplier}x ATR

Logic:
- Long when price touches lower band then bounces up
- Short when price touches upper band then bounces down
- Target: Middle band (mean reversion)
- ATR-based stops for risk management
"""


if __name__ == "__main__":
    # Test the strategy
    from src.data_fetcher import FMPDataFetcher

    fetcher = FMPDataFetcher()
    data = fetcher.get_historical_data('EURUSD', timeframe='1hour')

    if not data.empty:
        strategy1 = BollingerMeanReversion()
        signals1 = strategy1.generate_signals(data)

        print(strategy1.get_description())
        print(f"\nGenerated {len(signals1[signals1['signal'] != 0])} signals")
        print(f"Long signals: {len(signals1[signals1['signal'] == 1])}")
        print(f"Short signals: {len(signals1[signals1['signal'] == -1])}")

        print("\n" + "="*60 + "\n")

        strategy2 = BollingerBounce()
        signals2 = strategy2.generate_signals(data)

        print(strategy2.get_description())
        print(f"\nGenerated {len(signals2[signals2['signal'] != 0])} signals")
        print(f"Long signals: {len(signals2[signals2['signal'] == 1])}")
        print(f"Short signals: {len(signals2[signals2['signal'] == -1])}")
