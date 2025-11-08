"""
Adaptive Volume Divergence Strategy

Adjusts stop/target parameters based on recent trade performance:
- Start conservative (1.0x/2.0x)
- After winning streak → Tighten stops to capture more opportunities
- After losing streak → Widen stops to avoid choppy markets
- After period of low volatility → More aggressive
- After period of high volatility → More defensive

This addresses the 2025 problem: could have made 40% instead of 9%
"""
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from strategies.volume_divergence import VolumeDivergence


class AdaptiveVolumeDivergence(VolumeDivergence):
    """
    Volume Divergence with adaptive stop/target based on recent performance

    Adaptation rules:
    1. Win streak (3+ wins) → Tighten stops (0.75x/1.5x) to catch more trades
    2. Loss streak (2+ losses) → Widen stops (1.25x/2.5x) to avoid chop
    3. Low volatility regime → Aggressive (0.75x/1.5x)
    4. High volatility regime → Conservative (1.25x/2.0x)
    5. Mixed results → Baseline (1.0x/2.0x)
    """

    def __init__(self, lookback_period=15, volume_period=25, atr_period=14,
                 base_stop=1.0, base_target=2.0,
                 enable_streak_adaptation=True,
                 enable_volatility_adaptation=True,
                 win_streak_threshold=3,
                 loss_streak_threshold=2,
                 lookback_trades=10):
        """
        Args:
            base_stop: Default stop multiplier (1.0)
            base_target: Default target multiplier (2.0)
            enable_streak_adaptation: Adjust based on win/loss streaks
            enable_volatility_adaptation: Adjust based on market volatility
            win_streak_threshold: Wins needed to go aggressive
            loss_streak_threshold: Losses needed to go defensive
            lookback_trades: How many recent trades to consider
        """
        super().__init__(lookback_period, volume_period, atr_period, base_stop, base_target)

        self.base_stop = base_stop
        self.base_target = base_target
        self.enable_streak_adaptation = enable_streak_adaptation
        self.enable_volatility_adaptation = enable_volatility_adaptation
        self.win_streak_threshold = win_streak_threshold
        self.loss_streak_threshold = loss_streak_threshold
        self.lookback_trades = lookback_trades

        self.name = "AdaptiveVolDiv"

        # Track trade history for adaptation
        self.trade_history = []  # List of (entry_date, direction, outcome: 1=win, 0=loss)

    def get_current_regime(self, df, current_idx):
        """
        Determine current market regime based on recent trades and volatility

        Returns:
            tuple: (stop_multiplier, target_multiplier, regime_name)
        """
        # Default to baseline
        stop = self.base_stop
        target = self.base_target
        regime = "baseline"

        if len(self.trade_history) < 3:
            return stop, target, regime

        # Get recent trade outcomes
        recent_trades = self.trade_history[-self.lookback_trades:]
        recent_outcomes = [t[2] for t in recent_trades]

        # Streak adaptation
        if self.enable_streak_adaptation:
            # Count current streak
            current_streak = 0
            streak_type = None  # 'win' or 'loss'

            for outcome in reversed(recent_outcomes):
                if streak_type is None:
                    streak_type = 'win' if outcome == 1 else 'loss'
                    current_streak = 1
                elif (streak_type == 'win' and outcome == 1) or (streak_type == 'loss' and outcome == 0):
                    current_streak += 1
                else:
                    break

            # Win streak → Go aggressive (tighter stops, more trades)
            if streak_type == 'win' and current_streak >= self.win_streak_threshold:
                stop = 0.75
                target = 1.5
                regime = f"aggressive (win_streak_{current_streak})"

            # Loss streak → Go defensive (wider stops, fewer trades)
            elif streak_type == 'loss' and current_streak >= self.loss_streak_threshold:
                stop = 1.25
                target = 2.5
                regime = f"defensive (loss_streak_{current_streak})"

            # Mixed recent performance → Check win rate
            else:
                recent_win_rate = sum(recent_outcomes) / len(recent_outcomes)

                if recent_win_rate >= 0.6:
                    # Doing well overall → Slightly aggressive
                    stop = 0.85
                    target = 1.75
                    regime = f"positive ({recent_win_rate:.0%} WR)"
                elif recent_win_rate <= 0.3:
                    # Struggling overall → Defensive
                    stop = 1.15
                    target = 2.25
                    regime = f"negative ({recent_win_rate:.0%} WR)"

        # Volatility adaptation
        if self.enable_volatility_adaptation and current_idx >= 50:
            # Compare recent ATR to historical ATR
            recent_atr = df.loc[current_idx-10:current_idx, 'atr'].mean()
            historical_atr = df.loc[max(0, current_idx-50):current_idx, 'atr'].mean()

            volatility_ratio = recent_atr / historical_atr if historical_atr > 0 else 1.0

            # Low volatility → Can use tighter stops
            if volatility_ratio < 0.8:
                # Override to aggressive if volatility is low
                if stop > 0.75:  # Don't make already-aggressive more aggressive
                    stop = 0.75
                    target = 1.5
                    regime = f"low_vol ({volatility_ratio:.2f}x)"

            # High volatility → Use wider stops
            elif volatility_ratio > 1.3:
                if stop < 1.25:  # Don't make already-defensive more defensive
                    stop = 1.25
                    target = 2.0
                    regime = f"high_vol ({volatility_ratio:.2f}x)"

        return stop, target, regime

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals with adaptive stop/target levels
        """
        df = data.copy()

        # Calculate base indicators (ATR, highs/lows, volume)
        df['atr'] = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.atr_period
        ).average_true_range()

        df['highest_close'] = df['close'].rolling(window=self.lookback_period).max()
        df['lowest_close'] = df['close'].rolling(window=self.lookback_period).min()
        df['prev_highest'] = df['highest_close'].shift(1)
        df['prev_lowest'] = df['lowest_close'].shift(1)
        df['avg_volume'] = df['volume'].rolling(window=self.volume_period).mean()

        # Initialize columns
        df['signal'] = 0
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        df['regime'] = 'baseline'
        df['stop_mult'] = self.base_stop
        df['target_mult'] = self.base_target

        # Generate signals row by row with adaptive parameters
        for idx in range(len(df)):
            if idx < max(self.lookback_period, self.volume_period, self.atr_period):
                continue

            row = df.iloc[idx]

            # Get adaptive parameters based on current regime
            stop_mult, target_mult, regime = self.get_current_regime(df, idx)

            df.at[df.index[idx], 'regime'] = regime
            df.at[df.index[idx], 'stop_mult'] = stop_mult
            df.at[df.index[idx], 'target_mult'] = target_mult

            # Check for signals (same logic as base strategy)
            short_condition = (
                row['close'] >= row['prev_highest'] and
                row['volume'] < row['avg_volume'] and
                row['atr'] > 0 and
                row['avg_volume'] > 0
            )

            long_condition = (
                row['close'] <= row['prev_lowest'] and
                row['volume'] < row['avg_volume'] and
                row['atr'] > 0 and
                row['avg_volume'] > 0
            )

            if short_condition:
                df.at[df.index[idx], 'signal'] = -1
                df.at[df.index[idx], 'stop_loss'] = row['close'] + row['atr'] * stop_mult
                df.at[df.index[idx], 'take_profit'] = row['close'] - row['atr'] * target_mult

            elif long_condition:
                df.at[df.index[idx], 'signal'] = 1
                df.at[df.index[idx], 'stop_loss'] = row['close'] - row['atr'] * stop_mult
                df.at[df.index[idx], 'take_profit'] = row['close'] + row['atr'] * target_mult

        return df

    def update_trade_history(self, entry_date, direction, outcome):
        """
        Update trade history for adaptation

        Args:
            entry_date: Date of trade entry
            direction: 'long' or 'short'
            outcome: 1 for win, 0 for loss
        """
        self.trade_history.append((entry_date, direction, outcome))

        # Keep only recent trades
        if len(self.trade_history) > self.lookback_trades * 2:
            self.trade_history = self.trade_history[-self.lookback_trades * 2:]

    def get_description(self):
        """Return strategy description"""
        return f"""
Adaptive Volume Divergence Strategy

Base Parameters:
- Lookback: {self.lookback_period} bars
- Volume Period: {self.volume_period} bars
- ATR Period: {self.atr_period}
- Base Stop: {self.base_stop}x ATR
- Base Target: {self.base_target}x ATR

Adaptation Rules:
1. Win Streak ({self.win_streak_threshold}+ wins) → Aggressive: 0.75x/1.5x
2. Loss Streak ({self.loss_streak_threshold}+ losses) → Defensive: 1.25x/2.5x
3. Low Volatility (<0.8x avg) → Aggressive: 0.75x/1.5x
4. High Volatility (>1.3x avg) → Defensive: 1.25x/2.0x
5. Mixed Performance → Baseline: {self.base_stop}x/{self.base_target}x

Goal: Capture 40% gains in calm markets (2025) while surviving volatile markets (2020-2024)
"""


if __name__ == "__main__":
    print("Adaptive Volume Divergence Strategy")
    print("="*80)
    strategy = AdaptiveVolumeDivergence()
    print(strategy.get_description())
