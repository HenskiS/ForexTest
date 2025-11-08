"""
Volatility-Based Adaptive Strategy

Key insight from analysis:
- Tight stops (0.5x/1.5x) work in LOW volatility (2025, pre-COVID)
- Wide stops (1.0x-1.25x/2.0x-2.5x) needed in HIGH volatility (2020-2024)

Use volatility regime to determine parameters:
- Low vol (<0.8x avg) → Aggressive 0.75x/1.5x
- Normal vol (0.8-1.2x) → Baseline 1.0x/2.0x
- High vol (>1.2x) → Defensive 1.25x/2.5x
"""
import pandas as pd
import numpy as np
from strategies.volume_divergence import VolumeDivergence


class VolatilityAdaptive(VolumeDivergence):
    """
    Adapts stop/target based on recent volatility vs historical avg

    This addresses the core problem:
    - 2025 is low volatility → can use tight stops
    - 2020-2024 was high volatility → needed wide stops
    """

    def __init__(self, lookback_period=15, volume_period=25, atr_period=14,
                 vol_lookback=20, vol_historical=100):
        super().__init__(lookback_period, volume_period, atr_period, 1.0, 2.0)
        self.name = "VolatilityAdaptive"
        self.vol_lookback = vol_lookback  # Recent volatility window
        self.vol_historical = vol_historical  # Historical volatility window
        self.current_stop_mult = 1.0
        self.current_target_mult = 2.0
        self.atr_history = []

    def update_trade_history(self, entry_date, direction, outcome):
        """Called after each trade - not used here, but keep for compatibility"""
        pass

    def update_volatility(self, current_atr):
        """Update volatility regime based on recent vs historical ATR"""
        self.atr_history.append(current_atr)

        # Keep history limited
        if len(self.atr_history) > self.vol_historical + 50:
            self.atr_history = self.atr_history[-self.vol_historical - 50:]

        if len(self.atr_history) < self.vol_historical:
            # Not enough history, use baseline
            self.current_stop_mult = 1.0
            self.current_target_mult = 2.0
            return

        # Calculate volatility ratio
        recent_vol = np.mean(self.atr_history[-self.vol_lookback:])
        historical_vol = np.mean(self.atr_history[-self.vol_historical:])

        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0

        # Determine parameters based on volatility regime
        if vol_ratio < 0.8:
            # LOW VOLATILITY → Aggressive (tight stops)
            self.current_stop_mult = 0.75
            self.current_target_mult = 1.5

        elif vol_ratio < 0.95:
            # SLIGHTLY LOW → Moderately aggressive
            self.current_stop_mult = 0.85
            self.current_target_mult = 1.75

        elif vol_ratio < 1.15:
            # NORMAL → Baseline
            self.current_stop_mult = 1.0
            self.current_target_mult = 2.0

        elif vol_ratio < 1.3:
            # SLIGHTLY HIGH → Moderately defensive
            self.current_stop_mult = 1.15
            self.current_target_mult = 2.25

        else:
            # HIGH VOLATILITY → Defensive (wide stops)
            self.current_stop_mult = 1.25
            self.current_target_mult = 2.5

        # Update instance variables
        self.stop_atr_multiplier = self.current_stop_mult
        self.target_atr_multiplier = self.current_target_mult

    def get_stats(self):
        """Get current regime stats"""
        if len(self.atr_history) < self.vol_historical:
            return "Building volatility history..."

        recent_vol = np.mean(self.atr_history[-self.vol_lookback:])
        historical_vol = np.mean(self.atr_history[-self.vol_historical:])
        vol_ratio = recent_vol / historical_vol

        regime = ""
        if vol_ratio < 0.8:
            regime = "LOW VOL"
        elif vol_ratio < 1.15:
            regime = "NORMAL VOL"
        else:
            regime = "HIGH VOL"

        return f"{regime} (ratio={vol_ratio:.2f}x), using {self.current_stop_mult:.2f}x/{self.current_target_mult:.1f}x"
