"""
Simpler Adaptive Strategy

Instead of complex regime detection, use simple rules:
1. After 3+ wins → Tighten stops to 0.75x/1.5x
2. After 2+ losses → Widen stops to 1.25x/2.5x
3. Otherwise → Baseline 1.0x/2.0x

Works by adjusting stop/target AT ENTRY TIME based on recent trade history
"""
import pandas as pd
import numpy as np
from strategies.volume_divergence import VolumeDivergence


class SimpleAdaptive(VolumeDivergence):
    """
    Volume Divergence with simple win/loss streak adaptation

    Adjusts stops/targets based on last N trades:
    - Win streak (3+) → Aggressive (0.75x/1.5x)
    - Loss streak (2+) → Defensive (1.25x/2.5x)
    - Mixed → Baseline (1.0x/2.0x)
    """

    def __init__(self, lookback_period=15, volume_period=25, atr_period=14):
        super().__init__(lookback_period, volume_period, atr_period, 1.0, 2.0)
        self.name = "SimpleAdaptive"
        self.trade_outcomes = []  # List of 1 (win) or 0 (loss)
        self.current_stop_mult = 1.0
        self.current_target_mult = 2.0

    def update_trade_history(self, entry_date, direction, outcome):
        """Update trade history and recalculate parameters"""
        self.trade_outcomes.append(outcome)

        # Keep last 10 trades
        if len(self.trade_outcomes) > 10:
            self.trade_outcomes = self.trade_outcomes[-10:]

        # Recalculate parameters based on recent performance
        self._update_parameters()

    def _update_parameters(self):
        """Update stop/target multipliers based on recent trades"""
        if len(self.trade_outcomes) < 2:
            # Not enough history, use baseline
            self.current_stop_mult = 1.0
            self.current_target_mult = 2.0
            return

        # Check for streaks
        current_streak = 1
        last_outcome = self.trade_outcomes[-1]

        for i in range(len(self.trade_outcomes) - 2, -1, -1):
            if self.trade_outcomes[i] == last_outcome:
                current_streak += 1
            else:
                break

        # Win streak (3+) → Go aggressive
        if last_outcome == 1 and current_streak >= 3:
            self.current_stop_mult = 0.75
            self.current_target_mult = 1.5

        # Loss streak (2+) → Go defensive
        elif last_outcome == 0 and current_streak >= 2:
            self.current_stop_mult = 1.25
            self.current_target_mult = 2.5

        # Mixed or short streak → Baseline
        else:
            # Check overall recent win rate
            recent_wr = sum(self.trade_outcomes[-5:]) / min(len(self.trade_outcomes), 5)

            if recent_wr >= 0.6:
                # Doing well → Slightly aggressive
                self.current_stop_mult = 0.85
                self.current_target_mult = 1.75
            elif recent_wr <= 0.3:
                # Struggling → Slightly defensive
                self.current_stop_mult = 1.15
                self.current_target_mult = 2.25
            else:
                # Neutral → Baseline
                self.current_stop_mult = 1.0
                self.current_target_mult = 2.0

        # Update instance variables so backtester uses current values
        self.stop_atr_multiplier = self.current_stop_mult
        self.target_atr_multiplier = self.current_target_mult

    def get_stats(self):
        """Get current adaptation stats"""
        if len(self.trade_outcomes) == 0:
            return "No trades yet"

        recent_wr = sum(self.trade_outcomes) / len(self.trade_outcomes) * 100
        return f"{len(self.trade_outcomes)} trades, {recent_wr:.0f}% WR, using {self.current_stop_mult:.2f}x/{self.current_target_mult:.1f}x"
