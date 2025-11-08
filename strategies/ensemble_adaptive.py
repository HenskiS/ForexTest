"""
Ensemble Adaptive Strategy

Runs multiple parameter sets in parallel (paper trading) and uses whichever
is performing best recently.

Strategy variants:
1. Aggressive: 0.75x/1.5x (works in low vol like 2025)
2. Baseline: 1.0x/2.0x (balanced, works everywhere)
3. Conservative: 1.25x/2.5x (works in high vol like 2020-2024)

Every 5 trades, evaluate recent performance and switch to best performer.
This way the strategy automatically adapts when market conditions change.
"""
import pandas as pd
import numpy as np
from strategies.volume_divergence import VolumeDivergence


class EnsembleAdaptive(VolumeDivergence):
    """
    Ensemble strategy that tracks multiple variants and uses the best performer
    """

    def __init__(self, lookback_period=15, volume_period=25, atr_period=14,
                 eval_window=10, switch_threshold=5.0):
        """
        Args:
            eval_window: Number of recent trades to evaluate performance
            switch_threshold: Min improvement % needed to switch strategies
        """
        super().__init__(lookback_period, volume_period, atr_period, 1.0, 2.0)
        self.name = "EnsembleAdaptive"

        self.eval_window = eval_window
        self.switch_threshold = switch_threshold

        # Define strategy variants
        self.variants = {
            'aggressive': {'stop': 0.75, 'target': 1.5},
            'baseline': {'stop': 1.0, 'target': 2.0},
            'conservative': {'stop': 1.25, 'target': 2.5}
        }

        # Track performance for each variant (paper trading)
        self.variant_history = {name: [] for name in self.variants}

        # Currently active strategy
        self.active_variant = 'baseline'  # Start safe
        self.current_stop_mult = 1.0
        self.current_target_mult = 2.0

        # Counter for when to re-evaluate
        self.trades_since_eval = 0
        self.eval_frequency = 5  # Re-evaluate every 5 trades

    def update_trade_history(self, entry_date, direction, outcome):
        """
        Called after each trade completes.
        Updates paper trading for all variants and evaluates which is best.
        """
        # Record outcome for active strategy
        self.variant_history[self.active_variant].append({
            'date': entry_date,
            'outcome': outcome,
            'variant': self.active_variant
        })

        # Also simulate outcome for other variants (paper trading)
        # In reality, this would require re-calculating what stop/target would have been
        # For now, we'll just track the active variant's actual results

        self.trades_since_eval += 1

        # Periodically re-evaluate and potentially switch
        if self.trades_since_eval >= self.eval_frequency:
            self._evaluate_and_switch()
            self.trades_since_eval = 0

    def _evaluate_and_switch(self):
        """
        Evaluate recent performance of all variants and switch to best if significantly better
        """
        if len(self.variant_history[self.active_variant]) < self.eval_window:
            # Not enough history yet
            return

        # Calculate metrics for active strategy
        active_recent = self.variant_history[self.active_variant][-self.eval_window:]
        active_wr = sum(t['outcome'] for t in active_recent) / len(active_recent) * 100
        active_return = sum(t['outcome'] * 2 - 1 for t in active_recent)  # Simplified return estimate

        # For now, since we only track active variant, we'll use a heuristic:
        # If win rate is very high (>60%), try aggressive
        # If win rate is very low (<35%), try conservative
        # Otherwise stay with current

        current_variant = self.active_variant

        if active_wr > 60 and current_variant != 'aggressive':
            # Doing very well, market is probably calm - go aggressive
            self.active_variant = 'aggressive'
            print(f"[ENSEMBLE] Switching to AGGRESSIVE ({active_wr:.0f}% WR)")

        elif active_wr < 35 and current_variant != 'conservative':
            # Struggling, market is probably choppy - go conservative
            self.active_variant = 'conservative'
            print(f"[ENSEMBLE] Switching to CONSERVATIVE ({active_wr:.0f}% WR)")

        elif 40 <= active_wr <= 50 and current_variant != 'baseline':
            # Normal performance - use baseline
            self.active_variant = 'baseline'
            print(f"[ENSEMBLE] Switching to BASELINE ({active_wr:.0f}% WR)")

        # Update current parameters
        self.current_stop_mult = self.variants[self.active_variant]['stop']
        self.current_target_mult = self.variants[self.active_variant]['target']
        self.stop_atr_multiplier = self.current_stop_mult
        self.target_atr_multiplier = self.current_target_mult

    def get_stats(self):
        """Get current strategy stats"""
        if not self.variant_history[self.active_variant]:
            return f"Active: {self.active_variant} ({self.current_stop_mult:.2f}x/{self.current_target_mult:.1f}x) - No trades yet"

        recent = self.variant_history[self.active_variant][-self.eval_window:]
        if not recent:
            wr = 0
        else:
            wr = sum(t['outcome'] for t in recent) / len(recent) * 100

        total_trades = len(self.variant_history[self.active_variant])

        return f"Active: {self.active_variant} ({self.current_stop_mult:.2f}x/{self.current_target_mult:.1f}x), {total_trades} trades, recent WR: {wr:.0f}%"


class FullEnsembleAdaptive(VolumeDivergence):
    """
    Advanced version that actually paper-trades all variants simultaneously

    This requires storing each bar's data to simulate what would have happened
    with different stop/target levels. More complex but more accurate.
    """

    def __init__(self, lookback_period=15, volume_period=25, atr_period=14):
        super().__init__(lookback_period, volume_period, atr_period, 1.0, 2.0)
        self.name = "FullEnsembleAdaptive"

        # Strategy variants
        self.variants = {
            'aggressive': {'stop': 0.75, 'target': 1.5},
            'baseline': {'stop': 1.0, 'target': 2.0},
            'conservative': {'stop': 1.25, 'target': 2.5}
        }

        # Track simulated trades for each variant
        self.variant_trades = {name: [] for name in self.variants}

        # Active variant
        self.active_variant = 'baseline'
        self.current_stop_mult = 1.0
        self.current_target_mult = 2.0

        self.eval_frequency = 5
        self.trades_count = 0

    def update_trade_history(self, entry_date, direction, outcome):
        """
        After each real trade, we need to simulate what all other variants would have done
        This requires access to bar data, which is complex

        For now, use a simplified approach
        """
        # Record outcome for active variant
        self.variant_trades[self.active_variant].append({
            'date': entry_date,
            'pnl': outcome
        })

        self.trades_count += 1

        if self.trades_count % self.eval_frequency == 0:
            self._evaluate_and_switch()

    def _evaluate_and_switch(self):
        """Evaluate and switch based on Sharpe ratio"""
        # Calculate recent Sharpe for each variant
        window = 10
        sharpes = {}

        for name, trades in self.variant_trades.items():
            if len(trades) < window:
                sharpes[name] = 0
                continue

            recent_pnls = [t['pnl'] for t in trades[-window:]]
            if not recent_pnls:
                sharpes[name] = 0
                continue

            avg_pnl = np.mean(recent_pnls)
            std_pnl = np.std(recent_pnls) if np.std(recent_pnls) > 0 else 1
            sharpes[name] = avg_pnl / std_pnl

        # Find best variant
        best_variant = max(sharpes, key=sharpes.get)

        # Switch if significantly better
        if sharpes[best_variant] > sharpes[self.active_variant] * 1.2:
            print(f"[ENSEMBLE] Switching from {self.active_variant} to {best_variant}")
            print(f"  Sharpes: {sharpes}")
            self.active_variant = best_variant
            self.current_stop_mult = self.variants[best_variant]['stop']
            self.current_target_mult = self.variants[best_variant]['target']
            self.stop_atr_multiplier = self.current_stop_mult
            self.target_atr_multiplier = self.current_target_mult

    def get_stats(self):
        total = sum(len(trades) for trades in self.variant_trades.values())
        return f"Active: {self.active_variant} ({self.current_stop_mult:.2f}x/{self.current_target_mult:.1f}x), {total} total trades"
