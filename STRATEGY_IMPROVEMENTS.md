# Strategy Improvements Backlog

Potential improvements discovered through backtesting. Not implemented yet - document for future reference.

## Current Production Config
- Thresholds: 10/90 percentile
- Hold time: 5 days uniform
- Stop loss: 2.5%
- Leverage: 2.0x
- Backtest performance: ~98% annual, 4.69 Sharpe, -16% MaxDD

---

## 1. Longer Hold Times (HIGH IMPACT)

**Finding**: Model's directional accuracy improves significantly at longer hold periods.

| Hold Days | Directional Accuracy |
|-----------|---------------------|
| 1 day | 52.98% |
| 5 days | 57.59% |
| 13 days | 65.19% (peak) |
| 24 days | 64.8% |

**Backtest Results** (uniform 13-day hold):
- Annual: 423.7% (vs 98.2% at 5-day)
- Sharpe: 9.39 (vs 4.69)
- MaxDD: -14.5% (vs -16.1%)
- Trades: 5,214 (vs 7,616)

**Implementation**: Change `HOLDING_PERIOD_DAYS` from 5 to 13 in config.

**Risk**: In-sample optimization. 5-day was arbitrary, 13-day is optimized on historical data.

---

## 2. Per-Pair Optimized Hold Times (MEDIUM IMPACT)

**Finding**: Different pairs have different optimal hold periods.

| Pair | Optimal Hold | Stability |
|------|-------------|-----------|
| EURUSD | 24 days | YES |
| GBPUSD | 24 days | YES |
| AUDUSD | 14 days | NO |
| USDJPY | 13 days | NO |
| EURJPY | 13 days | YES |
| USDCAD | 24 days | YES |
| USDCHF | 13 days | NO |
| NZDUSD | 13 days | YES |

**Backtest Results**:
- Annual: 411.2%
- Sharpe: 9.65 (highest)
- Slot utilization: 2.06 (vs 1.07 at 5-day)

**Implementation**: Add `OPTIMAL_HOLD_DAYS` dict to config, modify trader to use per-pair holds.

**Risk**: Only 5/8 pairs show stable optimal holds across time periods. Likely some overfitting.

---

## 3. Signal Strength Filtering (RISK-ADJUSTED)

**Finding**: Signal strength (how far past 10/90 threshold) strongly predicts trade quality.

| Signal Strength | Win Rate | Avg Return |
|----------------|----------|------------|
| Weak (0-33% past threshold) | 56.6% | 0.16% |
| Medium (33-66%) | 73.4% | 0.89% |
| Strong (66-100%) | 86.1% | 1.89% |

**Option A: Filter weak signals**
```
Strong filter (66%) + 22d hold:
- Annual: 132.9%
- Sharpe: 12.39 (BEST)
- MaxDD: -7.6% (BEST)
- Win Rate: 83.5%
- Trades: 1,511
```

**Option B: Tighten thresholds to 5/95**
```
- Annual: 243.4%
- Sharpe: 10.02
- MaxDD: -10.9%
- Trades: 3,374
```

**Trade-off**: Better risk-adjusted returns but lower absolute returns.

**Implementation**:
- Option A: Add `MIN_SIGNAL_STRENGTH = 66` and filter in trader
- Option B: Change thresholds from 10/90 to 5/95

---

## 4. Dynamic Hold Times (NOT RECOMMENDED)

**Finding**: Adjusting hold time based on signal strength doesn't outperform static approaches.

Tested: Weak signals → 7d, Medium → 13d, Strong → 22d

Result: 299.4% annual vs 423.7% for uniform 13-day.

**Conclusion**: Better to either filter signals entirely OR hold everything longer. Dynamic holds add complexity without benefit.

---

## 5. Factors That DON'T Predict Optimal Hold

Analyzed but found no predictive power:
- Volatility (ATR)
- Trend strength (ADX)
- Recent momentum alignment
- Volatility regime (high/low)

Only signal strength showed meaningful correlation (r=0.148).

---

## 6. Stop Loss / Take Profit Optimization (13-Day Hold)

**Finding**: With longer holds, wider stops work better. Take profits can improve risk-adjusted returns.

**Stop Loss Only (no TP):**
| SL % | Annual | Sharpe | MaxDD |
|------|--------|--------|-------|
| 2.5% | 418.8% | 9.00 | -13.1% |
| 3.0% | 437.9% | 9.42 | -12.3% |
| 4.0% | 463.9% | 9.82 | -14.1% |

**With Take Profit:**
| Config | Annual | Sharpe | MaxDD |
|--------|--------|--------|-------|
| 3% SL, no TP | 437.9% | 9.42 | -12.3% |
| 3% SL, 3% TP (1:1) | 405.0% | **10.20** | **-10.4%** |

**Recommended**: 3% SL + 3% TP with 13-day hold
- Best risk-adjusted (10.20 Sharpe)
- Lowest max drawdown (10.4%)
- Still excellent returns (405% annual)

---

## Implementation Priority

1. **Quick win**: Change to 13-day uniform hold (simple, big improvement)
2. **Better risk-adjusted**: Add 3% SL + 3% TP (1:1 ratio)
3. **If wanting lower risk**: Tighten to 5/95 thresholds
4. **If wanting lowest drawdown**: Filter for strong signals only + 22d hold
5. **Skip**: Dynamic hold times, per-pair optimization, signal-based exits

---

## Testing Notes

All backtests used:
- 4,500 test days (~12 years)
- 8-pair portfolio
- 2.5% stop loss
- Averaged stops (OANDA netting reality)
- Spread costs included

Scripts created:
- `test_directional_accuracy.py` - accuracy by hold period
- `optimize_hold_times.py` - per-pair optimization
- `test_optimized_hold_times.py` - backtest per-pair holds
- `analyze_hold_time_factors.py` - factor analysis
- `test_dynamic_hold_times.py` - dynamic hold strategy
- `test_signal_filter.py` - signal filtering strategies
- `test_sl_tp_13day.py` - SL/TP optimization for 13-day holds
