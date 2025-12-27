# XGBoost Stop-Loss / Take-Profit Analysis

## Executive Summary

Testing 64 SL/TP combinations (8 levels each) on the XGBoost baseline strategy shows that **adding stop-loss or take-profit does NOT meaningfully improve performance**. The baseline strategy (no SL/TP, fixed 5-day hold) is already near-optimal.

## Test Configuration

- **Model**: XGBoost per-pair models (8 pairs)
- **Entry criteria**: 0.52 confidence threshold
- **Position size**: 2% per trade
- **Hold period**: 5 days (max)
- **SL levels tested**: None, 1%, 1.5%, 2%, 2.5%, 3%, 4%, 5%
- **TP levels tested**: None, 1%, 1.5%, 2%, 2.5%, 3%, 4%, 5%
- **Total combinations**: 64
- **Test period**: 2024-01-01 onwards (~2 years)

## Key Findings

### 1. Baseline Performance

| Metric | Value |
|--------|-------|
| Total Return | 1.89% |
| Annual Return | 0.97% |
| Win Rate | 56.3% |
| Sharpe Ratio | 2.323 |
| Max Drawdown | -0.31% |
| Trades | 694 |

### 2. Stop-Loss ALWAYS Hurts Performance

**Every stop-loss level reduces returns and Sharpe ratio:**

| SL Level | Return | Sharpe | Win Rate | Impact |
|----------|--------|--------|----------|--------|
| None | 1.89% | 2.323 | 56.3% | Baseline |
| 1.0% | 1.45% | 1.938 | 53.2% | -23% return, -17% Sharpe |
| 1.5% | 1.37% | 1.767 | 54.6% | -28% return, -24% Sharpe |
| 2.0% | 1.38% | 1.735 | 55.3% | -27% return, -25% Sharpe |
| 2.5% | 1.64% | 1.998 | 55.9% | -13% return, -14% Sharpe |
| 3.0% | 1.55% | 1.826 | 55.9% | -18% return, -21% Sharpe |
| 4.0% | 1.79% | 2.125 | 56.3% | -5% return, -9% Sharpe |
| 5.0% | 1.70% | 1.950 | 56.3% | -10% return, -16% Sharpe |

**Why SL hurts:**
- Model predicts 5-day directional moves, not intraday/short-term moves
- Temporary adverse price action within 5 days is normal market behavior
- Cutting losses early prevents the position from recovering
- The 0.52 confidence threshold already provides quality control
- Tight SLs (1-3%) are hit frequently, reducing win rate
- Wide SLs (4-5%) still reduce performance, suggesting markets need time to develop

### 3. Take-Profit Has Minimal Impact

**TP provides marginal improvement at best:**

| TP Level | Return | Sharpe | Improvement |
|----------|--------|--------|-------------|
| None | 1.89% | 2.323 | Baseline |
| 1.0% | 1.44% | 2.154 | -24% return (cuts winners too early) |
| 1.5% | 1.53% | 2.105 | -19% return |
| 2.0% | 1.72% | 2.228 | -9% return |
| 2.5% | 1.88% | **2.346** | -1% return, +1% Sharpe ✓ |
| 3.0% | 1.90% | 2.333 | +1% return, +0.4% Sharpe ✓ |
| 4.0% | **1.90%** | 2.324 | +1% return, +0% Sharpe ✓ |
| 5.0% | 1.89% | 2.323 | Same as baseline |

**Why TP barely helps:**
- Model is trained on 3-day forward returns
- 5-day holding period already captures most predicted moves
- TP at 2.5-4% only locks in gains slightly earlier
- Improvement is marginal: 1.90% vs 1.89% (0.5% difference)
- Sharpe improvement is minimal: 2.346 vs 2.323 (1% difference)

### 4. Best Configurations (Negligible Improvement)

| Configuration | Return | Annual | Sharpe | Win Rate | Max DD |
|---------------|--------|--------|--------|----------|--------|
| **Baseline (None/None)** | **1.89%** | 0.97% | **2.323** | 56.3% | -0.31% |
| None SL / 4% TP | **1.90%** | 0.98% | 2.324 | 56.3% | -0.31% |
| None SL / 3% TP | 1.90% | 0.97% | 2.333 | 56.3% | -0.31% |
| None SL / 2.5% TP | 1.88% | 0.97% | **2.346** | 56.5% | -0.31% |

Best improvement: +0.01% return or +0.023 Sharpe (1% improvement)

### 5. Worst Configurations (For Reference)

| Configuration | Return | Sharpe | Impact |
|---------------|--------|--------|--------|
| 1% SL / 1% TP | 1.04% | 1.744 | -45% return, -25% Sharpe |
| 1.5% SL / 1.5% TP | 1.06% | 1.523 | -44% return, -34% Sharpe |
| 2% SL / 1% TP | 1.00% | 1.499 | -47% return, -35% Sharpe |

Tight SL/TP combinations severely damage performance.

## Analysis

### Why the Baseline is Already Optimal

1. **Model design**: XGBoost is trained to predict 3-day directional returns
2. **Hold period match**: 5-day hold captures full predicted move
3. **Confidence filtering**: 0.52 threshold already filters low-quality signals
4. **Diversification**: 8 pairs × 694 trades = sufficient sample size
5. **Natural exits**: Fixed 5-day hold acts as a time-based exit rule

### Market Behavior Insights

1. **Temporary adverse moves are common**: Markets don't move in straight lines
2. **Patience is rewarded**: Allowing 5 days for prediction to play out is optimal
3. **Early profit-taking**: Reduces upside capture without meaningful risk reduction
4. **Risk management**: Already handled by position sizing (2%) and confidence threshold

## Recommendations

### For Live Trading

**Use the baseline strategy (no SL/TP):**
- Enter on 0.52 confidence threshold
- Hold for 5 days
- Exit at close on day 5
- Position size: 2% per trade (or higher based on risk tolerance)

**Do NOT use SL/TP because:**
- Adds complexity without benefit
- Reduces returns by 5-45%
- Requires additional monitoring/infrastructure
- Baseline is already near-optimal

### Optional: Minimal TP (If Desired)

If you prefer having a take-profit for psychological comfort:
- Use 2.5-4% TP only (no stop-loss)
- Expect 0-1% improvement in Sharpe (negligible)
- Accept 0-1% potential reduction in total return
- Understand this adds complexity for minimal gain

**Not recommended** - baseline is simpler and performs just as well.

## Comparison to Position Sizing

**Impact comparison:**

| Change | Return Impact | Sharpe Impact | Complexity |
|--------|---------------|---------------|------------|
| 2% → 100% position size | +7,600% | -4.5% | None |
| Baseline → Best SL/TP | +0.5% | +1% | High |

**Conclusion**: Position sizing has 15,000x more impact than SL/TP optimization.

## Files

- [test_xgb_sl_tp_v2.py](test_xgb_sl_tp_v2.py) - SL/TP testing script
- [sl_tp_comprehensive_results.txt](sl_tp_comprehensive_results.txt) - Full results
- [debug_spread_issue.py](debug_spread_issue.py) - Debugging spread bug
- [test_xgb_position_sizing.py](test_xgb_position_sizing.py) - Position sizing analysis

## Technical Notes

### Bug Fixed During Testing

Initial SL/TP script returned -0.65% instead of +1.89% baseline. Root cause:
- Spread was applied as percentage: `exit_price = price * (1 - spread)`
- Should be absolute: `exit_price = price - spread`
- For JPY pairs with spread=0.02: 2% cost vs 0.018% cost (100x difference!)
- Bug caused massive transaction costs on JPY pairs

Fixed by changing all exit calculations from multiplication to subtraction.

## Conclusion

**The XGBoost baseline strategy does not benefit from stop-loss or take-profit rules.** The fixed 5-day holding period, combined with confidence filtering, is already optimal. Adding SL/TP:
- Reduces returns by 0-47%
- Reduces Sharpe by 0-35%
- Adds implementation complexity
- Provides no meaningful improvement

**Recommendation**: Use the baseline strategy as-is. Focus optimization efforts on:
1. Position sizing (much higher impact)
2. Feature engineering
3. Model improvements
4. Additional pairs/markets

Stop-loss and take-profit are not valuable for this strategy.
