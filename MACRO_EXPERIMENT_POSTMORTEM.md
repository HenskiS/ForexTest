# Macro Features Experiment - Post-Mortem

## Summary
**Attempted:** Add macro features (Fed rates + VIX) to improve model edge
**Result:** FAILED - Performance collapsed
**Action:** Reverted to technical-only model

## What We Added
- **5 Fed rate features:** rate changes, current level, days since change, cumulative changes
- **6 VIX features:** level, change, moving averages, percentile, regime
- **Total:** 37 features (26 technical + 11 macro)

## Results Comparison

| Metric | Baseline (Tech) | Macro-Enhanced | Change |
|--------|-----------------|----------------|--------|
| **Sharpe Ratio** | 0.676 | 0.059 | -91.2% |
| **Total Return** | +311% | +3.6% | -98.9% |
| **Max Drawdown** | -18.4% | -28.8% | +56.5% worse |
| **Win Rate** | 52.1% | 43.7% | -16.1% |
| **Correlation** | 0.0629 | 0.0411 | -35% |
| **Q4-Q1 Spread** | 7.5 bps | 8.7 bps | +22% (misleading!) |

## What Went Wrong

1. **Calibration metrics mislead:** Q4-Q1 spread improved 22%, looked promising
2. **Overall correlation dropped 35%:** This was the killer
3. **Model got worse at mid-range predictions:** Led to bad trade entries
4. **Macro features added noise, not signal:** No predictive value for EUR/USD

## Key Lesson

**Calibration analysis can be misleading.**

Better extreme separation (Q4 vs Q1) doesn't matter if:
- Overall correlation drops significantly
- Quartile thresholds shift to worse values
- Model starts entering bad trades

**Always backtest before declaring victory.**

## Why Macro Features Failed

1. **VIX coverage sparse:** Only 20% of data (2020+)
2. **Fed rates too slow-moving:** Daily forex too fast for rate change signals
3. **EUR/USD specific factors missing:** Need EU data, not just US macro
4. **Technical patterns sufficient:** Price action already embeds macro information

## Files Reverted

- `train_xgboost_multitarget.py`:
  - Removed 11 macro features from FEATURE_COLS
  - Reverted to standard `df.dropna()` instead of selective NaN handling
  - Back to 26 technical features only

## Files Created (for reference)

- `add_macro_features.py` - Added Fed rate features
- `add_vix_from_fmp.py` - Added VIX features from FMP API
- `backtest_macro_model.py` - Backtest script showing failure
- `quick_backtest_macro.py` - Quick calibration analysis

## What's Next

1. **Stick with technical-only model** (26 features)
2. **Retrain to reproduce baseline** (Sharpe 0.676, +311% return)
3. **Consider alternative improvements:**
   - Better stop-loss optimization
   - Multi-timeframe features (4H, 1H data)
   - Cross-pair momentum (EUR/GBP, EUR/JPY)
   - Time-of-day/day-of-week features
   - Feature selection to remove redundant indicators

## Recommendation

**Technical indicators are sufficient for this strategy.**

If we want to improve beyond Sharpe 0.676, focus on:
- Refining existing features (remove redundancy)
- Better exit strategy optimization
- Multi-timeframe analysis
- NOT adding slow-moving macro features
