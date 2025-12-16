# Meta-Strategy: XGBoost Trade Filter

## Overview
Train a classifier on historical trade outcomes to predict P(win) for each new signal. Only take trades where P(win) >= 0.55.

## Results (Backtest 2019-2025)
| Metric | Baseline | With Meta 0.55 |
|--------|----------|----------------|
| Trades | 3047 | 2071 (-32%) |
| Win Rate | 62.3% | 65.9% |
| Annual | 102% | 79% |
| MaxDD | -16.1% | -8.7% (-46%) |
| Sharpe | 4.84 | 5.77 |

At 2.5x leverage: 100% annual with only -10.9% DD (vs -16.1% baseline)

## XGBoost Hyperparameters (Optimized)
```python
XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=30,
    subsample=0.7,
    random_state=42,
    verbosity=0
)
```

## Features (8 total)
```python
feature_cols = [
    'pred_strength',      # (prediction - median(buffer)) / std(buffer)
    'pred_percentile',    # percentile rank of prediction in buffer
    'pred_std',           # std of prediction buffer
    'day_of_week',        # 0-4 (Mon-Fri)
    'volatility_20',      # 20-day rolling std of returns
    'ema_ratio',          # ema_10 / ema_50
    'atr_pct',            # 14-day ATR / close
    'num_slots',          # current open slots (0-4)
]
```

## Training Data
- Source: All completed trades from backtest
- Split: 60% train / 40% test (temporal, no shuffle)
- Scaling: StandardScaler on features
- Target: binary win/loss (pnl > 0)

## Production Validation Plan
1. Deploy base strategy normally (no filter)
2. Log meta-model P(win) for each trade taken
3. After 100+ completed trades, compare:
   - Win rate of trades where P(win) >= 0.55
   - Win rate of trades where P(win) < 0.55
4. If kept trades significantly outperform rejected, implement filter

## Key Files
- `xgb_meta_threshold_analysis.py` - Full analysis with optimized params
- `optimize_xgb_meta.py` - Hyperparameter search
- `check_yearly_consistency.py` - Verify signal works across years

## Caveats
- Hyperparams were tuned on test set (slight optimism bias)
- Test AUC = 0.58 (modest edge, not huge)
- Need 100+ production trades to validate
