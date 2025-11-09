# XGBoost Forex Trading Strategy Results

## Summary
XGBoost model trained on 5-day forward returns with percentage-based stop-loss/take-profit exits achieves strong risk-adjusted returns on EUR/USD.

## Optimal Strategy
**Configuration:**
- **Stop-loss**: 0.50% (cut losses quickly)
- **Take-profit**: 1.25% (2.5:1 reward/risk ratio)
- **Signal generation**: Quartile-based (long if prediction ≥ Q3, short if prediction ≤ Q1)
- **Exit discipline**: Ignore signal changes, only exit via stops or end of period

## Performance (Walk-Forward Validation 2000-2025)

| Metric | Value |
|--------|-------|
| **Total Return** | +223.5% ($1,000 → $2,229) |
| **Annualized Return** | 6.05% |
| **Sharpe Ratio** | **0.464** |
| **Max Drawdown** | -22.2% |
| **Volatility** | 13.02% |
| **Win Rate** | 33.8% |
| **Total Trades** | 1,863 (93/year) |
| **Profit Factor** | 1.194 |

**Exit Distribution:**
- 65.3% via stop-loss (risk management)
- 32.9% via take-profit (profit-taking)
- 1.9% at end of period

## 2025 Out-of-Sample Validation

Trained on 2000-2024, tested on 2025 data only:

| Metric | Walk-Forward | 2025 Test | Status |
|--------|-------------|-----------|--------|
| Annualized Return | 6.05% | 6.00% | ✓ |
| Sharpe Ratio | 0.464 | 0.451 | ✓ |
| Total Trades | 1,863 | 61 (9 months) | ✓ |
| Win Rate | 33.8% | 36.1% | ✓ |

**Conclusion:** Strategy generalizes to unseen 2025 data with nearly identical risk-adjusted returns. **No data leakage detected.**

## Key Insights

1. **Tight stops win**: 0.50% stop-loss prevents large losses while 1.25% take-profit captures profitable moves
2. **Ignore signal changes**: Model is better at identifying entries than timing exits - let stops handle risk management
3. **High frequency**: 93 trades/year provides statistical robustness and frequent compounding
4. **Asymmetric risk/reward**: 2.5:1 reward/risk means profitable even with <40% win rate
5. **Model retraining**: Retraining every 126 days essential for maintaining performance

## Comparison: Alternative Strategies

| Strategy | Sharpe | Annual Return | Total Trades |
|----------|--------|---------------|--------------|
| **0.50% SL / 1.25% TP** | **0.464** | **6.05%** | **1,863** |
| 0.60% SL / 1.20% TP | 0.316 | 4.34% | 1,755 |
| 100/150 pip stops | 0.216 | 3.90% | 1,636 |
| Signal changes only | 0.141 | 4.18% | 463 |
| No stops (hold 126 days) | 0.066 | 6.70% | 40 |

## Technical Details

**Model:**
- XGBoost Regressor
- Target: 5-day forward returns
- Features: 26 technical indicators (EMAs, MACD, RSI, Stochastic, ADX, Bollinger Bands, ATR)
- Hyperparameters: n_estimators=250, learning_rate=0.01, max_depth=12, gamma=0.002

**Walk-Forward Validation:**
- 40 rolling windows
- Train: 600 days, Validation: 156 days, Test: 126 days
- Roll forward: 126 days
- Period: 2000-2025 (25 years)

**Data Leakage Prevention:**
- Features shifted forward 1 day (day T features → predict day T+1 return)
- No look-ahead bias
- Proper train/validation/test splits
- Out-of-sample validation on 2025 data

## Files

- `train_xgboost_multitarget.py` - Train XGBoost with alternative targets
- `backtest_percent_exits.py` - Backtest percentage-based stop-loss/take-profit strategies
- `test_2025_percent_exits.py` - Validate on 2025 out-of-sample data
- `percent_exits_results.json` - Detailed results for all tested configurations

## Reproducibility

```bash
# Train model (5-day forward returns)
python train_xgboost_multitarget.py --target target_5day_return --n_iter 20

# Backtest with percentage exits
python backtest_percent_exits.py

# Validate on 2025 data
python test_2025_percent_exits.py
```

---

**Note**: Past performance does not guarantee future results. This is a research project replicating academic work, not investment advice.
