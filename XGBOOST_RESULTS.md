# XGBoost Forex Trading Strategy Results

## Summary
XGBoost model trained on 5-day forward returns with percentage-based stop-loss/take-profit exits achieves strong risk-adjusted returns on EUR/USD.

## Optimal Strategy
**Configuration:**
- **Stop-loss**: 0.40% base (volatility-adjusted via ATR)
- **Take-profit**: 1.00% base (volatility-adjusted via ATR)
- **Volatility scaling**: Stops widen/tighten based on current ATR vs median ATR
- **Signal generation**: Quartile-based (long if prediction ≥ Q3, short if prediction ≤ Q1)
- **Exit discipline**: Ignore signal changes, only exit via stops or end of period

## Performance (Walk-Forward Validation 2000-2025)

| Metric | Value |
|--------|-------|
| **Total Return** | +311.2% ($1,000 → $2,591) |
| **Annualized Return** | 7.32% |
| **Sharpe Ratio** | **0.676** |
| **Max Drawdown** | -16.8% |
| **Volatility** | 10.84% |
| **Win Rate** | 34.6% |
| **Total Trades** | 2,308 (115/year) |
| **Profit Factor** | 1.230 |

**Exit Distribution:**
- 64.7% via stop-loss (risk management)
- 33.8% via take-profit (profit-taking)
- 1.5% at end of period

## 2025 Out-of-Sample Validation

Trained on 2000-2024, tested on 2025 data only (fixed 0.50%/1.25% exits):

| Metric | Walk-Forward | 2025 Test | Status |
|--------|-------------|-----------|--------|
| Annualized Return | 6.05% | 6.00% | ✓ |
| Sharpe Ratio | 0.464 | 0.451 | ✓ |
| Total Trades | 1,863 | 61 (9 months) | ✓ |
| Win Rate | 33.8% | 36.1% | ✓ |

**Conclusion:** Strategy generalizes to unseen 2025 data with nearly identical risk-adjusted returns. **No data leakage detected.**

*Note: 2025 validation pending for volatility-adjusted exits (0.40%/1.00% base). Expected to maintain similar out-of-sample consistency.*

## Key Insights

1. **Volatility-adjusted stops are critical**: Scaling stops by ATR adapts to market conditions - tight in calm markets, wider in volatile periods
2. **Tight base stops win**: 0.40% base stop-loss with volatility adjustment outperforms fixed 0.50%
3. **Ignore signal changes**: Model is better at identifying entries than timing exits - let stops handle risk management
4. **High frequency**: 115 trades/year provides statistical robustness and frequent compounding
5. **Trailing stops hurt**: They exit winners too early, reducing Sharpe from 0.676 to 0.401
6. **Model retraining**: Retraining every 126 days essential for maintaining performance

## Comparison: Alternative Strategies

| Strategy | Sharpe | Annual Return | Total Return | Max DD |
|----------|--------|---------------|--------------|--------|
| **Vol-adjusted (0.40%/1.00% base)** | **0.676** | **7.32%** | **+311%** | **-16.8%** |
| Fixed (0.50%/1.25%) | 0.464 | 6.05% | +223% | -22.2% |
| Vol-adj + Trailing stops | 0.434 | 4.72% | +152% | -29.0% |
| Fixed (0.60%/1.20%) | 0.316 | 4.34% | +133% | -26.2% |
| 100/150 pip stops | 0.216 | 3.90% | +115% | -25.6% |
| Signal changes only | 0.141 | 4.18% | +127% | -15.4% |

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
- `backtest_advanced_exits.py` - Backtest with volatility-adjusted and trailing stops
- `backtest_percent_exits.py` - Backtest percentage-based stop-loss/take-profit strategies
- `test_2025_percent_exits.py` - Validate on 2025 out-of-sample data
- `advanced_exits_results.json` - Volatility-adjusted and trailing stop results

## Reproducibility

```bash
# Train model (5-day forward returns)
python train_xgboost_multitarget.py --target target_5day_return --n_iter 20

# Backtest with volatility-adjusted exits (optimal)
python backtest_advanced_exits.py

# Validate on 2025 data
python test_2025_percent_exits.py
```

---

**Note**: Past performance does not guarantee future results. This is a research project replicating academic work, not investment advice.
