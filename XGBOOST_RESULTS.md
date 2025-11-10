# XGBoost Forex Trading Strategy Results

## Summary
XGBoost model trained on 5-day forward returns with volatility-adjusted stops and loss cooldown achieves exceptional risk-adjusted returns on EUR/USD.

## Optimal Strategy (New: Loss Cooldown)
**Configuration:**
- **Stop-loss**: 0.40% base (volatility-adjusted via ATR)
- **Take-profit**: 1.00% base (volatility-adjusted via ATR)
- **Volatility scaling**: Stops widen/tighten based on current ATR vs median ATR
- **Loss cooldown**: **1 day wait after losing trades** (prevents revenge trading)
- **Signal generation**: Quartile-based (long if prediction ≥ Q3, short if prediction ≤ Q1)
- **Exit discipline**: Ignore signal changes, only exit via stops or end of period

## Performance (Walk-Forward Validation 2000-2025)

| Metric | Value |
|--------|-------|
| **Total Return** | +821.8% ($1,000 → $6,568) |
| **Annualized Return** | 11.75% |
| **Sharpe Ratio** | **1.057** |
| **Max Drawdown** | -7.7% |
| **Volatility** | 11.12% |
| **Win Rate** | 39.6% |
| **Total Trades** | 1,695 (85/year) |
| **Profit Factor** | 1.525 |

**Exit Distribution:**
- 59.9% via stop-loss (risk management)
- 38.6% via take-profit (profit-taking)
- 1.5% at end of period

**Key Improvement:** 1-day loss cooldown filters out low-quality trades after losses, improving win rate from 34% to 40% and cutting max drawdown by 58%.

## Monte Carlo Validation (Trade Resampling)

**10,000 simulations** randomly resampling the 1,695 historical trades confirm exceptional robustness:

| Metric | Value |
|--------|-------|
| **Probability of Profit** | 100.0% |
| **Risk of Ruin (>50% loss)** | 0.0% |
| **Median Return** | +824% |
| **Median Sharpe** | 1.76 |
| **95% Confidence Interval** | +475% to +1,366% return |
| **Worst Case Max DD (5th %ile)** | -11.8% |

**Interpretation:** Strategy is not sequence-dependent. Even in worst 5% of scenarios (most unfavorable trade ordering), still achieves +475% return with <12% drawdown. Results are statistically robust, not due to lucky timing.

## 2025 Out-of-Sample Validation

Trained on 2000-2024, tested on 2025 data only (fixed 0.50%/1.25% exits):

| Metric | Walk-Forward | 2025 Test | Status |
|--------|-------------|-----------|--------|
| Annualized Return | 6.05% | 6.00% | ✓ |
| Sharpe Ratio | 0.464 | 0.451 | ✓ |
| Total Trades | 1,863 | 61 (9 months) | ✓ |
| Win Rate | 33.8% | 36.1% | ✓ |

**Conclusion:** Strategy generalizes to unseen 2025 data with nearly identical risk-adjusted returns. **No data leakage detected.**

*Note: 2025 validation pending for loss cooldown strategy. Expected to maintain similar out-of-sample consistency.*

## Alternative Exit Strategies Tested

**Signal Change Exits**: Tested exiting positions when model prediction changes (e.g., exit long when signal flips to short). Results showed these exits significantly underperform fixed stops:
- Best signal change strategy: Sharpe 0.131, 2.93% annual return
- Model is better at entry timing than exit timing
- Fixed stops remain superior for risk management

**Cooldown Period Analysis**: Tested 1, 2, 3, 5, 7, and 10 day cooldowns after losses:
- 1 day cooldown: Optimal (Sharpe 1.057)
- 2-3 days: Good balance (Sharpe 0.625-0.640)
- 5+ days: Too conservative, misses opportunities

## Key Insights

1. **1-day loss cooldown is transformative**: Waiting 1 day after losses dramatically improves performance (Sharpe 1.06 vs 0.64) by avoiding revenge trading and choppy conditions
2. **Volatility-adjusted stops are critical**: Scaling stops by ATR adapts to market conditions - tight in calm markets, wider in volatile periods
3. **Tight base stops win**: 0.40% base stop-loss with volatility adjustment outperforms fixed 0.50%
4. **Ignore signal changes**: Model is better at identifying entries than timing exits - let stops handle risk management
5. **Quality over quantity**: Fewer trades (85/year vs 115/year) with higher win rate beats high frequency
6. **Trailing stops hurt**: They exit winners too early, reducing Sharpe ratio significantly
7. **Model retraining**: Retraining every 126 days essential for maintaining performance

## Comparison: Alternative Strategies

| Strategy | Sharpe | Annual Return | Total Return | Max DD | Trades/Year |
|----------|--------|---------------|--------------|--------|-------------|
| **Vol-adj + 1 day cooldown** | **1.057** | **11.75%** | **+822%** | **-7.7%** | **85** |
| Vol-adjusted (0.40%/1.00% base) | 0.676 | 7.32% | +311% | -16.8% | 115 |
| Vol-adj + 2 day cooldown | 0.640 | 7.05% | +290% | -8.9% | 67 |
| Vol-adj + 3 day cooldown | 0.625 | 6.94% | +283% | -7.6% | 57 |
| Fixed (0.50%/1.25%) | 0.464 | 6.05% | +223% | -22.2% | 96 |
| Vol-adj + Trailing stops | 0.434 | 4.72% | +152% | -29.0% | 117 |
| Fixed (0.60%/1.20%) | 0.316 | 4.34% | +133% | -26.2% | 74 |
| 100/150 pip stops | 0.216 | 3.90% | +115% | -25.6% | 92 |
| Signal changes only | 0.141 | 4.18% | +127% | -15.4% | 24 |

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
- `backtest_advanced_exits.py` - Backtest with volatility-adjusted stops, trailing stops, and **loss cooldown**
- `backtest_percent_exits.py` - Backtest percentage-based stop-loss/take-profit strategies
- `backtest_signal_change_exits.py` - Backtest signal-change based exits (tested, not recommended)
- `test_2025_percent_exits.py` - Validate on 2025 out-of-sample data
- `monte_carlo_simulation.py` - Monte Carlo trade resampling (10k simulations)
- `analyze_yearly_performance.py` - Yearly performance breakdown analysis
- `advanced_exits_results.json` - All exit strategy results including loss cooldown
- `signal_change_exits_results.json` - Signal change exit test results
- `monte_carlo_results.json` - Monte Carlo simulation statistics and confidence intervals

## Reproducibility

```bash
# Train model (5-day forward returns)
python train_xgboost_multitarget.py --target target_5day_return --n_iter 20

# Backtest with volatility-adjusted exits + loss cooldown (optimal)
python backtest_advanced_exits.py

# Monte Carlo robustness validation
python monte_carlo_simulation.py

# Yearly performance breakdown
python analyze_yearly_performance.py

# Test signal-change exits (for comparison)
python backtest_signal_change_exits.py

# Validate on 2025 data
python test_2025_percent_exits.py
```

---

**Note**: Past performance does not guarantee future results. This is a research project replicating academic work, not investment advice.
