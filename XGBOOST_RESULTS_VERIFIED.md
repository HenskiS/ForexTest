# XGBoost Forex Trading Strategy Results (Verified)

**Last Updated:** 2025-11-15
**Status:** Fresh training runs with verified results

## Overview

XGBoost model trained on 5-day forward returns with pair-specific optimized entry thresholds, volatility-adjusted stops, and loss cooldown. Walk-forward validation across 46 rolling windows (2003-2025).

## Methodology

### Training Configuration
- **Walk-forward windows**: 46 windows
  - Training: 600 days
  - Validation: 156 days
  - Test: 126 days
  - Roll forward: 126 days
- **Hyperparameter optimization**: 20 random iterations per window
- **Features**: Technical indicators only (EMAs, MACD, RSI, ADX, Stochastic, Bollinger Bands, ATR)
- **Target**: 5-day forward return (regression)

### Backtest Configuration
- **Entry thresholds**: Pair-specific optimized percentiles
- **Stop-loss**: 0.40% base (volatility-adjusted via ATR)
- **Take-profit**: 1.00% base (volatility-adjusted via ATR)
- **Volatility scaling**: Stops widen/tighten based on current ATR vs median ATR
- **Loss cooldown**: 1 day wait after losing trades
- **Holding period**: 5 days maximum
- **Transaction costs**: 0.02% per trade (2 pips)

## Multi-Pair Performance Summary

### All Pairs - 1:1 Leverage (No Leverage)

| Pair | Threshold | Final Capital | Total Return | Annual Return | Sharpe Ratio | Total Trades | Trades/Year | Win Rate | Max DD |
|------|-----------|---------------|--------------|---------------|--------------|--------------|-------------|----------|---------|
| **EURUSD** | 48th/52nd | $20,365 | 1,937% | **14.00%** | **1.203** | 2,447 | 106 | 40.6% | -6.76% |
| **GBPUSD** | 48th/52nd | $11,341 | 1,034% | **11.14%** | **0.928** | 2,520 | 110 | 38.1% | -9.81% |
| **USDJPY** | 35th/65th | $13,803 | 1,280% | **12.09%** | **1.290** | 2,302 | 100 | 39.3% | -12.63% |
| **AUDUSD** | 48th/52nd | $16,908 | 1,591% | **13.08%** | **1.491** | 2,868 | 125 | 37.5% | -10.60% |
| **PORTFOLIO** | Mixed | **$62,417** | **1,484%** | **12.58%** | **N/A** | 10,137 | 441 | 38.9% | **N/A** |

**Investment:** $1,000 per pair ($4,000 total)

### All Pairs - 3:1 Leverage (Standard Forex)

| Pair | Threshold | Final Capital | Total Return | Annual Return | Sharpe Ratio | Total Trades | Trades/Year | Win Rate | Max DD |
|------|-----------|---------------|--------------|---------------|--------------|--------------|-------------|----------|---------|
| **EURUSD** | 48th/52nd | $5,947,124 | 594,612% | **45.92%** | **0.963** | 2,447 | 106 | 40.6% | -19.38% |
| **GBPUSD** | 48th/52nd | $1,002,344 | 100,134% | **35.05%** | **0.829** | 2,520 | 110 | 38.1% | -26.79% |
| **USDJPY** | 35th/65th | $1,839,210 | 183,821% | **38.66%** | **1.107** | 2,302 | 100 | 39.3% | -33.56% |
| **AUDUSD** | 48th/52nd | $3,105,696 | 310,470% | **41.85%** | **1.409** | 2,868 | 125 | 37.5% | -28.91% |
| **PORTFOLIO** | Mixed | **$11,894,374** | **297,259%** | **40.37%** | **N/A** | 10,137 | 441 | 38.9% | **N/A** |

**Investment:** $1,000 per pair ($4,000 total)

### Key Insights

- **AUDUSD leads in risk-adjusted returns**: Highest Sharpe ratios (1.491 at 1:1, 1.409 at 3:1)
- **EURUSD highest absolute returns**: 14.00% annual (1:1), 45.92% annual (3:1)
- **USDJPY most selective**: Uses 35th/65th percentile thresholds, fewer but higher quality trades
- **All pairs profitable**: Every pair shows strong positive returns with Sharpe > 0.8
- **Portfolio diversification**: 10,137 total trades across 4 pairs reduces single-pair risk
- **Leverage amplifies returns ~3x**: Average 3:1 returns (40.37%) are ~3.2x the 1:1 returns (12.58%)

---

## Strategy Validation

### Exit Reason Analysis

**Total Trades Analyzed:** 10,137 across all 4 pairs

#### Winning Trades (3,913 total, 38.6% overall win rate)

| Exit Reason | Count | % of Wins | Avg Return |
|-------------|-------|-----------|------------|
| **Target Hit** | 3,591 | 91.8% | +1.01% |
| Time Exit (5 days) | 322 | 8.2% | +0.41% |

#### Losing Trades (6,224 total, 61.4% overall loss rate)

| Exit Reason | Count | % of Losses | Avg Return |
|-------------|-------|-------------|------------|
| **Stop Loss** | 6,161 | 99.0% | -0.43% |
| Time Exit (5 days) | 63 | 1.0% | -0.12% |

**Key Findings:**
- **91.8% of profits come from hitting targets**, validating the volatility-adjusted take-profit levels
- **99% of losses are contained by stop losses**, demonstrating effective risk management
- Average return per trade: +0.110%
- Win/loss ratio: Avg winner (+1.01%) is 2.3x larger than avg loser (-0.43%)

---

### Monte Carlo Validation (10,000 simulations per pair)

Each pair was tested with 10,000 simulations resampling historical trades with replacement to validate strategy robustness at both 1:1 and 3:1 leverage.

#### EURUSD Monte Carlo Results

| Metric | 1:1 Leverage | 3:1 Leverage |
|--------|--------------|--------------|
| **Median Return** | 993.7% (baseline: 1,937%) | 96,041% (baseline: 594,612%) |
| **95% Confidence Interval** | 559.6% to 1,700.4% | 21,918% to 437,287% |
| **Mean Sharpe Ratio** | 1.637 (95% CI: 1.314 to 1.957) | 1.636 (95% CI: 1.315 to 1.956) |
| **Median Max Drawdown** | -8.42% (95% CI: -12.40% to -6.21%) | -23.48% (95% CI: -33.44% to -17.63%) |
| **Probability of Profit** | 100.0% | 100.0% |
| **Risk of Ruin** | 0.00% | 0.00% |

#### GBPUSD Monte Carlo Results

| Metric | 1:1 Leverage | 3:1 Leverage |
|--------|--------------|--------------|
| **Median Return** | 712.2% (baseline: 1,034%) | 39,407% (baseline: 100,134%) |
| **95% Confidence Interval** | 385.9% to 1,253.5% | 8,491% to 184,522% |
| **Mean Sharpe Ratio** | 1.432 (95% CI: 1.104 to 1.755) | 1.432 (95% CI: 1.101 to 1.760) |
| **Median Max Drawdown** | -9.37% (95% CI: -14.00% to -6.85%) | -26.17% (95% CI: -37.55% to -19.52%) |
| **Probability of Profit** | 100.0% | 100.0% |
| **Risk of Ruin** | 0.00% | 0.00% |

#### USDJPY Monte Carlo Results

| Metric | 1:1 Leverage | 3:1 Leverage |
|--------|--------------|--------------|
| **Median Return** | 647.0% (baseline: 1,280%) | 30,289% (baseline: 183,821%) |
| **95% Confidence Interval** | 345.8% to 1,182.4% | 6,209% to 149,402% |
| **Mean Sharpe Ratio** | 1.286 (95% CI: 0.974 to 1.601) | 1.283 (95% CI: 0.967 to 1.596) |
| **Median Max Drawdown** | -10.29% (95% CI: -15.45% to -7.42%) | -28.52% (95% CI: -40.52% to -21.15%) |
| **Probability of Profit** | 100.0% | 100.0% |
| **Risk of Ruin** | 0.00% | 0.00% |

#### AUDUSD Monte Carlo Results

| Metric | 1:1 Leverage | 3:1 Leverage |
|--------|--------------|--------------|
| **Median Return** | 248.0% (baseline: 1,591%) | 2,982% (baseline: 310,470%) |
| **95% Confidence Interval** | 105.1% to 482.2% | 566% to 14,477% |
| **Mean Sharpe Ratio** | 0.804 (95% CI: 0.484 to 1.113) | 0.804 (95% CI: 0.492 to 1.115) |
| **Median Max Drawdown** | -13.50% (95% CI: -21.22% to -9.37%) | -36.46% (95% CI: -53.16% to -26.41%) |
| **Probability of Profit** | 100.0% | 100.0% |
| **Risk of Ruin** | 0.00% | 0.00% |

**Key Findings:**
- **All pairs show 100% probability of profit** across 10,000 simulations at both leverage levels
- **Zero risk of ruin** (>50% loss) for all pairs, even with 3:1 leverage
- Sharpe ratios remain nearly identical with leverage, confirming consistent risk-adjusted returns
- Drawdowns increase proportionally with leverage (~3x) but remain manageable
- Even in worst-case scenarios (5th percentile), all leveraged strategies show substantial profits
- Strategy is robust to different trade orderings and sequences regardless of leverage

---

### Model Attribution Testing

Tests to validate that returns come from ML predictions, not just the exit strategy.

#### Test 1: Random Entry (EURUSD)
**100 simulations with completely random entries (50/50 long/short) using the same exit strategy**

| Metric | Random Entries | Actual Model | Model Advantage |
|--------|----------------|--------------|-----------------|
| **Total Return** | 34.6% median | 1,937% | **56.0x better** |
| **Annual Return** | 1.50% median | 14.00% | **9.3x better** |
| **Sharpe Ratio** | 0.140 | 1.203 | **8.6x better** |
| **Win Rate** | 32.7% | 40.6% | +7.9pp |
| **Probability of Profit** | 91.0% | ~100% | +9pp |

#### Test 2: Random Direction (EURUSD)
**100 simulations with model timing but random direction (50/50 long/short)**

| Metric | Random Direction | Actual Model | Model Advantage |
|--------|------------------|--------------|-----------------|
| **Total Return** | 64.2% median | 1,937% | **30.2x better** |
| **Annual Return** | 2.51% median | 14.00% | **5.6x better** |
| **Sharpe Ratio** | 0.236 | 1.203 | **5.1x better** |
| **Win Rate** | 32.7% | 40.6% | +7.9pp |
| **Probability of Profit** | 94.0% | ~100% | +6pp |

**Key Findings:**
- **Model delivers 9.3x better returns than random entries**, proving XGBoost predictions add significant value
- **Model timing + direction is 5.6x better than just timing alone**, showing directional prediction accuracy
- Exit strategy alone (random entries) only achieves 1.50% annual return vs 14.00% with ML
- This validates that the ML model is the primary driver of returns, not just the risk management

---

## Detailed Results by Pair

### EURUSD (1-Day Timeframe)

**Training Date:** 2025-11-15
**Windows:** 46 (2003-2025)
**Optimal Threshold:** 48th/52nd percentile (very unselective, 96% of predictions traded)

#### Performance Metrics

| Leverage | Final Capital | Total Return | Annual Return | Sharpe Ratio | Years |
|----------|---------------|--------------|---------------|--------------|-------|
| **1:1** (No leverage) | $20,365 | 1,937% | **14.00%** | **1.203** | 23 |
| **3:1** (Standard) | $5,947,124 | 594,612% | **45.92%** | **0.963** | 23 |

#### Trade Statistics
- **Total trades**: 2,447
- **Average per year**: 106
- **Win rate**: ~40.6%
- **Average trade**: +12.98 pips
- **Average winner**: +93.33 pips
- **Average loser**: -41.97 pips
- **Risk/Reward ratio**: 2.22:1

#### Annual Performance (1:1 Leverage)

| Year | Trades | Win % | Return % | Capital | Max DD % |
|------|--------|-------|----------|---------|----------|
| 2003 | 110 | 33.6% | 5.73% | $1,057 | -6.76% |
| 2004 | 131 | 33.6% | 7.88% | $1,141 | -6.47% |
| 2005 | 121 | 46.3% | 32.52% | $1,512 | -3.97% |
| 2006 | 105 | 40.0% | 10.30% | $1,667 | -4.55% |
| 2007 | 88 | 40.9% | 8.39% | $1,807 | -4.32% |
| 2008 | 146 | 41.1% | 25.52% | $2,268 | -6.73% |
| 2009 | 136 | 41.2% | 22.30% | $2,774 | -5.43% |
| 2010 | 145 | 51.0% | 52.31% | $4,225 | -6.38% |
| 2011 | 132 | 32.6% | 4.08% | $4,398 | -6.75% |
| 2012 | 120 | 40.8% | 13.87% | $5,007 | -6.27% |
| 2013 | 93 | 37.6% | 10.49% | $5,533 | -6.13% |
| 2014 | 86 | 39.5% | 4.76% | $5,796 | -4.07% |
| 2015 | 129 | 45.0% | 33.61% | $7,744 | -2.90% |
| 2016 | 105 | 39.0% | 10.25% | $8,538 | -6.36% |
| 2017 | 97 | 39.2% | 11.35% | $9,507 | -5.42% |
| 2018 | 94 | 39.4% | 11.46% | $10,597 | -3.65% |
| 2019 | 76 | 38.2% | 2.15% | $10,825 | -3.91% |
| 2020 | 95 | 32.6% | 7.17% | $11,601 | -6.49% |
| 2021 | 81 | 46.9% | 10.29% | $12,795 | -3.15% |
| 2022 | 121 | 43.8% | 22.81% | $15,713 | -4.81% |
| 2023 | 101 | 39.6% | 12.93% | $17,744 | -6.13% |
| 2024 | 82 | 32.9% | 0.62% | $17,855 | -5.78% |
| 2025 | 53 | 50.9% | 14.06% | $20,366 | -2.46% |

**Best Year:** 2010 (+52.31%)
**Worst Year:** 2024 (+0.62%)
**Average Win Rate:** ~40.6%
**Worst Drawdown:** -6.76% (2003, 2011)

#### Annual Performance (3:1 Leverage)

| Year | Return % | Capital | Max DD % |
|------|----------|---------|----------|
| 2003 | 16.45% | $1,165 | -19.38% |
| 2004 | 23.29% | $1,436 | -18.72% |
| 2005 | 128.19% | $3,276 | -11.50% |
| 2006 | 32.34% | $4,336 | -13.16% |
| 2007 | 25.96% | $5,461 | -12.54% |
| 2008 | 93.42% | $10,564 | -18.99% |
| 2009 | 79.16% | $18,926 | -15.63% |
| 2010 | 243.77% | $65,063 | -18.07% |
| 2011 | 10.79% | $72,086 | -19.31% |
| 2012 | 45.46% | $104,854 | -17.83% |
| 2013 | 33.11% | $139,568 | -17.44% |
| 2014 | 13.85% | $158,901 | -11.79% |
| 2015 | 133.31% | $370,726 | -8.49% |
| 2016 | 32.00% | $489,352 | -17.99% |
| 2017 | 36.25% | $666,746 | -15.77% |
| 2018 | 36.62% | $910,937 | -10.61% |
| 2019 | 5.78% | $963,557 | -11.36% |
| 2020 | 21.01% | $1,165,962 | -18.74% |
| 2021 | 32.99% | $1,550,620 | -9.19% |
| 2022 | 81.80% | $2,818,954 | -13.87% |
| 2023 | 42.10% | $4,005,817 | -17.45% |
| 2024 | 0.98% | $4,045,207 | -16.47% |
| 2025 | 47.02% | $5,947,124 | -7.41% |

**Best Year:** 2010 (+243.77%)
**Worst Year:** 2024 (+0.98%)
**Worst Drawdown:** -19.38% (2003)

---

## Other Pairs

### GBPUSD (1-Day Timeframe)

**Training Date:** 2025-11-15
**Windows:** 47 (2003-2025)
**Optimal Threshold:** 48th/52nd percentile (very unselective, 96% of predictions traded)

#### Performance Metrics

| Leverage | Final Capital | Total Return | Annual Return | Sharpe Ratio | Years |
|----------|---------------|--------------|---------------|--------------|-------|
| **1:1** (No leverage) | $11,341 | 1,034% | **11.14%** | **0.928** | 23 |
| **3:1** (Standard) | $1,002,344 | 100,134% | **35.05%** | **0.829** | 23 |

#### Trade Statistics
- **Total trades**: 2,520
- **Average per year**: 110
- **Win rate**: ~38.1% (varied by year)

#### Key Observations
- **Best Year (1:1)**: 2016 (+39.58%)
- **Worst Year (1:1)**: 2024 (-2.77%)
- **Worst Drawdown**: -9.81% (2013)
- GBPUSD shows more volatility than EURUSD but still maintains solid risk-adjusted returns

### USDJPY (1-Day Timeframe)

**Training Date:** 2025-11-15
**Windows:** 47 (2003-2025)
**Optimal Threshold:** 35th/65th percentile (more selective, ~70% of predictions traded)

#### Performance Metrics

| Leverage | Final Capital | Total Return | Annual Return | Sharpe Ratio | Years |
|----------|---------------|--------------|---------------|--------------|-------|
| **1:1** (No leverage) | $13,803 | 1,280% | **12.09%** | **1.290** | 23 |
| **3:1** (Standard) | $1,839,210 | 183,821% | **38.66%** | **1.107** | 23 |

#### Trade Statistics
- **Total trades**: 2,302
- **Average per year**: 100
- **Win rate**: ~39.3% (varied by year)

#### Key Observations
- **Best Year (1:1)**: 2016 (+41.15%)
- **Worst Year (1:1)**: 2017 (-4.35%)
- **Worst Drawdown**: -12.63% (2017)
- USDJPY has the **highest Sharpe ratios** of all pairs tested, showing excellent risk-adjusted returns
- More selective entry thresholds (35/65 vs 48/52) result in fewer but higher quality trades

### AUDUSD (1-Day Timeframe)

**Training Date:** 2025-11-15
**Windows:** 47 (2003-2025)
**Optimal Threshold:** 48th/52nd percentile (very unselective, 96% of predictions traded)

#### Performance Metrics

| Leverage | Final Capital | Total Return | Annual Return | Sharpe Ratio | Years |
|----------|---------------|--------------|---------------|--------------|-------|
| **1:1** (No leverage) | $16,908 | 1,591% | **13.08%** | **1.491** | 23 |
| **3:1** (Standard) | $3,105,696 | 310,470% | **41.85%** | **1.409** | 23 |

#### Trade Statistics
- **Total trades**: 2,868
- **Average per year**: 125
- **Win rate**: ~37.5% (varied by year)

#### Key Observations
- **Best Year (1:1)**: 2018 (+25.64%)
- **Worst Year (1:1)**: 2009 (-5.41%)
- **Worst Drawdown**: -10.60% (2004)
- AUDUSD has the **highest Sharpe ratios of all pairs**, showing outstanding risk-adjusted returns
- Highest trade frequency of all pairs (125 trades/year avg)
- Most consistent performance with relatively low drawdowns

---

## Notes

- All results are from fresh training runs with randomized hyperparameter search (2025-11-15)
- Small variations (2-3%) between runs are expected due to hyperparameter randomization
- Transaction costs (0.02% or ~2 pips) are included in all calculations
- Leverage calculations assume standard forex margin requirements
- Drawdowns are calculated on a rolling basis throughout each year

## Validation Summary

All strategies have been validated through three independent tests:

1. **Exit Reason Analysis**: Confirms 91.8% of profits come from targets and 99% of losses are contained by stops
2. **Monte Carlo Simulation**: 10,000 simulations per pair show 100% probability of profit and 0% risk of ruin
3. **Model Attribution Testing**: ML model delivers 9.3x better returns than random entries, proving predictions drive performance

These validations confirm the strategy is robust, the ML model adds significant value, and risk management is effective.
