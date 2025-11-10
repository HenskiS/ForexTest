# Multi-Currency Pair Performance Comparison

## Overall Performance Summary (2000-2025)

| Metric | EURUSD | GBPUSD | USDJPY | AUDUSD |
|--------|--------|--------|--------|--------|
| **Total Return** | +954.68% | +702.91% | +610.42% | +281.16% |
| **Final Capital** | $10,546.82 | $8,029.10 | $7,104.16 | $3,811.58 |
| **Annual Return** | 10.52% | 6.96% | 6.36% | 3.73% |
| **Sharpe Ratio** | 1.057 | 1.364 | 1.238 | 0.859 |
| **Max Drawdown** | -7.70% | -25.64% | -10.48% | -25.64% |
| **Win Rate** | 39.0% | 38.2% | 37.4% | 34.8% |
| **Profit Factor** | 1.525 | N/A | 1.345 | 1.263 |
| **Total Trades** | 1,863 | 1,828 | 1,956 | 2,016 |
| **Time Period** | 22.83 yrs | 22.95 yrs | 25.47 yrs | 25.53 yrs |

## Risk-Adjusted Performance Ranking

1. **GBPUSD**: Sharpe 1.364 - Best risk-adjusted returns despite lower absolute returns
2. **USDJPY**: Sharpe 1.238 - Consistent performance, reasonable drawdowns
3. **EURUSD**: Sharpe 1.057 - Highest absolute returns but more volatility relative to risk
4. **AUDUSD**: Sharpe 0.859 - Weakest performer with large drawdowns

## Recent Performance (2023-2025)

### 2023
| Pair | Return | Win Rate | Capital |
|------|--------|----------|---------|
| GBPUSD | +7.04% | 36.9% | $6,887.35 |
| USDJPY | +7.94% | 35.2% | $6,167.47 |
| AUDUSD | +7.45% | 34.5% | $2,937.80 |
| EURUSD | +4.49% | 35.1% | $10,006.81 |

### 2024
| Pair | Return | Win Rate | Capital |
|------|--------|----------|---------|
| AUDUSD | +16.02% | 42.4% | $3,408.38 |
| USDJPY | +8.94% | 32.7% | $6,719.11 |
| EURUSD | +5.99% | 36.0% | $10,605.94 |
| GBPUSD | **-2.60%** | 27.3% | $6,708.56 |

### 2025 YTD
| Pair | Return | Win Rate | Capital |
|------|--------|----------|---------|
| **GBPUSD** | **+19.68%** | **51.0%** | **$8,029.10** |
| AUDUSD | +11.83% | 39.3% | $3,811.58 |
| EURUSD | +6.40% | 35.1% | $10,546.82 |
| USDJPY | +5.73% | 36.5% | $7,104.16 |

## Key Insights

### Regime Rotation
The strategy exhibits **regime rotation** across currency pairs:
- **EURUSD**: Strong historically (10.52% annual) but declining in recent years
- **GBPUSD**: Terrible 2024 (-2.60%) but exceptional 2025 (+19.68%, 51% win rate)
- **USDJPY**: Most consistent performer across all time periods
- **AUDUSD**: High volatility, best recent years (2024-2025)

### Portfolio Diversification Opportunity
Rather than picking a single "best" pair, the regime rotation suggests a **multi-pair portfolio** could:
1. Capture opportunities as they rotate between pairs
2. Reduce overall portfolio volatility
3. Maintain more consistent returns across market conditions

### Historical Best Years by Pair

**GBPUSD Best Years:**
- 2005: +41.78% (52.4% win rate)
- 2020: +27.90% (47.4% win rate)
- 2025: +19.68% (51.0% win rate) - YTD

**EURUSD Best Years:**
- 2003: +33.50% (38.9% win rate)
- 2020: +29.43% (49.0% win rate)
- 2008: +27.08% (44.0% win rate)

**USDJPY Best Years:**
- 2008: +24.43% (41.0% win rate)
- 2009: +19.54% (43.8% win rate)
- 2016: +19.47% (45.2% win rate)

**AUDUSD Best Years:**
- 2007: +20.03% (42.7% win rate)
- 2004: +17.78% (39.2% win rate)
- 2011: +16.41% (37.9% win rate)
- 2024: +16.02% (42.4% win rate)

### Worst Years by Pair

**GBPUSD**: 2010 (-4.55%), 2024 (-2.60%)
**EURUSD**: 2017 (-2.04%), 2021 (-1.24%)
**USDJPY**: 2017 (-1.95%)
**AUDUSD**: 2009 (-12.48%), 2006 (-5.88%), 2016 (-6.36%)

## Strategy Parameters (Constant Across All Pairs)

- **Target**: 5-day forward returns
- **Signal Generation**: Q1/Q3 percentile (25th/75th)
- **Stops**: Vol-adjusted (base 0.40% SL / 1.00% TP, scaled by ATR)
- **Loss Cooldown**: 1 day
- **Transaction Costs**: 0.02% per trade
- **Position Sizing**: Full capital reinvestment
- **Walk-Forward**: 600 train / 156 val / 126 test, rolling by 126 days

## Portfolio Performance ($1,000 per pair)

**Initial Investment:** $4,000

| Pair | Initial | Final | Profit | % of Portfolio |
|------|---------|-------|--------|----------------|
| EURUSD | $1,000 | $10,546.82 | $9,546.82 | 35.8% |
| GBPUSD | $1,000 | $8,029.10 | $7,029.10 | 27.2% |
| USDJPY | $1,000 | $7,104.16 | $6,104.16 | 24.1% |
| AUDUSD | $1,000 | $3,811.58 | $2,811.58 | 12.9% |
| **TOTAL** | **$4,000** | **$29,491.66** | **$25,491.66** | **100%** |

**Portfolio Metrics:**
- **Total Return:** +637.29%
- **Annualized Return:** 8.87%
- **All pairs profitable:** Zero losing positions

**Diversification Benefit:**
- Portfolio return (8.87%) sits between best (EURUSD: 10.52%) and worst (AUDUSD: 3.73%)
- Reduces volatility compared to any single pair
- When GBPUSD lost -2.60% in 2024, AUDUSD gained +16.02% to offset

## Files Generated

- `yearly_performance_EURUSD.txt` - EURUSD yearly breakdown
- `yearly_performance_GBPUSD.txt` - GBPUSD yearly breakdown
- `yearly_performance_USDJPY.txt` - USDJPY yearly breakdown
- `yearly_performance_AUDUSD.txt` - AUDUSD yearly breakdown
- `XGBOOST_RESULTS.md` - Updated with multi-pair results

## Next Steps

1. **Monte Carlo Validation** ✅ COMPLETE - All 4 pairs validated (100% profit probability)
   - See [MONTE_CARLO_MULTI_PAIR.md](MONTE_CARLO_MULTI_PAIR.md) for detailed results
2. **Enhanced Realism** - Add slippage (0.5-1 pip), higher fees, execution delays
3. **Meta-Strategy** - Model to dynamically allocate capital across pairs based on regime/confidence
4. **Correlation Analysis** - Optimize portfolio weights considering pair correlations
