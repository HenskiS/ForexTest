# Sleep Well Strategy

**8-pair ANN forex strategy with independent stop losses. ~94% annual return, 2.0x leverage.**

## Current Configuration (December 2025)

| Parameter | Value |
|-----------|-------|
| Model | ANN (MLPRegressor) |
| Architecture | (13, 20, 31) layers |
| Thresholds | 10/90 percentile |
| Hold Period | 5 trading days |
| Stop Loss | 2.5% per slot (independent) |
| Take Profit | None (time-based exit) |
| Leverage | 2.0x |
| Pairs | 8: EURUSD, GBPUSD, AUDUSD, USDJPY, EURJPY, USDCAD, USDCHF, NZDUSD |
| Allocation | 22.5% per slot |
| Max Slots | 5 per pair (40 total) |

## Performance Summary

| Metric | Value |
|--------|-------|
| **Annual Return** | ~94% |
| **Sharpe Ratio** | 4.50 |
| **Max Drawdown** | -16.6% |
| **Win Rate** | 60.8% |
| **Total Trades** | ~7,600 (14 years) |

### Yearly Performance (8 pairs, 22.5% allocation, 2.0x leverage, 2.5% SL)

| Year | Return | Max DD | Win Rate | Trades | SL Exits |
|------|--------|--------|----------|--------|----------|
| 2011 | 61.3%  | -8.0%  | 67.3%    | 162    | 16       |
| 2012 | 107.2% | -5.6%  | 62.3%    | 531    | 6        |
| 2013 | 60.0%  | -15.4% | 59.7%    | 558    | 23       |
| 2014 | 99.3%  | -11.7% | 61.7%    | 629    | 7        |
| 2015 | 91.6%  | -14.1% | 56.9%    | 415    | 25       |
| 2016 | 89.7%  | -14.1% | 56.2%    | 608    | 39       |
| 2017 | 104.1% | -8.3%  | 61.5%    | 589    | 5        |
| 2018 | 84.3%  | -6.3%  | 60.5%    | 547    | 10       |
| 2019 | 49.2%  | -9.2%  | 56.4%    | 557    | 2        |
| 2020 | 27.4%  | -16.4% | 55.7%    | 481    | 28       |
| 2021 | 125.1% | -4.8%  | 63.8%    | 602    | 4        |
| 2022 | 177.0% | -9.6%  | 62.8%    | 567    | 33       |
| 2023 | 108.5% | -4.1%  | 66.7%    | 460    | 11       |
| 2024 | 166.8% | -8.5%  | 67.2%    | 536    | 3        |
| 2025 | 28.3%  | -9.6%  | 56.1%    | 374    | 11       |

**15 consecutive profitable years.** Worst year: 2020 (+27.4%), worst drawdown: -16.6%.

---

## How It Works

### Independent Stop Losses (Key Feature)

Each slot has its **own 2.5% stop loss** from its entry price:

```
Monday: Enter LONG EURUSD @ 1.1000 → SL @ 1.0725
Tuesday: Enter LONG EURUSD @ 1.1050 → SL @ 1.0774
Wednesday: Price drops to 1.0750
  → Monday's slot: SL NOT hit (SL @ 1.0725)
  → Tuesday's slot: SL HIT (SL @ 1.0774) → closed with loss
  → Monday's slot continues running
```

**Benefits:**
- Later entries can stop out without affecting earlier entries
- No correlated exits when one stop hits
- Simpler than DCA/averaged stops
- Each trade stands on its own merit

### FIFO Compliance

The strategy only adds positions in the **same direction**:
- If LONG, can add more LONG slots (up to 5)
- If LONG, cannot go SHORT until all longs exit
- This is US broker FIFO compliant

### Position Sizing

With 22.5% allocation per slot and avg 6.4 positions open:
- Effective leverage: 2.0x × 6.4 × 22.5% = ~2.9x average
- Peak exposure: 8 pairs × 5 slots × 22.5% × 2.0x = 18x (rare)

---

## Historical Context

### Original Bug Discovery

The original backtest had overlapping trades that created implicit leverage without tracking it. Instead of fixing by blocking overlaps, we embraced multi-slot trading with proper tracking.

### December 2025 Backtest Fixes

Two critical bugs were fixed that had inflated historical performance:

1. **Entry-day stop loss check** - Previously, if a stop was hit on the same day as entry, it was ignored. Now properly checks entry candle's low/high against stop price.

2. **Time exit price** - Previously used `close` price for time-based exits, but since we execute at 6am, we should use `open` price. Using `close` was lookahead bias (price not available at execution time).

These fixes reduced apparent annual return from ~98% to ~65% at 1.5x leverage. Optimization found that **wider stops (2.5% vs 2%)** actually improve both returns AND reduce drawdowns by avoiding whipsaw losses.

### Evolution

1. **4 pairs, 25% allocation, 2% SL** → 55% annual
2. **5 pairs, 20% allocation, 2% SL** (added EURJPY) → 55% annual
3. **8 pairs, 22.5% allocation, 2% SL** (added USDCAD, USDCHF, NZDUSD) → 60% annual (after fixes)
4. **8 pairs, 22.5% allocation, 2.5% SL, 2.0x leverage** → **~94% annual** (current)

The jump from 5→8 pairs nearly doubled returns due to more diversification and more signals.

## ANN Model Configuration

```python
ANN_PARAMS = {
    'hidden_layer_sizes': (13, 20, 31),  # 3 hidden layers
    'activation': 'tanh',
    'solver': 'sgd',
    'learning_rate_init': 0.001,
    'momentum': 0.4,
    'batch_size': 64,
    'max_iter': 20,
    'alpha': 0.0001,
    'learning_rate': 'adaptive',
    'random_state': 42
}
```

## Implementation Files

- `trading/config.py` - Strategy parameters (8 pairs, thresholds, etc.)
- `trading/position_manager.py` - Multi-slot position tracking with independent stops
- `trading/oanda_client.py` - OANDA API integration
- `oanda_multi_pair_trader.py` - Production trader (runs Sunday 2:30 PM PT)
- `test_dca_stop.py` - Backtest with independent stops

## Server Deployment

```bash
# Pull latest code
git pull

# Initialize buffers for new pairs (if needed)
python initialize_new_pair_buffers.py

# Verify config with dry run
python oanda_multi_pair_trader.py --live --dry-run --yes
```

---
*Strategy developed December 2025. Backtest fixes and optimization (2.5% SL, 2.0x leverage) applied December 14, 2025.*
