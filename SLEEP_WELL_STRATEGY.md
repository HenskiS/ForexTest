# Sleep Well Strategy

**8-pair ANN forex strategy with independent stop losses. ~98% annual return, 1.5x leverage.**

## Current Configuration (December 2025)

| Parameter | Value |
|-----------|-------|
| Model | ANN (MLPRegressor) |
| Architecture | (13, 20, 31) layers |
| Thresholds | 10/90 percentile |
| Hold Period | 5 trading days |
| Stop Loss | 2% per slot (independent) |
| Take Profit | None (time-based exit) |
| Leverage | 1.5x |
| Pairs | 8: EURUSD, GBPUSD, AUDUSD, USDJPY, EURJPY, USDCAD, USDCHF, NZDUSD |
| Allocation | 22.5% per slot |
| Max Slots | 5 per pair (40 total) |

## Performance Summary

| Metric | Value |
|--------|-------|
| **Annual Return** | ~98% |
| **Sharpe Ratio** | 5.49 |
| **Max Drawdown** | -14.2% |
| **Win Rate** | 63.5% |
| **Total Trades** | ~6,700 (14 years) |

### Yearly Performance (8 pairs, 22.5% allocation, 1.5x leverage)

| Year | Return | Max DD | Win Rate | Trades | SL Exits |
|------|--------|--------|----------|--------|----------|
| 2011 | 56.8%  | -5.5%  | 74.6%    | 321    | 14       |
| 2012 | 118.9% | -6.7%  | 70.4%    | 472    | 38       |
| 2013 | 64.0%  | -14.2% | 59.8%    | 498    | 82       |
| 2014 | 86.7%  | -12.6% | 62.9%    | 478    | 76       |
| 2015 | 69.9%  | -10.9% | 61.8%    | 489    | 71       |
| 2016 | 72.8%  | -9.9%  | 56.7%    | 505    | 94       |
| 2017 | 93.2%  | -7.6%  | 65.3%    | 478    | 60       |
| 2018 | 83.7%  | -7.9%  | 63.9%    | 494    | 59       |
| 2019 | 63.9%  | -10.6% | 58.9%    | 499    | 71       |
| 2020 | 74.4%  | -11.9% | 59.7%    | 497    | 90       |
| 2021 | 135.4% | -3.9%  | 68.0%    | 472    | 44       |
| 2022 | 195.9% | -7.9%  | 64.7%    | 511    | 59       |
| 2023 | 123.4% | -6.9%  | 69.3%    | 475    | 34       |
| 2024 | 155.7% | -4.5%  | 72.5%    | 480    | 30       |
| 2025 | 65.1%  | -7.6%  | 60.9%    | 507    | 56       |

**15 consecutive profitable years.** Worst year: 2011 (+56.8%), worst drawdown: 2013 (-14.2%).

---

## How It Works

### Independent Stop Losses (Key Feature)

Each slot has its **own 2% stop loss** from its entry price:

```
Monday: Enter LONG EURUSD @ 1.1000 → SL @ 1.0780
Tuesday: Enter LONG EURUSD @ 1.1050 → SL @ 1.0829
Wednesday: Price drops to 1.0800
  → Monday's slot: SL NOT hit (SL @ 1.0780)
  → Tuesday's slot: SL HIT (SL @ 1.0829) → closed with loss
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
- Effective leverage: 1.5x × 6.4 × 22.5% = ~2.2x average
- Peak exposure: 8 pairs × 5 slots × 22.5% × 1.5x = 13.5x (rare)

---

## Historical Context

### Original Bug Discovery

The original backtest had overlapping trades that created implicit leverage without tracking it. Instead of fixing by blocking overlaps, we embraced multi-slot trading with proper tracking.

### Evolution

1. **4 pairs, 25% allocation** → 55% annual
2. **5 pairs, 20% allocation** (added EURJPY) → 55% annual
3. **8 pairs, 22.5% allocation** (added USDCAD, USDCHF, NZDUSD) → **98% annual**

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
*Strategy developed December 2025. Expanded to 8 pairs with independent stop losses December 2025.*
