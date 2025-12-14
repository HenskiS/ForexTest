# Sleep Well Strategy

**ANN-based forex trading strategy optimized for consistent returns with manageable drawdowns.**

## Important Discovery (December 2025)

The original backtest showing 65.1% annual returns had a **bug that allowed overlapping trades** on the same pair. This effectively created implicit leverage without tracking it properly.

### What the Bug Did
- When already in a EURUSD long position, the backtest would enter *another* EURUSD long
- With 5-day holds and daily signals, up to 5 overlapping trades per pair could run simultaneously
- Each trade got full 25% allocation = up to 125% exposure per pair
- 4 pairs × 5 overlapping trades = up to 500% total exposure (5x implicit leverage)
- Average daily exposure was ~131%, peaking at 475%

### Corrected Results (No Overlap)

When properly blocking new trades while in a position:

| Metric | Buggy Backtest | Corrected |
|--------|----------------|-----------|
| **Annual Return** | 65.1% | 17-19% |
| **Trades** | 3,829 | ~850 |
| **Win Rate** | 69.0% | 68-69% |
| **Sharpe** | 7.19 | ~3.5 |
| **Max Drawdown** | -13.9% | -8% |

**Key insight:** The win rate is nearly identical. The difference in returns comes entirely from trading more often and deploying more capital, not from better individual trades.

---

## Two Implementation Options

### Option 1: Conservative (Single Position per Pair)

Use the corrected backtest approach - one position per pair at a time.

**Expected Performance:** ~17-19% annual return at 1.5x leverage

**Pros:**
- Simple implementation
- Lower drawdown risk
- Works on any account type

**Cons:**
- Lower returns
- Misses signals while in existing positions

### Option 2: DCA Stop (Overlapping with Averaged Stop Loss)

Implement the overlapping approach using **Dollar Cost Averaged stop losses**. This is implementable on Oanda's netting account.

**How it works:**
1. Track "virtual positions" internally (up to 5 per pair)
2. Each signal gets full 25% allocation
3. Calculate weighted average entry price across all virtual positions
4. Set ONE stop loss below (long) or above (short) the average entry
5. When stop hits: entire position closes (all virtual slots)
6. Time exits: each virtual slot exits after its own 5-day hold period

#### Linear Stop Tightening (Recommended)

As more positions are added, tighten the stop loss to reduce max drawdown:

| Slots Filled | Stop Loss % |
|--------------|-------------|
| 1            | 2.0%        |
| 2            | 1.75%       |
| 3            | 1.5%        |
| 4            | 1.25%       |
| 5            | 1.0%        |

**Expected Performance at 1.5x leverage (5 pairs):**

| Strategy | Annual Return | Max Drawdown | Sharpe | Win Rate |
|----------|---------------|--------------|--------|----------|
| Fixed 2% SL | 55.0% | -13.3% | 4.97 | 62.7% |
| **Linear Tightening** | **51.2%** | **-13.1%** | **4.74** | **61.2%** |

Linear tightening trades ~4% annual return for slightly better max drawdown.

> **Note (December 2025):** Performance revised down from ~68% due to OANDA data re-fetch that corrected timezone issues and changed the historical data composition. The current numbers reflect clean, properly-aligned daily candles.

#### Yearly Performance (Baseline 2% SL @ 1.5x, 5 pairs)

| Year | Return | Max DD | Win Rate |
|------|--------|--------|----------|
| 2011 | 28.8%  | -3.4%  | 73.6%    |
| 2012 | 76.2%  | -5.0%  | 68.6%    |
| 2013 | 27.1%  | -12.3% | 58.2%    |
| 2014 | 51.3%  | -11.2% | 61.8%    |
| 2015 | 36.3%  | -8.3%  | 60.9%    |
| 2016 | 39.7%  | -13.3% | 53.9%    |
| 2017 | 59.9%  | -8.0%  | 64.1%    |
| 2018 | 51.4%  | -5.8%  | 63.1%    |
| 2019 | 25.2%  | -9.0%  | 56.3%    |
| 2020 | 30.1%  | -11.5% | 57.5%    |
| 2021 | 76.8%  | -2.0%  | 67.5%    |
| 2022 | 110.9% | -7.9%  | 62.0%    |
| 2023 | 73.2%  | -3.1%  | 69.0%    |
| 2024 | 89.7%  | -2.4%  | 71.7%    |
| 2025 | 24.6%  | -7.2%  | 58.4%    |

**15 consecutive profitable years.** Worst year: 2019 (+25.2%), worst drawdown: 2016 (-13.3%).

**Pros:**
- Captures all signals
- Higher returns through more capital deployment
- Averaged stop prevents premature stop-outs
- Linear tightening reduces max drawdown

**Cons:**
- More complex position management
- Correlated exits when stop hits
- Requires tracking virtual positions internally

### Option 3: Add More Pairs

Instead of overlapping trades on 5 pairs, add more currency pairs:
- **25 pairs × 1 trade each** = same capital deployment as 5 pairs × 5 overlapping
- True diversification across different currencies
- Each pair has independent stop loss
- Simpler than DCA stop approach

**Already Added:** EURJPY (December 2025) - boosted returns from 40.9% to 55.0% for 4→5 pairs

**Candidates:** USDCAD, USDCHF, NZDUSD, EURGBP, GBPJPY, AUDJPY, etc.

**Requires:** Testing ANN predictions on additional pairs

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | ANN (MLPRegressor) |
| Architecture | (13, 20, 31) layers |
| Thresholds | 10/90 percentile |
| Hold Period | 5 trading days |
| Stop Loss | 2% → 1% (linear tightening) |
| Take Profit | None (time-based exit) |
| Leverage | 1.5x |
| Pairs | EURUSD, GBPUSD, AUDUSD, USDJPY, EURJPY |
| Allocation | 20% per pair |
| Max Slots | 5 per pair |

## Why the Overlapping Approach Works

The strategy's edge comes from **signal persistence**. When the ANN predicts a strong move:
1. The prediction often stays in the 90th+ percentile for several consecutive days
2. Each day you enter, you're DCA-ing into the same directional view
3. Multiple entries at slightly different prices provides time diversification
4. The 5-day hold captures the predicted move

It's essentially **daily entries into rolling 5-day positions** that capture signal persistence while diversifying entry timing risk.

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

### For DCA Stop Approach
- `test_dca_stop.py` - Backtest implementation with DCA averaged stop loss

### Core Strategy
- `trading/config.py` - Strategy parameters
- `trading/oanda_client.py` - Optional take profit
- `trading/position_manager.py` - Business day counting
- `oanda_multi_pair_trader.py` - Production trader

## Server Deployment

```bash
# Delete old position states (keep prediction buffers!)
rm data/oanda_cache/*_state.json

# Pull latest code
git pull

# Verify config
python oanda_multi_pair_trader.py --dry-run
```

---

## Decision Matrix

| If you want... | Choose |
|----------------|--------|
| Simplicity + lower risk | Option 1: Single position |
| Best risk-adjusted returns | Option 2: DCA Stop w/ Linear Tightening |
| True diversification | Option 3: More pairs |

---
*Strategy developed December 2025. Overlapping trade bug discovered, DCA stop solution developed, and linear stop tightening optimized December 2025. EURJPY added as 5th pair December 2025. Performance numbers revised after OANDA data quality fixes.*
