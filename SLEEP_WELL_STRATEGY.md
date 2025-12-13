# Sleep Well Strategy

**ANN-based forex trading strategy optimized for consistent returns with manageable drawdowns.**

## Important Discovery (December 2024)

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
4. Set ONE stop loss at 2% below (long) or above (short) the average entry
5. When stop hits: entire position closes (all virtual slots)
6. Time exits: each virtual slot exits after its own 5-day hold period

**Expected Performance at various leverage levels:**

| Leverage | Annual Return | Max Drawdown | Sharpe |
|----------|---------------|--------------|--------|
| 1.0x     | 43.4%         | -12.7%       | 7.02   |
| 1.5x     | 71.4%         | -18.4%       | 7.02   |
| 2.0x     | 104.7%        | -23.8%       | 7.02   |

**Pros:**
- Captures all signals
- Higher returns through more capital deployment
- Averaged stop prevents premature stop-outs

**Cons:**
- More complex position management
- Higher drawdown due to correlated exits
- Requires tracking virtual positions internally

### Option 3: Add More Pairs

Instead of overlapping trades on 4 pairs, add more currency pairs:
- **20 pairs × 1 trade each** = same capital deployment as 4 pairs × 5 overlapping
- True diversification across different currencies
- Each pair has independent stop loss
- Simpler than DCA stop approach

**Candidates:** USDCAD, USDCHF, NZDUSD, EURGBP, EURJPY, GBPJPY, etc.

**Requires:** Testing ANN predictions on additional pairs

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | ANN (MLPRegressor) |
| Architecture | (13, 20, 31) layers |
| Thresholds | 10/90 percentile |
| Hold Period | 5 trading days |
| Stop Loss | 2% |
| Take Profit | None (time-based exit) |
| Leverage | 1.5x |
| Pairs | EURUSD, GBPUSD, AUDUSD, USDJPY |
| Allocation | 25% per pair |

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
| Maximum returns + accept higher DD | Option 2: DCA Stop |
| True diversification | Option 3: More pairs |

---
*Strategy developed December 2024. Overlapping trade bug discovered and DCA stop solution developed December 2024.*
