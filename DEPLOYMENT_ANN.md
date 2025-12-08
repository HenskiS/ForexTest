# ANN Model Deployment Guide

This guide covers deploying the ANN model from your desktop to your server.

## Overview

The ANN model is now the default model type. It offers:
- **Higher returns**: ~66% annual @ 2x (vs ~56% for XGBoost)
- **Better Sharpe**: 5.1 (vs 4.7 for XGBoost)
- **Faster training**: ~1-3 sec/pair (vs 2-5 sec for XGBoost)
- **Median split signal**: Simpler 50/50 threshold (vs 48/52)
- **Higher take profit**: 5% (vs 3% for XGBoost)

## Deployment Steps

### 1. Generate Prediction Buffers (Desktop - Faster)

The prediction buffer allows the bot to generate signals immediately without waiting 50+ days to build up history.

```bash
# On your desktop
python initialize_oanda_buffer_ann.py
```

This will:
- Fetch 878 days of historical data for each pair
- Generate 200 predictions using rolling window training
- Save buffers to `data/oanda_cache/prediction_buffer_ann_{PAIR}.pkl`
- Take ~10-15 minutes per pair (~40-60 minutes total)

Expected output files:
```
data/oanda_cache/
├── prediction_buffer_ann_EURUSD.pkl
├── prediction_buffer_ann_GBPUSD.pkl
├── prediction_buffer_ann_AUDUSD.pkl
└── prediction_buffer_ann_USDJPY.pkl
```

### 2. Commit Code Changes (Desktop)

```bash
git add .
git commit -m "Add ANN model support with optimized hyperparameters

- Add ANN model to TradingModel class
- Implement 50/50 median split thresholds
- Use 5% take profit for ANN (vs 3% for XGBoost)
- Make ANN the default model
- Add model flag to CLI"
git push origin production-clean  # or your branch name
```

### 3. Copy Buffers to Server

Option A: Using scp (recommended):
```bash
# From your desktop
scp data/oanda_cache/prediction_buffer_ann_*.pkl user@yourserver:~/ForexTest/data/oanda_cache/
```

Option B: Using rsync:
```bash
# From your desktop
rsync -av data/oanda_cache/prediction_buffer_ann_*.pkl user@yourserver:~/ForexTest/data/oanda_cache/
```

Option C: If you have direct access to server filesystem:
```bash
# Just copy the files directly
cp data/oanda_cache/prediction_buffer_ann_*.pkl /path/to/server/ForexTest/data/oanda_cache/
```

### 4. Deploy to Server

```bash
# SSH to your server
ssh user@yourserver

# Navigate to ForexTest directory
cd ~/ForexTest

# Pull latest changes
git pull origin production-clean  # or your branch name

# Verify buffer files exist
ls -lh data/oanda_cache/prediction_buffer_ann_*.pkl

# Should see 4 files, each ~10-50 KB
```

### 5. Test on Server (Dry Run)

```bash
# Test with dry-run mode (no actual trades)
python oanda_multi_pair_trader.py --dry-run

# Should see:
# - Model: ANN
# - Loaded prediction buffer: 200 predictions (per pair)
# - Training ANN model on last 378 days...
# - Signal Generation (ANN): with median threshold
```

### 6. Test on Practice Account

```bash
# Run on practice account (default)
python oanda_multi_pair_trader.py

# Monitor for a few days to verify:
# - Signals are generated correctly
# - 5% take profit is being used
# - Trades execute properly
# - Buffer is growing (201, 202, 203... predictions)
```

### 7. Go Live (When Ready)

```bash
# Run on live account
python oanda_multi_pair_trader.py --live

# Or with explicit leverage
python oanda_multi_pair_trader.py --live --leverage 2.0
```

## File Changes Summary

### Modified Files
1. **trading/config.py** - Added ANN hyperparameters and thresholds
2. **trading/trading_models.py** - Extended to support both XGBoost and ANN
3. **oanda_multi_pair_trader.py** - Added model flag, made ANN default

### New Files
1. **initialize_oanda_buffer_ann.py** - Script to generate ANN prediction buffers
2. **DEPLOYMENT_ANN.md** - This deployment guide

### Generated Files (Not in Git)
```
data/oanda_cache/
├── prediction_buffer_ann_EURUSD.pkl  # ~10-50 KB each
├── prediction_buffer_ann_GBPUSD.pkl
├── prediction_buffer_ann_AUDUSD.pkl
└── prediction_buffer_ann_USDJPY.pkl
```

## Switching Between Models

You can still use XGBoost if needed:

```bash
# Use XGBoost explicitly
python oanda_multi_pair_trader.py --model xgboost

# Use ANN (default, no flag needed)
python oanda_multi_pair_trader.py
```

## Troubleshooting

### "No prediction buffer found" error
**Solution**: Run `initialize_oanda_buffer_ann.py` on desktop and copy files to server.

### "Failed to fetch data" during buffer initialization
**Solution**: Check OANDA API credentials in `.env` file.

### XGBoost still running instead of ANN
**Solution**: Check that you pulled latest code and default is set to 'ann' in line 423.

### Signals not being generated
**Solution**:
1. Check buffer has 50+ predictions: Should see "Loaded prediction buffer: XXX predictions"
2. If < 50, wait a few days or re-run initialization script

## Performance Monitoring

Monitor these metrics in first few weeks:
- **Trade frequency**: Should see trades on ~50% of days (median split takes every signal)
- **Stop-out rate**: Should be ~56-60% (validated with intraday analysis)
- **Take profit hits**: ~40% of trades should hit 5% TP
- **Average win**: ~0.6%
- **Average loss**: ~-0.2%

## Rollback Plan

If ANN underperforms, revert to XGBoost:

```bash
# Temporary rollback (no code changes)
python oanda_multi_pair_trader.py --model xgboost

# Permanent rollback
git revert HEAD
git push
# Then pull on server
```

## Expected Performance

Based on backtest results (4,500 days):

| Metric | 2x Leverage | 4x Leverage |
|--------|-------------|-------------|
| Annual Return | 63.80% | 165.50% |
| Sharpe Ratio | 5.03 | 5.03 |
| Max Drawdown | -4.21% | -8.26% |
| Trade Win Rate | 36.9% | 36.9% |
| Daily Win Rate | 54.4% | 54.4% |

## Support

If issues arise:
1. Check logs for errors
2. Verify buffer files copied correctly
3. Test with `--dry-run` first
4. Revert to XGBoost if critical issues: `--model xgboost`
