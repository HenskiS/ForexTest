# Forex Trading Model - Production Clean Branch

This is a minimal production-ready branch containing only the essential files for:
1. Fetching forex data
2. Adding technical features
3. Training XGBoost models
4. Backtesting strategies
5. Exporting trades

## Files

### Core Scripts
- **data_fetcher.py** - Fetch OHLCV data from Dukascopy
- **add_features.py** - Add technical indicators (EMAs, MACD, RSI, etc.)
- **train_xgboost_multitarget.py** - Train walk-forward XGBoost models
- **backtest_multitarget.py** - Backtest with optimized entry thresholds
- **analyze_1day_eurusd_annual.py** - Annual performance analysis
- **export_1day_trades_eurusd.py** - Export trades to CSV

### Configuration
- **requirements.txt** - Python dependencies
- **.env** - API keys and configuration (not tracked)
- **.gitignore** - Git ignore rules

### Data
- **data/** - Historical OHLCV data and feature files
  - Raw data: `*_1day.csv`, `*_1hour.csv`, `*_4hour.csv`
  - With features: `*_with_features*.csv`
  - Pairs: EURUSD, GBPUSD, USDJPY, AUDUSD

### Documentation
- **XGBOOST_RESULTS.md** - Model performance results

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train model for a pair:
```bash
python train_xgboost_multitarget.py --target target_5day_return --n_iter 20 --pair EURUSD
```

3. Analyze results:
```bash
python analyze_1day_eurusd_annual.py
```

4. Export trades:
```bash
python export_1day_trades_eurusd.py
```

## Results Summary

**EURUSD (1-day timeframe):**
- **Annual Return**: 47.84% (at 3:1 leverage)
- **Win Rate**: 40.6%
- **Average Trade**: +12.98 pips
- **Risk/Reward**: 2.22:1 (93 pip wins vs 42 pip losses)
- **Total Trades**: 2,445 over 23 years (106/year)

See `XGBOOST_RESULTS.md` for full details on all pairs and timeframes.
