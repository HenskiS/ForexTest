# ForexTest Project Setup Guide

Quick setup guide for initializing the ForexTest trading strategy on a new device.

## Prerequisites

1. **Python 3.8+** with virtual environment
2. **OANDA Account** (Practice or Live)
   - Sign up at: https://www.oanda.com/
   - Get your API key and Account ID from account settings
3. **Git** (to clone the repository)

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone <your-repo-url>
cd ForexTest

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure OANDA API

Create a `.env` file in the project root:

```env
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
```

### 3. Run Interactive Setup

```bash
python initialize_project.py
```

This interactive script will guide you through 4 steps:

1. **Fetch Historical OHLC Data** (required, ~5-10 min)
   - Downloads 5000 days of price data from OANDA
   - Creates: `data/{PAIR}_1day_oanda.csv`

2. **Fetch Spread Data** (required for backtesting, ~5-10 min)
   - Downloads bid/ask data for accurate backtest simulation
   - Creates: `data/{PAIR}_1day_with_spreads.csv`

3. **Initialize Prediction Buffers** (required for production, ~2-3 hours)
   - Generates 200-day prediction history for each pair
   - Creates: `data/oanda_cache/prediction_buffer_ann_{PAIR}.pkl`

4. **Generate Backtest Predictions** (optional, ~6-10 hours)
   - Generates predictions for entire historical dataset
   - Only needed if you want to run backtests
   - Creates: `optimized_ann_predictions/predictions_{PAIR}.pkl`

The script is **idempotent** - safe to run multiple times. It will:
- Detect existing files and let you skip completed steps
- Check if data is up-to-date (warns if data is >7 days old)
- Automatically identify which pairs need updating

## Manual Setup (Alternative)

If you prefer to run steps individually:

### Step 1: Fetch Historical Data

```bash
# Fetch for all 8 pairs (run for each pair)
# Note: Add --live flag if you only have live account credentials
# IMPORTANT: Use --daily-alignment 9 to match production system (9 AM EST)
python oanda_data_fetcher.py --pair EURUSD --count 5000 --live --daily-alignment 9
python oanda_data_fetcher.py --pair GBPUSD --count 5000 --live --daily-alignment 9
python oanda_data_fetcher.py --pair AUDUSD --count 5000 --live --daily-alignment 9
python oanda_data_fetcher.py --pair USDJPY --count 5000 --live --daily-alignment 9
python oanda_data_fetcher.py --pair EURJPY --count 5000 --live --daily-alignment 9
python oanda_data_fetcher.py --pair USDCAD --count 5000 --live --daily-alignment 9
python oanda_data_fetcher.py --pair USDCHF --count 5000 --live --daily-alignment 9
python oanda_data_fetcher.py --pair NZDUSD --count 5000 --live --daily-alignment 9
```

### Step 2: Fetch Spread Data

```bash
# Note: Add --live flag if you only have live account credentials
python fetch_spread_data.py --live
```

### Step 3: Initialize Prediction Buffers

```bash
# Note: Add --live flag if you only have live account credentials
python initialize_oanda_buffer_ann.py --live
```

### Step 4: Generate Backtest Predictions (Optional)

```bash
# Train all 8 pairs (~5-10 min per pair with parallel execution)
python train_all_pairs_optimized_hyperparams.py
```

## After Setup

### Test Production Trader

```bash
# Dry run (no actual trades)
python oanda_multi_pair_trader.py --dry-run

# Practice account
python oanda_multi_pair_trader.py

# Live account (⚠️ REAL MONEY!)
python oanda_multi_pair_trader.py --live
```

### Run Backtests

```bash
python test_dca_stop_fixed.py
```

## Project Structure

```
ForexTest/
├── data/
│   ├── *_1day_oanda.csv           # Historical OHLC data
│   ├── *_1day_with_spreads.csv    # Spread data for backtesting
│   └── oanda_cache/
│       └── prediction_buffer_ann_*.pkl  # Prediction buffers
├── optimized_ann_predictions/
│   └── predictions_*.pkl           # Full backtest predictions
├── oanda_multi_pair_trader.py      # Production trader (main entry point)
├── test_dca_stop_fixed.py          # Backtest script
├── initialize_project.py           # Interactive setup script
├── oanda_data_fetcher.py           # OHLC data fetcher
├── fetch_spread_data.py            # Spread data fetcher
├── initialize_oanda_buffer_ann.py        # Prediction buffer initializer
└── train_all_pairs_optimized_hyperparams.py  # Backtest prediction generator
```

## Troubleshooting

### "No module named 'trading'"

Make sure you're in the project directory and virtual environment is activated.

### "OANDA_API_KEY not found"

Create a `.env` file with your OANDA credentials (see step 2 above).

### "Failed to fetch data"

- Check your internet connection
- Verify OANDA API credentials are correct
- Ensure you're using the correct account type (practice vs live)

### Prediction buffer initialization is slow

This is normal. Step 3 trains the model 200 times per pair (1600 times total for 8 pairs). It takes 2-3 hours on most machines. You can:
- Run it overnight
- Run it on a faster machine and copy the buffer files afterward
- Or be patient!

## Configuration

Key trading parameters in `trading/config.py`:

- **DEFAULT_PAIRS**: 8-pair portfolio (Sleep Well strategy)
- **STOP_LOSS_PCT**: 2.5% stop loss
- **HOLDING_PERIOD_DAYS**: 5-day holding period
- **ANN_PERCENTILE_LOWER/UPPER**: 10/90 signal thresholds
- **LEVERAGE**: 2.0x leverage
- **TRAIN_WINDOW_SIZE**: 378-day training window
- **PREDICTION_BUFFER_SIZE**: 200-day buffer

See [SLEEP_WELL_STRATEGY.md](SLEEP_WELL_STRATEGY.md) for strategy details.

## Deployment

For production deployment on a server:

1. Run initialization on your local machine (faster)
2. Commit and push code changes
3. On server: `git pull`
4. Copy prediction buffers to server:
   ```bash
   scp data/oanda_cache/prediction_buffer_ann_*.pkl user@server:~/ForexTest/data/oanda_cache/
   ```
5. Setup cron job for daily execution (see [run_multi_pair_trader.sh](run_multi_pair_trader.sh))

## Questions?

Check existing documentation:
- [SLEEP_WELL_STRATEGY.md](SLEEP_WELL_STRATEGY.md) - Strategy overview
- [FUTURE_META_STRATEGY.md](FUTURE_META_STRATEGY.md) - Future plans

Or open an issue on GitHub.
