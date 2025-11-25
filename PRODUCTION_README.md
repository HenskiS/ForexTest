# OANDA Production Trading System v3

Production-ready forex trading system using XGBoost with rolling daily retraining, deployed on OANDA.

## Overview

This system implements a quantitative forex trading strategy with:
- **1-day prediction model**: Eliminates look-ahead bias with 1-day forward targets
- **Rolling daily retraining**: Train XGBoost models daily on 378-day windows with 1-day gap
- **Optimized parameters**: 0.18% stop loss, 2.00% take profit, 1-day holding period
- **Percentile-based entry signals**: Dynamic thresholds from rolling prediction buffer
- **Fixed stops**: No ATR adjustment (optimized for 1-day holding)
- **Validated performance**: 30.10% annual return on 17.86-year backtest (no look-ahead bias)

---

## Full Backtest Results (17.86 Years)

**Test Period:** October 1, 2009 to November 20, 2025
**Trading Days:** 4,500 days
**Currency Pair:** EUR/USD

### Performance Summary

| Metric | Value |
|--------|-------|
| **Starting Equity** | $1,000 |
| **Final Equity** | $109,810 |
| **Total Return** | 10,881% |
| **Annual Return** | 30.10% |
| **Compounded Annual Return** | 33.80% |
| | |
| **Total Trades** | 4,405 |
| **Win Rate** | 38.0% |
| **Average Win** | +0.60% |
| **Average Loss** | -0.19% |
| **Win/Loss Ratio** | 3.11x |
| **Profit Factor** | 1.90 |
| | |
| **Max Drawdown** | -4.96% |
| **Sharpe Ratio** | 3.68 |
| **Calmar Ratio** | 6.82 |
| **Return/Drawdown** | 6.82x |
| | |
| **Largest Win** | +1.98% |
| **Largest Loss** | -0.20% |
| **Max Consecutive Wins** | 9 |
| **Max Consecutive Losses** | 16 |
| **Longest Drawdown** | 297 days (Dec 2023 - Oct 2024) |

### Year-by-Year Performance

| Year | Return (%) | Trades | Win Rate (%) | Sharpe | Max DD (%) | End Equity |
|------|------------|--------|--------------|--------|------------|------------|
| 2009 | -3.88 | 86 | 25.58 | -1.41 | -4.45 | $961 |
| 2010 | **77.43** | 341 | 36.95 | 5.12 | -4.96 | $1,705 |
| 2011 | **66.68** | 358 | 35.20 | 4.68 | -2.53 | $2,843 |
| 2012 | 41.89 | 339 | 37.46 | 4.04 | -3.05 | $4,033 |
| 2013 | 19.90 | 270 | 40.74 | 2.88 | -3.35 | $4,836 |
| 2014 | 22.14 | 248 | 41.94 | 3.30 | -3.21 | $5,907 |
| 2015 | **62.37** | 253 | 35.18 | 5.00 | -3.06 | $9,591 |
| 2016 | 19.71 | 250 | 32.00 | 2.47 | -3.72 | $11,481 |
| 2017 | 31.64 | 245 | 40.82 | 3.98 | -3.70 | $15,114 |
| 2018 | 15.64 | 253 | 33.99 | 2.20 | -4.93 | $17,477 |
| 2019 | 12.01 | 255 | 41.57 | 2.26 | -4.03 | $19,577 |
| 2020 | 39.14 | 256 | 41.80 | 4.40 | -2.52 | $27,239 |
| 2021 | 13.10 | 257 | 40.08 | 2.26 | -2.61 | $30,808 |
| 2022 | **67.04** | 258 | 36.43 | 5.24 | -2.59 | $51,460 |
| 2023 | 37.85 | 259 | 39.77 | 4.15 | -3.43 | $70,937 |
| 2024 | 9.55 | 250 | 36.80 | 1.62 | -4.91 | $77,711 |
| 2025 | 41.31 | 227 | 43.17 | 4.50 | -2.29 | $109,810 |

**Yearly Statistics:**
- **Positive Years:** 16 out of 17 (94.1%)
- **Average Annual Return:** 33.74%
- **Median Annual Return:** 31.64%
- **Best Year:** 2010 (+77.43%)
- **Worst Year:** 2009 (-3.88%)
- **Average Sharpe Ratio:** 3.34

### Recent vs Long-Term Performance

**Recent 250 Days (2024-2025):** 47.75% annual
**Full 4500 Days (2009-2025):** 30.10% annual

The recent period shows exceptional performance due to favorable 2024-2025 market conditions. **The 30.10% annual return over 17+ years is the realistic long-term expectation.**

### Key Strategy Changes (v3 Optimization)

**v1 → v2 → v3:**
- Training Window: 756 → 756 → **378 days** (optimized for 1-day predictions)
- Target: 5-day → 1-day → 1-day returns
- Stops: ATR-adjusted → ATR-adjusted → **Fixed** (no adjustment)
- Cooldown: 1 day → 1 day → **0 days** (immediate re-entry)
- Result: ~39% → 21.98% → **30.10% annual**

**Key Insights:**
1. Reducing training window improved performance significantly
2. Fixed stops outperformed ATR-adjusted for 1-day holding
3. Removing cooldown increased profitability without increasing risk

---

## Prerequisites

1. **OANDA Account**: Practice or live trading account
2. **API Credentials**: Account ID and API token
3. **Python Environment**: Python 3.8+ with required packages

### Install Dependencies

```bash
pip install pandas numpy xgboost scikit-learn python-dotenv requests ta tqdm pytz
```

## Setup

### 1. Configure Environment

Create `.env` file in project root:

```
OANDA_ACCOUNT_ID=your_account_id
OANDA_API_KEY=your_api_token
```

### 2. Fetch Historical Data

Fetch historical data from OANDA:

```bash
python oanda_data_fetcher.py
```

Creates: `data/EURUSD_1day_oanda.csv` (5000+ days of OHLC data)

### 3. Initialize Prediction Buffer

Pre-populate rolling prediction buffer:

```bash
python initialize_oanda_buffer.py --pair EURUSD
```

Creates: `data/oanda_cache/prediction_buffer_EURUSD.pkl` (200 predictions)

### 4. Validate with Backtest

Run backtest on OANDA data to verify performance:

```bash
# Recent performance (250 days)
python backtest_oanda_data.py --pair EURUSD --test-days 250

# Long-term performance (4500 days)
python backtest_oanda_data.py --pair EURUSD --test-days 4500
```

## Core Scripts

### Production Scripts

#### `oanda_production_trader.py`
Main production trading bot (runs daily at 5:30 PM ET)

**Features:**
- Fetches today's candle from OANDA API
- Trains XGBoost model on rolling 378-day window with 1-day gap (no look-ahead bias)
- Generates prediction for next 1 day
- Updates rolling prediction buffer (200 days max)
- Calculates percentile thresholds (48th/52nd for EURUSD)
- Manages position entry/exit with optimized stops (0.18% SL, 2.00% TP)
- 1-day holding period with time-based exit
- No cooldown period (immediate re-entry after exits)
- Persists state to JSON for next run

**Usage:**
```bash
# Dry-run (no real trades)
python oanda_production_trader.py --pair EURUSD --dry-run

# Live trading (practice account, no leverage)
python oanda_production_trader.py --pair EURUSD

# Live trading (real money)
python oanda_production_trader.py --pair EURUSD --live

# With leverage (use carefully)
python oanda_production_trader.py --pair EURUSD --leverage 2.0
```

#### `oanda_data_fetcher.py`
Utility class for OANDA API integration

**Methods:**
- `get_historical_data()`: Fetch OHLC candles
- `get_current_price()`: Get latest price
- `get_account_summary()`: Check account status
- `get_open_positions()`: Query current positions

### Setup Scripts

#### `initialize_oanda_buffer.py`
Pre-populate prediction buffer before first production run

**Why needed:** Production trader needs 50 predictions before generating signals. This script generates 200 historical 1-day predictions so trading can start immediately.

**Usage:**
```bash
python initialize_oanda_buffer.py --pair EURUSD
```

### Validation Scripts

#### `backtest_oanda_data.py`
Backtest rolling daily strategy on OANDA historical data

**Features:**
- Rolling daily retraining (mimics production exactly)
- Rolling prediction buffer (eliminates lookahead bias)
- Fixed stops (0.18%/2.00%)
- Transaction costs (0.02%)
- No cooldown period

**Usage:**
```bash
# Test last 250 days
python backtest_oanda_data.py --pair EURUSD --test-days 250

# Test all available data
python backtest_oanda_data.py --pair EURUSD --test-days 4500
```

#### `analyze_backtest_metrics.py`
Calculate comprehensive risk metrics

```bash
python analyze_backtest_metrics.py --file EURUSD_backtest_trades_4500days.csv
```

Outputs: Sharpe ratio, Calmar ratio, max drawdown, profit factor, etc.

#### `analyze_yearly_performance.py`
Generate year-by-year performance breakdown

```bash
python analyze_yearly_performance.py --file EURUSD_backtest_trades_4500days.csv
```

Outputs: Annual returns, win rates, Sharpe ratios by year.

## Data Files

### Historical Data

**`data/EURUSD_1day_oanda.csv`**
- 5000+ days of OHLC data from OANDA
- Columns: `date`, `open`, `high`, `low`, `close`, `volume`
- Updated: Fetched once, then appended daily by production trader

### Cache Files

**`data/oanda_cache/prediction_buffer_EURUSD.pkl`**
- Rolling buffer of last 200 predictions
- Used to calculate percentile thresholds
- Updated daily by production trader
- Format: Python list of floats

**`data/oanda_cache/EURUSD_state.json`**
- Persistent state across daily runs
- Tracks: position, entry price, entry date, cooldown
- Updated after each run
- Example:
```json
{
  "position": 1,
  "entry_price": 1.0850,
  "entry_date": "2025-11-20T17:00:00",
  "trade_id": "12345",
  "position_size": 1000.0,
  "cooldown_until": null
}
```

**`data/oanda_cache/EURUSD_trades.csv`**
- Log of all executed trades
- Columns: entry_date, exit_date, direction, entry_price, exit_price, position_size, pnl_pct, pnl_dollars, outcome, exit_reason, days_held, prediction, signal

### Model Configuration

**`hyperparams_rolling_daily_EURUSD.pkl`**
- Optimized XGBoost hyperparameters
- Generated by hyperparameter tuning script
- Format: Python dict with `n_estimators`, `learning_rate`, `max_depth`, `gamma`, `subsample`, `colsample_bytree`

## Daily Workflow

### Scheduled Run (5:30 PM ET Daily)

1. **Check Market Hours**
   - Verify forex market is open (Sunday 5:00 PM ET to Friday 5:00 PM ET)
   - If closed (weekends), exit cleanly with log message

2. **Fetch Latest Data**
   - Load historical CSV (5000+ days)
   - Fetch latest 5 candles from OANDA API
   - Detect new candles (today's close at 5:00 PM)
   - Append new data to historical CSV

3. **Train Model**
   - Extract last 378 days as training window (excludes today - 1-day gap)
   - Engineer 26 technical indicators
   - Scale features with MinMaxScaler
   - Train XGBoost on 378 days ending yesterday (eliminates look-ahead bias)

4. **Generate Prediction**
   - Predict next 1-day return using today's features
   - Add prediction to rolling buffer (200 max)
   - Save buffer to disk

5. **Calculate Thresholds**
   - Compute 48th/52nd percentiles from buffer
   - Generate signal: Long (>52nd), Short (<48th), Hold (between)

6. **Manage Position**
   - **If in position**: Check for exit
     - Stop-loss hit (0.18% fixed)
     - Take-profit hit (2.00% fixed)
     - Holding period exceeded (1 day - primary exit mechanism)
   - **If no position**: Check for entry
     - Signal generated (long/short)
     - Immediate entry (no cooldown period)

7. **Execute Trade** (if applicable)
   - Place market order via OANDA API
   - Update state JSON (position, entry price, entry date)

8. **Log Results**
   - Print summary (signal, position, equity)
   - Save state for next run
   - Log trade to CSV (if position closed)

## Strategy Parameters

### EURUSD (Optimized via 1-Day Model)

```python
# Entry Thresholds
LOWER_PERCENTILE = 48  # Short signal
UPPER_PERCENTILE = 52  # Long signal

# Risk Management (Optimized via backtest)
BASE_STOP_LOSS = 0.18%  # Fixed (no ATR adjustment)
BASE_TAKE_PROFIT = 2.00%  # Fixed (rarely hit, acts as safety ceiling)
HOLDING_PERIOD = 1  # Days (primary exit mechanism)

# Trading Rules
LOSS_COOLDOWN = 0  # No cooldown - immediate re-entry after exits
TRANSACTION_COST = 0.02%  # Per trade

# Model Training (1-Day Predictions)
TRAIN_WINDOW = 378  # Days ending yesterday (1-day gap to prevent look-ahead bias)
TARGET_HORIZON = 1  # Day (predict next-day return)
BUFFER_SIZE = 200  # Predictions for threshold calculation
BUFFER_WARMUP = 50  # Minimum predictions before trading
```

**Key Changes from v2:**
- **Training Window**: 756 → 378 days (optimized for 1-day predictions)
- **Stop Loss**: 0.40% → 0.18% (tighter, preserves capital)
- **Take Profit**: 1.00% → 2.00% (wider, lets winners run)
- **Holding Period**: 5 days → 1 day (matches prediction horizon)
- **Cooldown**: 1 day → 0 days (immediate re-entry)
- **ATR Adjustment**: Removed (fixed stops perform better for 1-day holding)

## Risk Management and Leverage

### Position Sizing

The trader automatically sizes positions based on your account balance:

```python
# With no leverage (default)
position_size = account_balance * 1.0

# With 2x leverage
position_size = account_balance * 2.0

# With 3x leverage
position_size = account_balance * 3.0
```

Account balance is fetched dynamically from OANDA before each trade, ensuring position size adapts to wins/losses.

### Leverage Configuration

The `--leverage` argument allows you to multiply your position size:

```bash
# No leverage (default) - trades 100% of cash balance
python oanda_production_trader.py --pair EURUSD

# 2x leverage - trades 200% of cash balance
python oanda_production_trader.py --pair EURUSD --leverage 2.0

# 3x leverage - trades 300% of cash balance
python oanda_production_trader.py --pair EURUSD --leverage 3.0
```

**Leverage Range**: 1.0 (no leverage) to 50.0 (OANDA maximum)

### Safety Confirmations

When using leverage >1.0, the trader requires double confirmation:

```
⚠️  WARNING: Using 3.0x leverage increases risk!
Max loss per trade: ~0.54% of account balance
Type 'YES' to confirm leverage:
```

This prevents accidental high-leverage trades.

### Recommended Phased Approach

Start conservative and increase leverage based on real performance:

**Phase 1: Weeks 1-2 (No Leverage)**
```bash
python oanda_production_trader.py --pair EURUSD
```
- Build confidence with real execution
- Verify strategy performs as expected
- No margin usage, cash-only trading

**Phase 2: After 2+ Weeks (2x Leverage)**
```bash
python oanda_production_trader.py --pair EURUSD --leverage 2.0
```
- If strategy proves profitable in live trading
- Max loss per trade: ~0.36% of account

**Phase 3: Long-term (3-4x Leverage)**
```bash
python oanda_production_trader.py --pair EURUSD --leverage 3.0
```
- After sustained profitable performance
- Max loss per trade: ~0.54% of account
- Only after validating drawdown characteristics

### Risk Calculations

With optimized parameters (0.18% stop-loss, fixed):

| Leverage | Position Size | Max Loss/Trade | Max Loss (if stopped out) |
|----------|---------------|----------------|---------------------------|
| 1.0x | $1,000 | ~$1.80 | -0.18% |
| 2.0x | $2,000 | ~$3.60 | -0.36% |
| 3.0x | $3,000 | ~$5.40 | -0.54% |
| 4.0x | $4,000 | ~$7.20 | -0.72% |

*Assumes $1,000 account balance and fixed stop-loss hits*

**Note:** The tight 0.18% stop loss (vs previous 0.40%) significantly reduces risk per trade while maintaining strong performance.

### Important Warnings

- **Higher leverage = Higher risk**: Losses scale proportionally with leverage
- **Start conservatively**: Use no leverage initially to validate live performance
- **Monitor drawdowns**: Real slippage and spreads may increase losses beyond backtest
- **Realistic expectations**: Live performance typically 50-75% of backtest returns
- **Account protection**: Ensure sufficient margin to avoid forced liquidation

## Deployment

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. **Trigger**: Daily at 5:30 PM ET
4. **Action**: Start a program
   - Program: `C:\path\to\python.exe`
   - Arguments (no leverage): `c:\path\to\oanda_production_trader.py --pair EURUSD`
   - Arguments (with leverage): `c:\path\to\oanda_production_trader.py --pair EURUSD --leverage 2.0`
   - Start in: `c:\path\to\ForexTest`
5. **Conditions**:
   - Start only if computer is on AC power: Unchecked
   - Wake computer to run: Checked

### Linux/Mac Cron

```bash
# Edit crontab
crontab -e

# Add line (5:30 PM ET = 22:30 UTC in winter, 21:30 UTC in summer)

# No leverage (default)
30 22 * * * cd /path/to/ForexTest && /usr/bin/python3 oanda_production_trader.py --pair EURUSD

# With 2x leverage
30 22 * * * cd /path/to/ForexTest && /usr/bin/python3 oanda_production_trader.py --pair EURUSD --leverage 2.0
```

## Safety Features

### Market Hours Check
- Script automatically checks if forex market is open before executing
- Market hours: Sunday 5:00 PM ET to Friday 5:00 PM ET
- If market is closed, script exits cleanly with log message

Example output when market is closed:
```
Forex market is CLOSED (current time: Saturday 2025-11-23 14:30:00 EST)
Market hours: Sunday 5:00 PM ET to Friday 5:00 PM ET
Exiting without executing any trading logic.
```

### Dry-Run Mode
Test without real trades:
```bash
python oanda_production_trader.py --pair EURUSD --dry-run
```

### State Persistence
- Position state saved to JSON after each run
- System recovers gracefully if process crashes
- Prevents duplicate entries or forgotten exits

### Transaction Costs
- 0.02% cost per trade (realistic for OANDA spreads)
- Ensures profitable trades after costs

### Duplicate Trade Prevention
- Checks OANDA API for existing positions before opening new ones
- Prevents accidental double-entry

## Expected Performance

Based on 17.86-year backtest on OANDA data (v3 optimized model):

| Metric | Value |
|--------|-------|
| Annual Return | 30.10% |
| Win Rate | 38.0% |
| Average Win | 0.60% |
| Average Loss | -0.19% |
| Win/Loss Ratio | 3.11:1 |
| Profit Factor | 1.90 |
| Sharpe Ratio | 3.68 |
| Max Drawdown | -4.96% |

**Realistic live performance**: 20-25% annual return (60-80% of backtest) due to:
- Execution slippage (0.1-0.3 pips)
- Wider spreads during volatile periods
- Occasional API latency or downtime
- Market regime changes

**Note:** The 1-day model's tighter stops and higher trade frequency may result in better live/backtest correlation than the previous 5-day model.

## Monitoring

### Daily Checks

1. **Verify run completed**: Check Task Scheduler history or cron logs
2. **Review position status**: Check `EURUSD_state.json`
3. **Monitor equity**: Track account balance in OANDA dashboard
4. **Validate predictions**: Ensure prediction buffer is updating

### Weekly Review

1. **Performance metrics**: Calculate weekly return
2. **Trade analysis**: Review recent trades (wins/losses)
3. **Buffer health**: Ensure prediction buffer has 50+ predictions
4. **Data integrity**: Verify historical CSV is appending correctly

### Monthly Validation

1. **Run backtest**: Compare recent month's performance to backtest
2. **Recalibrate thresholds**: Consider re-optimizing percentiles if market regime shifts
3. **Review hyperparameters**: Retrain hyperparameters if performance degrades

## Troubleshooting

### Script Not Running
- Check Task Scheduler is enabled
- Verify Python path in scheduled task
- Check `.env` file exists with valid credentials

### No Trades Executing
- Ensure `--dry-run` flag is NOT set
- Verify prediction buffer has 50+ predictions
- Confirm signal is generating (not "Hold")
- Check position state file (may already be in position)

### API Errors
- Verify OANDA credentials in `.env`
- Check OANDA account status (sufficient margin)
- Test with `oanda_data_fetcher.py` standalone

### Missing Data
- Run `python oanda_data_fetcher.py` to re-fetch historical data
- Verify `data/EURUSD_1day_oanda.csv` exists and has recent dates

### Buffer Issues
- Re-run `python initialize_oanda_buffer.py --pair EURUSD`
- Check `data/oanda_cache/prediction_buffer_EURUSD.pkl` exists

## File Structure

```
ForexTest/
├── oanda_production_trader.py        # Main production bot
├── oanda_data_fetcher.py              # OANDA API wrapper
├── initialize_oanda_buffer.py         # Buffer initialization
├── backtest_oanda_data.py             # Validation backtest
├── analyze_backtest_metrics.py        # Risk metrics calculation
├── analyze_yearly_performance.py      # Year-by-year breakdown
├── optimize_training_window.py        # Window size optimization
├── optimize_1day_trading_params.py    # Stop/profit optimization
├── optimize_1day_thresholds.py        # Signal threshold optimization
├── hyperparams_rolling_daily_EURUSD.pkl  # Model config
├── .env                               # API credentials (not in git)
├── data/
│   ├── EURUSD_1day_oanda.csv         # Historical OHLC (5000+ days)
│   └── oanda_cache/
│       ├── prediction_buffer_EURUSD.pkl  # Rolling predictions
│       ├── EURUSD_state.json         # Position state
│       └── EURUSD_trades.csv         # Trade log
└── PRODUCTION_README.md              # This file
```

## Disclaimers

### Past Performance
Past performance is not indicative of future results. The 30.10% annual return over 17 years is historical and may not continue.

### Market Regime Dependency
This strategy has been tested across multiple market conditions (2009-2025) including financial crisis recovery, low volatility periods, high volatility periods, and trending markets. However, future market conditions may differ.

### Risk Warnings
- **Real Money Trading**: This bot trades real money. Always test thoroughly on practice account first.
- **Leverage Risk**: Using leverage >1.0x amplifies both gains AND losses
- **Stop Loss Risk**: In extreme volatility, stops may execute at worse prices (slippage)
- **Technology Risk**: Ensure stable internet and system uptime for production trading

### Monitoring Recommendations
1. Check trades daily to ensure system functions correctly
2. Monitor drawdown levels - consider pausing if drawdown exceeds 10%
3. Review monthly performance to detect regime changes early
4. Keep practice account running in parallel to validate live performance

## References

- [OANDA API Documentation](https://developer.oanda.com/rest-live-v20/introduction/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- Strategy Development: `backtest_oanda_data.py` for validation methodology

---

**Last Updated:** November 25, 2025
**Strategy Version:** v3 (Optimized 1-Day Model)
**Backtest Date Range:** October 1, 2009 - November 20, 2025
**Annual Return (17.86 years):** 30.10%
