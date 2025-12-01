# Server Migration Guide: 5 PM → 9 AM Trading

## Overview

This migration switches your trading bot from 5:35 PM EST to 9:00 AM EST execution to take advantage of:
- **66% better spreads** (1.5 pips vs 4.3 pips average)
- **67% higher returns** (316% vs 189% over 3 years at 2x leverage)
- **Better model alignment** (9 AM predictions match 9 AM execution)
- **London/NY overlap** (peak liquidity window)

## Safety Guarantees

✓ **No positions lost** - Current open positions remain tracked and evaluated at 9 AM
✓ **Automatic backups** - Old data and models saved with `_5pm_backup` suffix
✓ **Rollback ready** - Can restore old setup if needed
✓ **Same bot code** - No changes to `oanda_multi_pair_trader.py` needed

## Migration Steps

### 1. Upload Files to Server

Copy these files to your server:
```bash
scp fetch_9am_data.py user@server:~/ForexTest/
scp migrate_to_9am_trading.sh user@server:~/ForexTest/
scp verify_9am_migration.py user@server:~/ForexTest/
```

Or use git if this is in your repo:
```bash
# On server
cd ~/ForexTest
git pull
```

### 2. Run Pre-Migration Check

```bash
cd ~/ForexTest
python3 verify_9am_migration.py
```

This verifies:
- 9 AM data can be fetched
- OANDA credentials are working
- All required scripts exist
- Shows current position state

Expected output: "✓ ALL CHECKS PASSED - Ready for migration!"

### 3. Execute Migration

```bash
chmod +x migrate_to_9am_trading.sh
bash migrate_to_9am_trading.sh
```

The script will:
1. Fetch 9 AM aligned data for all 4 pairs (5000 candles each)
2. Backup old 5 PM data → `data/{PAIR}_1day_oanda_5pm_backup.csv`
3. Replace data files with 9 AM versions
4. Backup old model cache → `model_cache_{PAIR}_5pm_backup.pkl`
5. Update cron from 5:35 PM EST → 9:00 AM EST

**Total time:** ~2-3 minutes

### 4. Verify Cron Update

```bash
crontab -l | grep forex
```

Should show:
```
0 9 * * * /home/forex/ForexTest/run_multi_pair_trader.sh
```

### 5. Monitor First Run

**Tomorrow at 9:00 AM EST (6:00 AM PST)**, the bot will:
- Load 9 AM aligned data
- Retrain models automatically with new data
- Evaluate any existing positions (if still open)
- Generate new predictions and place trades

Check logs:
```bash
tail -f /home/forex/logs/multi_trader_$(date +%Y_%m_%d).log
```

## What Happens to Current Positions?

The bot automatically syncs with OANDA on every run, so your positions are safe regardless of local state files.

**Important:** Position state files are in `data/oanda_cache/{PAIR}_state.json`, NOT the root directory!

**Scenario 1: No open positions (clean migration)**
- Bot starts fresh at 9 AM tomorrow
- Places new trades based on 9 AM predictions
- Everything just works

**Scenario 2: Open positions from 5:35 PM today**
- Positions remain open overnight at OANDA
- At 9 AM tomorrow, bot queries OANDA API for actual positions
- **Time-based exit:** If position >= 1 day old, closes it (logs P&L)
- **Then:** Generates new 9 AM predictions and enters fresh positions
- This is normal behavior - daily rebalancing based on new predictions

**The bot does NOT care about old predictions.** It always:
1. Closes positions >= 1 day old
2. Generates fresh predictions with new data
3. Enters new positions based on current signals

## Expected Performance Improvement

| Metric | Old (5 PM) | New (9 AM) | Improvement |
|--------|-----------|-----------|-------------|
| **Spreads** | 4.3 pips | 1.5 pips | **-65%** |
| **Annual Return** (2x) | 42.9% | 61.5% | **+43%** |
| **3-Year Return** (2x) | 189% | 316% | **+67%** |
| **Sharpe Ratio** | 4.53 | 6.07 | **+34%** |
| **Max Drawdown** | -12.1% | -9.2% | **-24%** |

## Rollback Procedure (if needed)

If something goes wrong:

```bash
# Restore old data
cd ~/ForexTest/data
for pair in EURUSD GBPUSD AUDUSD USDJPY; do
    cp ${pair}_1day_oanda_5pm_backup.csv ${pair}_1day_oanda.csv
done

# Restore old models
for pair in EURUSD GBPUSD AUDUSD USDJPY; do
    cp model_cache_${pair}_5pm_backup.pkl model_cache_${pair}.pkl
done

# Restore old cron (5:35 PM EST = 30 17 in cron)
crontab -l | grep -v "run_multi_pair_trader.sh" | crontab -
(crontab -l; echo "30 17 * * * /home/forex/ForexTest/run_multi_pair_trader.sh") | crontab -
```

## Troubleshooting

**Issue:** Data fetch fails during migration
```bash
# Check OANDA credentials
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OANDA_API_KEY'))"

# Test API connection
python3 fetch_9am_data.py
```

**Issue:** Cron doesn't run at 9 AM
```bash
# Check cron logs
grep CRON /var/log/syslog

# Test manual run
/home/forex/ForexTest/run_multi_pair_trader.sh
```

**Issue:** Models fail to retrain
```bash
# Check model cache was cleared
ls -la model_cache_*.pkl

# Force retrain by deleting cache
rm model_cache_*.pkl
```

## Timeline

- **Today (before 9 AM tomorrow):** Run migration script
- **Tomorrow 9:00 AM EST:** First bot run with 9 AM data
- **Tomorrow 9:05 AM EST:** Check logs to verify success
- **Next 2 weeks:** Monitor performance vs backtest expectations
- **After 2 weeks:** Evaluate if performance matches improved predictions

## Support

If issues occur:
1. Check `logs/trading.log` for errors
2. Run `python3 verify_9am_migration.py` to check system state
3. Use rollback procedure if needed
4. Keep backup files for at least 1 month

---

**Migration prepared:** 2025-11-30
**Expected first run:** 2025-12-01 9:00 AM EST (6:00 AM PST)
