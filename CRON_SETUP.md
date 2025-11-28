# Cron Job Setup for Automated Trading

## Overview

Two scripts run on your server:
1. **Daily Trader** (`run_trader.sh`) - Places new trades once per day
2. **Position Monitor** (`monitor_positions.sh`) - Checks hourly if positions closed

## Cron Schedule

Edit your crontab on the server:
```bash
crontab -e
```

Add these lines:

```bash
# Daily trader - runs at 2:30 PM PT (5:30 PM ET) on weekdays
30 22 * * 1-5 /home/forex/ForexTest/run_trader.sh

# Position monitor - runs every hour, 7 days a week
0 * * * * /home/forex/ForexTest/monitor_positions.sh
```

## Script Permissions

Make sure scripts are executable:
```bash
chmod +x /home/forex/ForexTest/run_trader.sh
chmod +x /home/forex/ForexTest/monitor_positions.sh
```

## What Each Script Does

### Daily Trader (run_trader.sh)
- Runs daily at 2:30 PM PT
- Trains model on latest data
- Checks if existing position should close
- Places new trade if signal generated
- Logs to `/home/forex/logs/trader_YYYY_MM_DD.log`

### Position Monitor (monitor_positions.sh)
- Runs every hour
- Checks if state file shows open position
- Verifies position still exists at OANDA
- If position closed at OANDA:
  - Calculates final P&L
  - Sends Telegram notification
  - Clears state file
- Logs to `/home/forex/logs/monitor_YYYY_MM_DD.log`

## Testing

Test manually before adding to cron:

```bash
# Test daily trader (dry-run mode)
cd /home/forex/ForexTest
source venv/bin/activate
python oanda_production_trader.py --pair EURUSD --live --yes --dry-run

# Test position monitor
python monitor_positions.py --pair EURUSD --live
```

## Monitoring Logs

Check today's logs:
```bash
# Trader log
tail -f /home/forex/logs/trader_$(date +%Y_%m_%d).log

# Monitor log
tail -f /home/forex/logs/monitor_$(date +%Y_%m_%d).log
```

## Timezone Notes

- Cron uses server timezone (likely UTC)
- Forex market hours: Sunday 5 PM ET to Friday 5 PM ET
- The trader script checks if market is open before executing

Convert PT to your server's timezone for cron schedule:
- 2:30 PM PT = 5:30 PM ET = 10:30 PM UTC (standard time)
- 2:30 PM PT = 5:30 PM ET = 9:30 PM UTC (daylight time)

Adjust cron times accordingly based on your server's timezone.
