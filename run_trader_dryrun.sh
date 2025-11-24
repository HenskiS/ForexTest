#!/bin/bash
# Dry-run version for testing (no real trades)

cd /home/forex/ForexTest
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p /home/forex/logs

# Run in dry-run mode
python oanda_production_trader.py --pair EURUSD --live --dry-run 2>&1 | tee -a /home/forex/logs/trader_dryrun_$(date +%Y_%m_%d).log

exit ${PIPESTATUS[0]}
