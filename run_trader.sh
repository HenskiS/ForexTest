#!/bin/bash
# Production forex trader runner
# Runs daily at 2:30 PM PT / 5:30 PM ET

cd /home/forex/ForexTest
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p /home/forex/logs

# Run trader with tee to show output AND log it
# --yes flag skips confirmation prompts for automated runs
python oanda_production_trader.py --pair EURUSD --live --yes 2>&1 | tee -a /home/forex/logs/trader_$(date +%Y_%m_%d).log

# Exit code from python, not tee
exit ${PIPESTATUS[0]}
