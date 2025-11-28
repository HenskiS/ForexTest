#!/bin/bash
# Multi-pair forex trader runner
# Runs daily at 2:30 PM PT / 5:30 PM ET
# Manages 4 pairs: EURUSD, GBPUSD, AUDUSD, USDJPY

cd /home/forex/ForexTest
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p /home/forex/logs

# Run multi-pair trader with tee to show output AND log it
# --yes flag skips confirmation prompts for automated runs
# --leverage 2.0 = 2:1 leverage ($250 per pair on $500 account)
python oanda_multi_pair_trader.py --live --leverage 2.0 --yes 2>&1 | tee -a /home/forex/logs/multi_trader_$(date +%Y_%m_%d).log

# Exit code from python, not tee
exit ${PIPESTATUS[0]}
