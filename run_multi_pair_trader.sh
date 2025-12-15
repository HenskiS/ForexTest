#!/bin/bash
# Multi-pair forex trader runner - "Sleep Well" config
# Runs daily at 6:00 AM PT (14:00 UTC)
# 8 pairs, 10/90 thresholds, 5-day hold, 2.5% SL, 2.0x leverage

cd /home/forex/ForexTest
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p /home/forex/logs

# Run multi-pair trader with tee to show output AND log it
# --yes flag skips confirmation prompts for automated runs
python oanda_multi_pair_trader.py --live --yes 2>&1 | tee -a /home/forex/logs/multi_trader_$(date +%Y_%m_%d).log

# Exit code from python, not tee
exit ${PIPESTATUS[0]}
