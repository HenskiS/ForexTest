#!/bin/bash
# Position monitor runner
# Runs hourly to check if positions have closed

cd /home/forex/ForexTest
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p /home/forex/logs

# Run monitor with tee to show output AND log it
python monitor_positions.py --pair EURUSD --live 2>&1 | tee -a /home/forex/logs/monitor_$(date +%Y_%m_%d).log

# Exit code from python, not tee
exit ${PIPESTATUS[0]}
