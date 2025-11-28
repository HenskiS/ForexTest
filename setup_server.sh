#!/bin/bash
# Server Setup Script for Multi-Pair Trading System
# Run this after git pull to initialize data and buffers

set -e  # Exit on error

echo "================================================================================"
echo "MULTI-PAIR TRADING SYSTEM - SERVER SETUP"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Fetch OANDA data for 4 currency pairs"
echo "  2. Initialize prediction buffers for each pair"
echo "  3. Run a dry-run test"
echo "  4. Optionally update your cron job"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Create necessary directories
mkdir -p data
mkdir -p data/oanda_cache
mkdir -p /home/forex/logs

PAIRS=("EURUSD" "GBPUSD" "AUDUSD" "USDJPY")

# Step 1: Fetch OANDA data
echo ""
echo "================================================================================"
echo "STEP 1: FETCHING OANDA DATA"
echo "================================================================================"
echo ""

for PAIR in "${PAIRS[@]}"; do
    echo "Fetching $PAIR..."
    if python oanda_data_fetcher.py --pair "$PAIR" --count 5000 --live; then
        echo "  [OK] $PAIR data fetched"
    else
        echo "  [FAIL] Failed to fetch $PAIR data"
        exit 1
    fi
    echo ""
done

# Verify data files
echo "Verifying data files..."
for PAIR in "${PAIRS[@]}"; do
    if [ -f "data/${PAIR}_1day_oanda.csv" ]; then
        SIZE=$(du -h "data/${PAIR}_1day_oanda.csv" | cut -f1)
        echo "  [OK] data/${PAIR}_1day_oanda.csv ($SIZE)"
    else
        echo "  [FAIL] Missing data/${PAIR}_1day_oanda.csv"
        exit 1
    fi
done

# Step 2: Initialize prediction buffers
echo ""
echo "================================================================================"
echo "STEP 2: INITIALIZING PREDICTION BUFFERS"
echo "================================================================================"
echo ""
echo "This will take ~1-2 minutes per pair (total ~5-8 minutes)..."
echo ""

for PAIR in "${PAIRS[@]}"; do
    echo "Initializing $PAIR buffer..."
    if python initialize_oanda_buffer.py --pair "$PAIR" --live 2>&1 | grep -v "UnicodeEncodeError"; then
        echo "  [OK] $PAIR buffer initialized"
    else
        echo "  [FAIL] Failed to initialize $PAIR buffer"
        exit 1
    fi
    echo ""
done

# Verify buffer files
echo "Verifying prediction buffers..."
for PAIR in "${PAIRS[@]}"; do
    if [ -f "data/oanda_cache/prediction_buffer_${PAIR}.pkl" ]; then
        SIZE=$(du -h "data/oanda_cache/prediction_buffer_${PAIR}.pkl" | cut -f1)
        echo "  [OK] prediction_buffer_${PAIR}.pkl ($SIZE)"
    else
        echo "  [FAIL] Missing prediction_buffer_${PAIR}.pkl"
        exit 1
    fi
done

# Step 3: Run dry-run test
echo ""
echo "================================================================================"
echo "STEP 3: RUNNING DRY-RUN TEST"
echo "================================================================================"
echo ""

if python oanda_multi_pair_trader.py --live --dry-run --yes; then
    echo ""
    echo "  [OK] Dry-run test passed"
else
    echo ""
    echo "  [FAIL] Dry-run test failed"
    exit 1
fi

# Step 4: Cron job setup
echo ""
echo "================================================================================"
echo "STEP 4: CRON JOB SETUP"
echo "================================================================================"
echo ""
echo "Current cron jobs:"
crontab -l 2>/dev/null | grep -v "^#" | grep ForexTest || echo "  (none found)"
echo ""
echo "Recommended cron job for multi-pair trading:"
echo "  30 22 * * 0 /home/forex/ForexTest/run_multi_pair_trader.sh"
echo ""
echo "This runs every Sunday at 2:30 PM PT (before market opens at 5 PM ET)"
echo ""
read -p "Would you like to update your cron job now? (yes/no): " UPDATE_CRON

if [ "$UPDATE_CRON" = "yes" ] || [ "$UPDATE_CRON" = "y" ]; then
    # Make script executable
    chmod +x run_multi_pair_trader.sh

    # Backup current crontab
    crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null || true

    # Remove old ForexTest entries and add new one
    (crontab -l 2>/dev/null | grep -v "ForexTest"; echo "30 14 * * 0 /home/forex/ForexTest/run_multi_pair_trader.sh") | crontab -

    echo ""
    echo "  [OK] Cron job updated"
    echo ""
    echo "New cron schedule:"
    crontab -l | grep ForexTest
else
    echo ""
    echo "  Skipped cron job update"
    echo ""
    echo "To update manually later, run:"
    echo "  crontab -e"
    echo "And add this line:"
    echo "  30 22 * * 0 /home/forex/ForexTest/run_multi_pair_trader.sh"
fi

# Summary
echo ""
echo "================================================================================"
echo "SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  - Data fetched for 4 pairs: ${PAIRS[@]}"
echo "  - Prediction buffers initialized"
echo "  - Dry-run test passed"
echo "  - System ready for live trading"
echo ""
echo "Next run: Sunday at 2:30 PM PT"
echo ""
echo "What happens on next run:"
echo "  1. Checks and closes any existing positions (like your current EURUSD trade)"
echo "  2. Trains models for all 4 pairs"
echo "  3. Generates signals for all 4 pairs"
echo "  4. Enters new positions (up to 4 simultaneous)"
echo ""
echo "Position sizing: \$124.43 capital per pair x 2.0 leverage = \$248.86 per position"
echo ""
echo "Monitor logs in: /home/forex/logs/multi_trader_*.log"
echo ""
echo "================================================================================"
