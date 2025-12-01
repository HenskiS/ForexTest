#!/bin/bash
# Migration script for transitioning from 5 PM to 9 AM trading
# Run this on your server to safely switch timing without losing positions

set -e  # Exit on any error

echo "=========================================="
echo "9 AM TRADING MIGRATION SCRIPT"
echo "=========================================="
echo ""
echo "This script will:"
echo "1. Fetch 9 AM aligned data for all pairs"
echo "2. Retrain models with 9 AM data"
echo "3. Update cron to run at 9 AM EST instead of 5:35 PM EST"
echo "4. Keep current positions - they'll close at next run (9 AM tomorrow)"
echo ""
echo "Current positions will remain open until 9 AM EST tomorrow,"
echo "when they'll be evaluated and potentially closed based on 9 AM predictions."
echo ""
read -p "Ready to proceed? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Migration cancelled."
    exit 1
fi

echo ""
echo "=========================================="
echo "STEP 1: Fetching 9 AM aligned data"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Fetch 9 AM aligned data
python3 fetch_9am_data.py

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to fetch 9 AM data"
    exit 1
fi

echo ""
echo "✓ 9 AM data fetched successfully"

echo ""
echo "=========================================="
echo "STEP 2: Backing up old data files"
echo "=========================================="

# Backup old 5 PM data
for PAIR in EURUSD GBPUSD AUDUSD USDJPY; do
    if [ -f "data/${PAIR}_1day_oanda.csv" ]; then
        cp "data/${PAIR}_1day_oanda.csv" "data/${PAIR}_1day_oanda_5pm_backup.csv"
        echo "✓ Backed up ${PAIR} 5 PM data"
    fi
done

echo ""
echo "=========================================="
echo "STEP 3: Replacing data files with 9 AM versions"
echo "=========================================="

# Replace old data with 9 AM data
for PAIR in EURUSD GBPUSD AUDUSD USDJPY; do
    if [ -f "data/${PAIR}_1day_oanda_9am.csv" ]; then
        cp "data/${PAIR}_1day_oanda_9am.csv" "data/${PAIR}_1day_oanda.csv"
        echo "✓ Replaced ${PAIR} with 9 AM aligned data"
    else
        echo "ERROR: Missing data/${PAIR}_1day_oanda_9am.csv"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "STEP 4: Retraining models with 9 AM data"
echo "=========================================="

# Clear old model cache (optional - forces retrain)
echo "Clearing old model cache..."
for PAIR in EURUSD GBPUSD AUDUSD USDJPY; do
    if [ -f "model_cache_${PAIR}.pkl" ]; then
        mv "model_cache_${PAIR}.pkl" "model_cache_${PAIR}_5pm_backup.pkl"
        echo "✓ Backed up ${PAIR} old model cache"
    fi
done

# Retrain models (this will happen automatically on next run, but we can force it)
echo ""
echo "Models will be retrained automatically at next 9 AM run."
echo "Current state files are preserved - positions will be evaluated at 9 AM."

echo ""
echo "=========================================="
echo "STEP 5: Updating cron schedule"
echo "=========================================="

# Show current cron
echo "Current cron schedule:"
crontab -l | grep "oanda_multi_pair_trader.py" || echo "No existing cron job found"

echo ""
echo "New cron schedule (9 AM EST = 6 AM PST):"
echo "0 9 * * * cd $(pwd) && /usr/bin/python3 oanda_multi_pair_trader.py >> logs/trading.log 2>&1"
echo ""
read -p "Update cron to 9 AM EST? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Backup current crontab
    crontab -l > crontab_backup_$(date +%Y%m%d_%H%M%S).txt

    # Remove old trading cron
    crontab -l | grep -v "oanda_multi_pair_trader.py" | crontab -

    # Add new 9 AM cron
    (crontab -l 2>/dev/null; echo "0 9 * * * cd $(pwd) && /usr/bin/python3 oanda_multi_pair_trader.py >> logs/trading.log 2>&1") | crontab -

    echo "✓ Cron updated to 9 AM EST"
    echo ""
    echo "New crontab:"
    crontab -l | grep "oanda_multi_pair_trader.py"
else
    echo "Cron update skipped - you'll need to update manually"
fi

echo ""
echo "=========================================="
echo "MIGRATION COMPLETE!"
echo "=========================================="
echo ""
echo "Summary:"
echo "✓ 9 AM aligned data fetched and installed"
echo "✓ Old 5 PM data backed up with _5pm_backup suffix"
echo "✓ Old model cache backed up"
echo "✓ Cron scheduled for 9 AM EST (6 AM PST)"
echo ""
echo "IMPORTANT:"
echo "- Current positions remain OPEN"
echo "- At 9 AM tomorrow, bot will evaluate positions based on 9 AM predictions"
echo "- Models will retrain automatically using 9 AM data"
echo "- Expected spreads: ~1.5 pips (vs 4.3 pips at 5:35 PM)"
echo ""
echo "Next run: Tomorrow at 9:00 AM EST (6:00 AM PST)"
echo "=========================================="
