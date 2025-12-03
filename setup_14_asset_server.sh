#!/bin/bash
#
# Setup 14-Asset Trading Bot on Server (After Git Pull)
#
# This script configures the server to run the 14-asset trading bot after you've
# already done a git pull to update the code.
#
# Usage: ./setup_14_asset_server.sh
#

set -e  # Exit on error

echo "=================================================="
echo "SETTING UP 14-ASSET TRADING BOT"
echo "=================================================="
echo ""
echo "This will:"
echo "  1. Create necessary directories"
echo "  2. Install/update Python dependencies"
echo "  3. Set up cron job for 9 AM EST (14:00 UTC)"
echo "  4. Run a dry-run test"
echo ""

# Confirm setup
read -p "Continue with setup? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Setup cancelled."
    exit 0
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "Step 1: Creating directories..."
mkdir -p data logs trading

echo ""
echo "Step 2: Checking Python environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install dependencies
echo "Installing/updating dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Step 3: Setting up cron job for 9 AM EST daily execution..."

# Create daily execution script
cat > run_14_asset_trading.sh << 'EOF'
#!/bin/bash
# Daily 14-asset trading bot execution script
# Runs at 9 AM EST (14:00 UTC)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

LOG_DIR=$SCRIPT_DIR/logs
DATE=$(date +%Y-%m-%d)
LOG_FILE=$LOG_DIR/trading_$DATE.log

source venv/bin/activate

echo "====================================" >> $LOG_FILE
echo "14-Asset Trading Bot: $(date)" >> $LOG_FILE
echo "====================================" >> $LOG_FILE

# Run the 14-asset trader (use --yes to skip confirmations)
python3 oanda_14_asset_trader.py --live --yes >> $LOG_FILE 2>&1

echo "Finished: $(date)" >> $LOG_FILE
echo "" >> $LOG_FILE

# Keep only last 30 days of logs
find $LOG_DIR -name "trading_*.log" -mtime +30 -delete
EOF

chmod +x run_14_asset_trading.sh

# Add cron job (9 AM EST = 14:00 UTC)
echo "Installing cron job..."

# Remove old trading bot cron jobs
crontab -l 2>/dev/null | grep -v "oanda_.*_trader" | grep -v "run_.*_trading.sh" | crontab - 2>/dev/null || true

# Add new cron job at 14:00 UTC (9 AM EST)
(crontab -l 2>/dev/null; echo "0 14 * * * $SCRIPT_DIR/run_14_asset_trading.sh") | crontab -

echo "✓ Cron job installed:"
crontab -l | grep "run_14_asset_trading.sh"

echo ""
echo "Step 4: Running dry-run test..."
source venv/bin/activate
python3 oanda_14_asset_trader.py --live --dry-run --yes

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Dry-run test PASSED"
else
    echo ""
    echo "✗ Dry-run test FAILED - check errors above"
    exit 1
fi

echo ""
echo "=================================================="
echo "SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  - Trading bot: oanda_14_asset_trader.py"
echo "  - Schedule: Daily at 9 AM EST (14:00 UTC)"
echo "  - Mode: LIVE trading (real money)"
echo "  - Assets: 14 (4 Forex + 5 Metals + 5 Commodities/Indices)"
echo "  - Allocation: 7.14% per asset ($35.71 capital each)"
echo "  - Leverage: 2x ($71.43 position per asset on $500 account)"
echo "  - Expected return: 78.5% annual @ 1x, 156.9% @ 2x"
echo "  - Sharpe ratio: 7.60 (exceptional)"
echo "  - Max drawdown: -2.63% (portfolio level)"
echo "  - Logs: $SCRIPT_DIR/logs/"
echo ""
echo "Asset Breakdown:"
echo "  Forex (4): EURUSD, GBPUSD, AUDUSD, USDJPY"
echo "  Metals (5): XAUUSD, XAGUSD, XPTUSD, XPDUSD, XCUUSD"
echo "  Commodities/Indices (5): SUGARUSD, SPX500USD, DE30EUR, WTICOUSD, BCOUSD"
echo ""
echo "Spread Monitoring:"
echo "  - Checks spreads before every trade"
echo "  - Skips assets with spreads > EV-based thresholds"
echo "  - Protects against 15-20% annual degradation"
echo "  - Expected to skip 10-15% of trading days"
echo ""
echo "Useful commands:"
echo "  - View cron schedule: crontab -l"
echo "  - Monitor today's log: tail -f logs/trading_\$(date +%Y-%m-%d).log"
echo "  - Manual dry-run: ./venv/bin/python3 oanda_14_asset_trader.py --live --dry-run"
echo "  - Manual live run: ./venv/bin/python3 oanda_14_asset_trader.py --live --yes"
echo "  - Check spreads: ./venv/bin/python3 spread_monitor.py"
echo ""
echo "IMPORTANT: Bot will start trading automatically at next 9 AM EST!"
echo "=================================================="
