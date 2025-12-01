"""
Pre-migration verification script.
Run this BEFORE migrate_to_9am_trading.sh to ensure everything is ready.
"""
import os
import sys
import pandas as pd
from datetime import datetime

# Fix Windows Unicode encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']

print("=" * 80)
print("9 AM MIGRATION VERIFICATION")
print("=" * 80)
print()

all_checks_passed = True

# Check 1: Verify 9 AM data files exist
print("Check 1: 9 AM data files")
print("-" * 80)
for pair in PAIRS:
    filename = f'data/{pair}_1day_oanda_9am.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ {pair}: {len(df)} candles, {df['date'].min()} to {df['date'].max()}")
    else:
        print(f"✗ {pair}: MISSING! Run fetch_9am_data.py first")
        all_checks_passed = False

print()

# Check 2: Verify old data files exist (for backup)
print("Check 2: Old data files (will be backed up)")
print("-" * 80)
for pair in PAIRS:
    filename = f'data/{pair}_1day_oanda.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        print(f"✓ {pair}: {len(df)} candles (will backup to _5pm_backup.csv)")
    else:
        print(f"⚠ {pair}: No existing data (fresh install?)")

print()

# Check 3: Check for current position state files
print("Check 3: Current position state")
print("-" * 80)
has_positions = False
for pair in PAIRS:
    state_file = f'data/oanda_cache/{pair}_state.json'
    if os.path.exists(state_file):
        import json
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            if state.get('position') != 0:
                print(f"✓ {pair}: Position state found (will preserve)")
                print(f"  → Direction: {'LONG' if state['position'] == 1 else 'SHORT'}")
                print(f"  → Entry: {state['entry_price']} @ {state['entry_date']}")
                has_positions = True
            else:
                print(f"○ {pair}: No position state (no open position)")
        except Exception as e:
            print(f"⚠ {pair}: State file exists but couldn't read: {e}")
    else:
        print(f"○ {pair}: No position state file found")

print()

# Check 4: Verify fetch_9am_data.py exists
print("Check 4: Required scripts")
print("-" * 80)
if os.path.exists('fetch_9am_data.py'):
    print("✓ fetch_9am_data.py found")
else:
    print("✗ fetch_9am_data.py MISSING!")
    all_checks_passed = False

if os.path.exists('oanda_multi_pair_trader.py'):
    print("✓ oanda_multi_pair_trader.py found")
else:
    print("✗ oanda_multi_pair_trader.py MISSING!")
    all_checks_passed = False

if os.path.exists('run_multi_pair_trader.sh'):
    print("✓ run_multi_pair_trader.sh found")
else:
    print("✗ run_multi_pair_trader.sh MISSING!")
    all_checks_passed = False

if os.path.exists('oanda_data_fetcher.py'):
    print("✓ oanda_data_fetcher.py found")
else:
    print("✗ oanda_data_fetcher.py MISSING!")
    all_checks_passed = False

print()

# Check 5: Verify .env file
print("Check 5: Environment configuration")
print("-" * 80)
if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')

    if api_key:
        print(f"✓ OANDA_API_KEY found ({api_key[:10]}...)")
    else:
        print("✗ OANDA_API_KEY missing!")
        all_checks_passed = False

    if account_id:
        print(f"✓ OANDA_ACCOUNT_ID found ({account_id})")
    else:
        print("✗ OANDA_ACCOUNT_ID missing!")
        all_checks_passed = False
else:
    print("✗ .env file missing!")
    all_checks_passed = False

print()

# Check 6: Calculate timing
print("Check 6: Timing information")
print("-" * 80)
now = datetime.now()
print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Migration will change schedule from:")
print(f"  OLD: 5:35 PM EST (2:35 PM PST) - Bad spreads")
print(f"  NEW: 9:00 AM EST (6:00 AM PST) - Good spreads")
print()
if has_positions:
    print("⚠ You have OPEN POSITIONS")
    print("  → They will remain open until 9 AM tomorrow")
    print("  → Bot will evaluate them with 9 AM predictions")
    print("  → This is SAFE - no positions will be lost")
else:
    print("○ No open positions - clean migration")

print()
print("=" * 80)
if all_checks_passed:
    print("✓ ALL CHECKS PASSED - Ready for migration!")
    print()
    print("Next steps:")
    print("1. Review this output carefully")
    print("2. Run: bash migrate_to_9am_trading.sh")
    print("3. Confirm cron is updated to 9 AM EST")
    print("4. First trade will execute at 9 AM EST tomorrow")
else:
    print("✗ SOME CHECKS FAILED - Fix issues before migrating")
    print()
    print("Fix the issues above, then run this script again")
print("=" * 80)
