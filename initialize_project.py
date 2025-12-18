"""
Interactive Project Initialization Script

This script sets up the ForexTest trading project on a new device by:
1. Fetching historical OHLC data from OANDA
2. Fetching spread data for backtesting
3. Initializing prediction buffers for production trading
4. (Optional) Generating full backtest predictions

Safe to run multiple times (idempotent) - will check for existing data
and allow you to skip steps that are already complete.
"""
import os
import sys
import subprocess


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*100)
    print(text)
    print("="*100 + "\n")


def print_step(step_num, total_steps, description):
    """Print a step header"""
    print(f"\n{'='*100}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'='*100}\n")


def check_env_file():
    """Check if .env file exists with required variables"""
    if not os.path.exists('.env'):
        return False, "No .env file found"

    with open('.env', 'r') as f:
        content = f.read()

    has_api_key = 'OANDA_API_KEY=' in content
    has_account_id = 'OANDA_ACCOUNT_ID=' in content

    if not has_api_key or not has_account_id:
        return False, "Missing OANDA_API_KEY or OANDA_ACCOUNT_ID"

    return True, "Environment configured"


def check_data_files():
    """Check which data files exist and if they're up-to-date"""
    from trading import TradingConfig
    import pandas as pd
    from datetime import datetime, timedelta

    pairs = TradingConfig.DEFAULT_PAIRS
    ohlc_files = []
    spread_files = []
    buffer_files = []

    ohlc_outdated = []
    spread_outdated = []

    # Consider data outdated if it's more than 7 days old
    # (accounting for weekends when markets are closed)
    max_age_days = 7

    for pair in pairs:
        ohlc_path = f'data/{pair}_1day_oanda.csv'
        spread_path = f'data/{pair}_1day_with_spreads.csv'
        buffer_path = f'data/oanda_cache/prediction_buffer_ann_{pair}.pkl'

        # Check OHLC data
        if os.path.exists(ohlc_path):
            ohlc_files.append(pair)
            try:
                df = pd.read_csv(ohlc_path)
                if len(df) > 0:
                    df['date'] = pd.to_datetime(df['date'])
                    last_date = df['date'].max()
                    days_old = (pd.Timestamp.now() - last_date).days
                    if days_old > max_age_days:
                        ohlc_outdated.append((pair, days_old))
            except:
                pass

        # Check spread data
        if os.path.exists(spread_path):
            spread_files.append(pair)
            try:
                df = pd.read_csv(spread_path)
                if len(df) > 0:
                    df['date'] = pd.to_datetime(df['date'])
                    last_date = df['date'].max()
                    days_old = (pd.Timestamp.now() - last_date).days
                    if days_old > max_age_days:
                        spread_outdated.append((pair, days_old))
            except:
                pass

        # Check buffer
        if os.path.exists(buffer_path):
            buffer_files.append(pair)

    return {
        'ohlc': ohlc_files,
        'spreads': spread_files,
        'buffers': buffer_files,
        'total_pairs': len(pairs),
        'ohlc_outdated': ohlc_outdated,
        'spread_outdated': spread_outdated
    }


def run_script(script_name, description, args=None):
    """Run a Python script and return success status"""
    print(f"Running: {script_name}")
    if args:
        print(f"Arguments: {' '.join(args)}")
    print()

    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[FAIL] {description} failed: {e}")
        return False


def ask_yes_no(question, default=True):
    """Ask a yes/no question"""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        response = input(question + suffix).strip().lower()
        if response == '':
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Please answer 'y' or 'n'")


def main():
    """Main initialization flow"""
    print_header("ForexTest Project Initialization")

    print("This script will guide you through setting up the ForexTest trading project")
    print("on this device. You can safely run this multiple times - it will check for")
    print("existing data and allow you to skip completed steps.")
    print()

    # Check environment
    print("Checking environment setup...")
    env_ok, env_msg = check_env_file()
    print(f"  {'OK' if env_ok else 'FAIL'} {env_msg}")

    if not env_ok:
        print("\n[!] You need to set up your .env file first!")
        print("\nCreate a .env file in this directory with:")
        print("  OANDA_API_KEY=your_api_key_here")
        print("  OANDA_ACCOUNT_ID=your_account_id_here")
        print("\nGet these from your OANDA account settings.")
        return

    # Check existing data
    print("\nChecking existing data files...")
    existing = check_data_files()
    total = existing['total_pairs']

    print(f"  OHLC data: {len(existing['ohlc'])}/{total} pairs")
    if existing['ohlc_outdated']:
        print(f"    [!] {len(existing['ohlc_outdated'])} pairs outdated (>7 days old):")
        for pair, days in existing['ohlc_outdated']:
            print(f"        - {pair}: {days} days old")

    print(f"  Spread data: {len(existing['spreads'])}/{total} pairs")
    if existing['spread_outdated']:
        print(f"    [!] {len(existing['spread_outdated'])} pairs outdated (>7 days old):")
        for pair, days in existing['spread_outdated']:
            print(f"        - {pair}: {days} days old")

    print(f"  Prediction buffers: {len(existing['buffers'])}/{total} pairs")

    print("\n" + "-"*100)
    print("\nInitialization has 3 required steps + 1 optional step:")
    print("  1. Fetch historical OHLC data (required for everything)")
    print("  2. Fetch spread data (required for accurate backtesting)")
    print("  3. Initialize prediction buffers (required for production trading)")
    print("  4. Generate backtest predictions (optional, only for backtesting)")
    print()

    if not ask_yes_no("Ready to begin?", default=True):
        print("\nInitialization cancelled.")
        return

    # Step 1: Fetch OHLC data
    print_step(1, 4, "Fetch Historical OHLC Data")

    # Check if we need to update any data
    needs_update = len(existing['ohlc']) < total or len(existing['ohlc_outdated']) > 0

    if len(existing['ohlc']) == total and not existing['ohlc_outdated']:
        print(f"[OK] All {total} pairs have up-to-date OHLC data")
        if not ask_yes_no("Re-fetch all data anyway?", default=False):
            print("Skipping OHLC data fetch")
        else:
            if ask_yes_no("Proceed with fetching OHLC data for all pairs?", default=True):
                from trading import TradingConfig
                for i, pair in enumerate(TradingConfig.DEFAULT_PAIRS, 1):
                    print(f"\n[{i}/{total}] Fetching {pair}...")
                    success = run_script('oanda_data_fetcher.py', f'{pair} data fetch',
                                       ['--pair', pair, '--count', '5000', '--live', '--daily-alignment', '9'])
                    if not success and not ask_yes_no(f"Continue anyway?", default=True):
                        print("Stopping OHLC data fetch")
                        break
            else:
                print("Skipping OHLC data fetch")

    elif needs_update:
        # Determine which pairs need updating
        from trading import TradingConfig
        missing = [p for p in TradingConfig.DEFAULT_PAIRS if p not in existing['ohlc']]
        outdated_pairs = [pair for pair, days in existing['ohlc_outdated']]
        to_update = list(set(missing + outdated_pairs))

        if missing:
            print(f"Missing OHLC data for: {', '.join(missing)}")
        if existing['ohlc_outdated']:
            print(f"Outdated OHLC data for: {', '.join(outdated_pairs)}")

        if ask_yes_no(f"Update {len(to_update)} pair(s)?", default=True):
            for i, pair in enumerate(to_update, 1):
                print(f"\n[{i}/{len(to_update)}] Fetching {pair}...")
                success = run_script('oanda_data_fetcher.py', f'{pair} data fetch',
                                   ['--pair', pair, '--count', '5000', '--live', '--daily-alignment', '9'])
                if not success and not ask_yes_no(f"Continue anyway?", default=True):
                    print("Stopping OHLC data fetch")
                    break

    # Step 2: Fetch spread data
    print_step(2, 4, "Fetch Spread Data for Backtesting")

    needs_spread_update = len(existing['spreads']) < total or len(existing['spread_outdated']) > 0

    if len(existing['spreads']) == total and not existing['spread_outdated']:
        print(f"[OK] All {total} pairs have up-to-date spread data")
        if not ask_yes_no("Re-fetch spread data anyway?", default=False):
            print("Skipping spread data fetch")
        else:
            if ask_yes_no("Proceed with fetching spread data?", default=True):
                run_script('fetch_spread_data.py', 'Spread data fetch', ['--live'])
    elif needs_spread_update:
        from trading import TradingConfig
        missing = [p for p in TradingConfig.DEFAULT_PAIRS if p not in existing['spreads']]
        outdated_pairs = [pair for pair, days in existing['spread_outdated']]

        if missing:
            print(f"Missing spread data for: {', '.join(missing)}")
        if existing['spread_outdated']:
            print(f"Outdated spread data for: {', '.join(outdated_pairs)}")

        if ask_yes_no(f"Fetch/update spread data for all pairs?", default=True):
            run_script('fetch_spread_data.py', 'Spread data fetch', ['--live'])

    # Step 3: Initialize prediction buffers
    print_step(3, 4, "Initialize Prediction Buffers for Production")

    if len(existing['buffers']) == total:
        print(f"[OK] All {total} pairs already have prediction buffers")
        if not ask_yes_no("Re-initialize buffers anyway?", default=False):
            print("Skipping buffer initialization")
        else:
            print("\n[!] WARNING: This step takes a long time (~2-3 hours on most machines)")
            print("It generates 200 predictions per pair using rolling window training.")
            if ask_yes_no("Proceed with buffer initialization?", default=True):
                # First, clean up old buffers with wrong naming
                print("\nCleaning up old buffer files...")
                old_pattern = 'data/oanda_cache/prediction_buffer_*.pkl'
                import glob
                old_buffers = [f for f in glob.glob(old_pattern) if 'prediction_buffer_ann_' not in f and 'prediction_buffer_xgboost_' not in f]
                if old_buffers:
                    print(f"Found {len(old_buffers)} old buffer files to remove:")
                    for f in old_buffers:
                        print(f"  - {f}")
                    if ask_yes_no("Remove these old files?", default=True):
                        for f in old_buffers:
                            os.remove(f)
                            print(f"  Removed {f}")

                run_script('initialize_oanda_buffer_ann.py', 'Prediction buffer initialization', ['--live'])
    else:
        print(f"Found {len(existing['buffers'])}/{total} pairs with prediction buffers")
        print("\n[!] WARNING: This step takes a long time (~2-3 hours on most machines)")
        print("It generates 200 predictions per pair using rolling window training.")
        if ask_yes_no("Initialize prediction buffers?", default=True):
            # Clean up old buffers first
            print("\nCleaning up old buffer files...")
            old_pattern = 'data/oanda_cache/prediction_buffer_*.pkl'
            import glob
            old_buffers = [f for f in glob.glob(old_pattern) if 'prediction_buffer_ann_' not in f and 'prediction_buffer_xgboost_' not in f]
            if old_buffers:
                print(f"Found {len(old_buffers)} old buffer files to remove:")
                for f in old_buffers:
                    print(f"  - {f}")
                if ask_yes_no("Remove these old files?", default=True):
                    for f in old_buffers:
                        os.remove(f)
                        print(f"  Removed {f}")

            run_script('initialize_oanda_buffer_ann.py', 'Prediction buffer initialization', ['--live'])

    # Step 4: Generate backtest predictions (optional)
    print_step(4, 4, "Generate Backtest Predictions (Optional)")

    predictions_exist = os.path.exists('optimized_ann_predictions') and \
                       len(os.listdir('optimized_ann_predictions')) == total

    if predictions_exist:
        print(f"[OK] Backtest predictions already exist for all {total} pairs")
        if not ask_yes_no("Re-generate backtest predictions?", default=False):
            print("Skipping backtest prediction generation")
        else:
            print("\n[!] WARNING: This step takes VERY long (~6-10 hours on most machines)")
            print("It generates predictions for the entire historical dataset.")
            print("Only needed if you want to run backtests (test_dca_stop_fixed.py).")
            if ask_yes_no("Proceed with backtest prediction generation?", default=False):
                run_script('train_all_pairs_optimized_hyperparams.py', 'Backtest prediction generation')
    else:
        print("Backtest predictions not found.")
        print("\nThis step is OPTIONAL - only needed for running backtests.")
        print("[!] WARNING: This takes VERY long (~6-10 hours on most machines)")
        if ask_yes_no("Generate backtest predictions?", default=False):
            run_script('train_all_pairs_optimized_hyperparams.py', 'Backtest prediction generation')

    # Final summary
    print_header("Initialization Complete!")

    # Re-check what we have now
    final_check = check_data_files()

    print("Current status:")
    print(f"  [OK] OHLC data: {len(final_check['ohlc'])}/{total} pairs")
    print(f"  [OK] Spread data: {len(final_check['spreads'])}/{total} pairs")
    print(f"  [OK] Prediction buffers: {len(final_check['buffers'])}/{total} pairs")

    predictions_exist = os.path.exists('optimized_ann_predictions') and \
                       len(os.listdir('optimized_ann_predictions')) > 0
    print(f"  [{'OK' if predictions_exist else '--'}] Backtest predictions: {'Yes' if predictions_exist else 'No'}")

    print("\nNext steps:")
    if len(final_check['buffers']) == total:
        print("  1. Test the production trader:")
        print("     python oanda_multi_pair_trader.py --dry-run")
        print("  2. Run on practice account:")
        print("     python oanda_multi_pair_trader.py")
        print("  3. When ready, run on live account:")
        print("     python oanda_multi_pair_trader.py --live")
    else:
        print("  [!] Complete prediction buffer initialization to run production trading")

    if predictions_exist:
        print("  4. Run backtests:")
        print("     python test_dca_stop_fixed.py")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInitialization interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
