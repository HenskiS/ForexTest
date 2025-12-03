"""
Initialize prediction buffers for all 14 assets in the diversified portfolio.

This script runs the buffer initialization for each asset sequentially.
"""
import subprocess
import sys

# All 14 assets
ASSETS = [
    # 4 Forex pairs
    'EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY',
    # 5 Metals
    'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'XCUUSD',
    # 5 Commodities/Indices
    'SUGARUSD', 'SPX500USD', 'DE30EUR', 'WTICOUSD', 'BCOUSD'
]

def main():
    print("="*80)
    print("INITIALIZING PREDICTION BUFFERS FOR ALL 14 ASSETS")
    print("="*80)
    print(f"\nAssets to initialize: {len(ASSETS)}")
    print("This will take approximately 30-45 minutes...")
    print()

    failed = []
    succeeded = []

    for i, asset in enumerate(ASSETS, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(ASSETS)}] Initializing {asset}")
        print(f"{'='*80}")

        try:
            # Run initialization for this asset
            result = subprocess.run(
                [sys.executable, 'initialize_oanda_buffer.py', '--pair', asset, '--live'],
                check=True
            )

            if result.returncode == 0:
                succeeded.append(asset)
                print(f"✓ {asset} initialized successfully")
            else:
                failed.append(asset)
                print(f"✗ {asset} failed")

        except subprocess.CalledProcessError as e:
            failed.append(asset)
            print(f"✗ {asset} failed with error: {e}")
        except KeyboardInterrupt:
            print("\n\nInitialization interrupted by user")
            print(f"\nSucceeded: {len(succeeded)}/{len(ASSETS)}")
            print(f"Failed: {len(failed)}/{len(ASSETS)}")
            sys.exit(1)

    # Summary
    print("\n" + "="*80)
    print("INITIALIZATION COMPLETE")
    print("="*80)
    print(f"\nSucceeded: {len(succeeded)}/{len(ASSETS)}")
    for asset in succeeded:
        print(f"  ✓ {asset}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(ASSETS)}")
        for asset in failed:
            print(f"  ✗ {asset}")
        sys.exit(1)
    else:
        print("\n✓ All assets initialized successfully!")
        print("\nYou can now run the 14-asset trader:")
        print("  venv/Scripts/python.exe oanda_14_asset_trader.py --live --dry-run --yes")

if __name__ == '__main__':
    main()
