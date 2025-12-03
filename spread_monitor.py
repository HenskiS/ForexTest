"""
Real-time spread monitoring for OANDA trading
Checks if current spreads are acceptable before trading
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.oanda_client import OandaClient
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Maximum acceptable spreads (in percentage)
# Based on Expected Value (EV) analysis with 50% EV preservation
# Low-spread assets (forex, indices): 2-3x typical spread
# High-spread assets (platinum, palladium, sugar): 50% EV preservation
MAX_ACCEPTABLE_SPREADS = {
    # Forex - normally 0.007-0.015%, max 2-3x (VERY SAFE)
    'EURUSD': 0.020,  # 2.7x typical - preserves 79% of EV
    'GBPUSD': 0.020,  # 2.2x typical - preserves 70% of EV
    'AUDUSD': 0.030,  # 1.9x typical - preserves 68% of EV
    'USDJPY': 0.015,  # 2.2x typical - preserves 87% of EV

    # Low-spread metals - normally 0.012-0.04%, max 2x (VERY SAFE)
    'XAUUSD': 0.025,  # 2.0x typical - preserves 92% of EV
    'XAGUSD': 0.025,  # 2.1x typical - preserves 93% of EV
    'XCUUSD': 0.075,  # 2.0x typical - preserves 79% of EV

    # High-spread metals - use 50% EV preservation (ADJUSTED - was too loose)
    'XPTUSD': 0.216,  # 0.9x typical - preserves 50% of EV (was 0.400)
    'XPDUSD': 0.303,  # 1.0x typical - preserves 50% of EV (was 0.450)

    # Commodities/Indices
    'SUGARUSD': 0.240,  # 1.2x typical - preserves 50% of EV (was 0.300)
    'SPX500USD': 0.025,  # 2.3x typical - preserves 89% of EV
    'DE30EUR': 0.025,    # 2.0x typical - preserves 90% of EV
    'WTICOUSD': 0.080,   # 2.0x typical - preserves 81% of EV
    'BCOUSD': 0.100,     # 1.6x typical - preserves 73% of EV
}

def get_current_spreads(client, assets):
    """
    Get current bid/ask spreads for all assets
    Returns dict with spread info
    """
    spreads = {}

    for asset in assets:
        try:
            # Use the fetcher's get_current_price method
            pricing = client.fetcher.get_current_price(asset)

            if pricing and 'bid' in pricing and 'ask' in pricing:
                bid = pricing['bid']
                ask = pricing['ask']
                mid = pricing['mid']
                spread = pricing['spread']
                spread_pct = (spread / mid) * 100

                max_acceptable = MAX_ACCEPTABLE_SPREADS.get(asset, 0.100)
                is_acceptable = spread_pct <= max_acceptable

                spreads[asset] = {
                    'bid': bid,
                    'ask': ask,
                    'spread': spread,
                    'spread_pct': spread_pct,
                    'max_acceptable': max_acceptable,
                    'is_acceptable': is_acceptable,
                    'ratio': spread_pct / max_acceptable
                }
            else:
                spreads[asset] = {
                    'error': 'No pricing data available'
                }

        except Exception as e:
            spreads[asset] = {
                'error': str(e)
            }

    return spreads

def check_spreads_acceptable(assets, practice=True):
    """
    Check if current spreads are acceptable for trading
    Returns (all_acceptable, spreads_dict, unacceptable_list)
    """
    # Create client with first asset (doesn't matter which for pricing calls)
    client = OandaClient(assets[0], practice=practice)
    spreads = get_current_spreads(client, assets)

    unacceptable = []

    for asset, data in spreads.items():
        if 'error' in data:
            unacceptable.append(f"{asset}: {data['error']}")
        elif not data['is_acceptable']:
            unacceptable.append(
                f"{asset}: {data['spread_pct']:.3f}% (max {data['max_acceptable']:.3f}%, {data['ratio']:.1f}x normal)"
            )

    all_acceptable = len(unacceptable) == 0

    return all_acceptable, spreads, unacceptable

def print_spread_report(spreads):
    """Print formatted spread report"""
    print('='*100)
    print('CURRENT OANDA SPREADS')
    print('='*100)
    print()
    print(f"{'Asset':<12} {'Bid':>12} {'Ask':>12} {'Spread':>10} {'Spread %':>10} {'Max %':>10} {'Status':<15}")
    print('-'*100)

    for asset, data in spreads.items():
        if 'error' in data:
            print(f"{asset:<12} ERROR: {data['error']}")
        else:
            status = '✓ OK' if data['is_acceptable'] else f"✗ TOO WIDE ({data['ratio']:.1f}x)"
            print(f"{asset:<12} {data['bid']:>12.5f} {data['ask']:>12.5f} "
                  f"{data['spread']:>10.5f} {data['spread_pct']:>9.3f}% {data['max_acceptable']:>9.3f}% {status:<15}")

    print()

if __name__ == '__main__':
    # Test with all 10 assets
    ASSETS = [
        'EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY',
        'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'XCUUSD',
        'SUGARUSD', 'SPX500USD', 'DE30EUR', 'WTICOUSD', 'BCOUSD'
    ]

    print("Checking current spreads...")
    print()

    # Use live mode (no practice API key available)
    all_acceptable, spreads, unacceptable = check_spreads_acceptable(ASSETS, practice=False)

    print_spread_report(spreads)

    if all_acceptable:
        print('='*100)
        print('✓ ALL SPREADS ACCEPTABLE - SAFE TO TRADE')
        print('='*100)
    else:
        print('='*100)
        print('✗ SOME SPREADS TOO WIDE - DO NOT TRADE THESE ASSETS')
        print('='*100)
        print()
        print('Unacceptable spreads:')
        for msg in unacceptable:
            print(f'  • {msg}')
        print()
        print('Recommendation: Wait for spreads to normalize or skip these assets today')
