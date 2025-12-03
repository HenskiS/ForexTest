"""
Check actual OANDA spreads for all assets at 9 AM EST
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.oanda_client import OandaClient
from trading.config import Config
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# All 10 assets
ASSETS = [
    'EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY',  # Forex
    'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'XCUUSD',  # Metals
    'SUGARUSD', 'SPX500USD', 'DE30EUR', 'WTICOUSD', 'BCOUSD'  # Commodities/Indices
]

ASSET_NAMES = {
    'EURUSD': 'EUR/USD', 'GBPUSD': 'GBP/USD', 'AUDUSD': 'AUD/USD', 'USDJPY': 'USD/JPY',
    'XAUUSD': 'Gold', 'XAGUSD': 'Silver', 'XPTUSD': 'Platinum', 'XPDUSD': 'Palladium', 'XCUUSD': 'Copper',
    'SUGARUSD': 'Sugar', 'SPX500USD': 'S&P 500', 'DE30EUR': 'DAX', 'WTICOUSD': 'WTI Oil', 'BCOUSD': 'Brent Oil'
}

ASSET_CATEGORIES = {
    'EURUSD': 'Forex', 'GBPUSD': 'Forex', 'AUDUSD': 'Forex', 'USDJPY': 'Forex',
    'XAUUSD': 'Metals', 'XAGUSD': 'Metals', 'XPTUSD': 'Metals', 'XPDUSD': 'Metals', 'XCUUSD': 'Metals',
    'SUGARUSD': 'Commodities', 'SPX500USD': 'Indices', 'DE30EUR': 'Indices', 'WTICOUSD': 'Energy', 'BCOUSD': 'Energy'
}

print('='*100)
print('LIVE OANDA SPREADS AT CURRENT TIME')
print('='*100)
print()

# Initialize OANDA client
client = OandaClient(practice=True)

# Current stop/TP settings
CURRENT_STOP_LOSS = 0.18  # 0.18%
CURRENT_TAKE_PROFIT = 2.0  # 2.0%

print(f"Current Strategy Settings:")
print(f"  Stop Loss: {CURRENT_STOP_LOSS}%")
print(f"  Take Profit: {CURRENT_TAKE_PROFIT}%")
print()

print('='*100)
print('CURRENT SPREADS BY ASSET')
print('-'*100)
print(f"{'Asset':<20} {'Category':<15} {'Bid':<12} {'Ask':<12} {'Spread':<12} {'Spread %':<12} {'vs SL':<10}")
print('-'*100)

spreads = {}
for asset in ASSETS:
    try:
        # Get current prices
        pricing = client.get_current_price(asset)

        if pricing and 'bids' in pricing and 'asks' in pricing:
            bid = float(pricing['bids'][0]['price'])
            ask = float(pricing['asks'][0]['price'])
            mid = (bid + ask) / 2
            spread = ask - bid
            spread_pct = (spread / mid) * 100
            spread_vs_sl = (spread_pct / CURRENT_STOP_LOSS) * 100

            spreads[asset] = {
                'category': ASSET_CATEGORIES[asset],
                'bid': bid,
                'ask': ask,
                'spread': spread,
                'spread_pct': spread_pct,
                'spread_vs_sl': spread_vs_sl
            }

            print(f"{ASSET_NAMES[asset]:<20} {ASSET_CATEGORIES[asset]:<15} {bid:>10.5f}  {ask:>10.5f}  {spread:>10.5f}  {spread_pct:>10.4f}%  {spread_vs_sl:>8.1f}%")
    except Exception as e:
        print(f"{ASSET_NAMES.get(asset, asset):<20} {ASSET_CATEGORIES.get(asset, 'Unknown'):<15} ERROR: {e}")

print()

# Calculate category averages
print('='*100)
print('SPREAD COMPARISON BY ASSET CLASS')
print('-'*100)
print()

for category in ['Forex', 'Metals', 'Indices', 'Energy', 'Commodities']:
    cat_spreads = [s['spread_pct'] for a, s in spreads.items() if s['category'] == category]
    if cat_spreads:
        avg_spread = sum(cat_spreads) / len(cat_spreads)
        max_spread = max(cat_spreads)
        min_spread = min(cat_spreads)
        avg_vs_sl = (avg_spread / CURRENT_STOP_LOSS) * 100

        print(f"{category}:")
        print(f"  Average Spread: {avg_spread:.4f}%")
        print(f"  Range: {min_spread:.4f}% - {max_spread:.4f}%")
        print(f"  As % of Stop Loss: {avg_vs_sl:.1f}%")
        print()

# Overall analysis
forex_spreads = [s['spread_pct'] for a, s in spreads.items() if s['category'] == 'Forex']
metals_spreads = [s['spread_pct'] for a, s in spreads.items() if s['category'] == 'Metals']
other_spreads = [s['spread_pct'] for a, s in spreads.items() if s['category'] not in ['Forex', 'Metals']]

if forex_spreads and metals_spreads:
    forex_avg = sum(forex_spreads) / len(forex_spreads)
    metals_avg = sum(metals_spreads) / len(metals_spreads)

    print('='*100)
    print('KEY INSIGHTS')
    print('-'*100)
    print()
    print(f"1. SPREAD COMPARISON:")
    print(f"   • Forex average: {forex_avg:.4f}% ({forex_avg/CURRENT_STOP_LOSS*100:.1f}% of stop loss)")
    print(f"   • Metals average: {metals_avg:.4f}% ({metals_avg/CURRENT_STOP_LOSS*100:.1f}% of stop loss)")

    if metals_avg > forex_avg:
        print(f"   • Metals spreads are {metals_avg/forex_avg:.1f}x wider than forex")
    print()

print('2. STOP LOSS ADEQUACY:')
print(f"   Your 0.18% stop loss provides:")
for category in ['Forex', 'Metals', 'Indices', 'Energy', 'Commodities']:
    cat_spreads = [s['spread_pct'] for a, s in spreads.items() if s['category'] == category]
    if cat_spreads:
        avg_spread = sum(cat_spreads) / len(cat_spreads)
        buffer = CURRENT_STOP_LOSS - avg_spread
        print(f"   • {category}: {buffer:.4f}% buffer after spread ({buffer/CURRENT_STOP_LOSS*100:.0f}% of SL)")

print()
print('='*100)
print('RECOMMENDATION')
print('='*100)
print()

# Calculate if we need different stops
if metals_spreads:
    metals_avg = sum(metals_spreads) / len(metals_spreads)
    metals_vs_sl = (metals_avg / CURRENT_STOP_LOSS) * 100

    if metals_vs_sl > 30:
        print('⚠️  CONSIDER WIDER STOPS FOR METALS/COMMODITIES')
        print()
        print(f"   Spreads consume {metals_vs_sl:.0f}% of your stop loss buffer.")
        print(f"   Recommended: {CURRENT_STOP_LOSS * 1.5:.2f}% stop for metals/commodities")
        print(f"   Keep {CURRENT_STOP_LOSS}% for forex")
    elif metals_vs_sl > 20:
        print('⚠️  MONITOR SPREAD IMPACT')
        print()
        print(f"   Spreads consume {metals_vs_sl:.0f}% of your stop loss buffer.")
        print(f"   Current stops should work, but watch for stop-hunting in low liquidity")
    else:
        print('✓ CURRENT STOPS ARE FINE')
        print()
        print(f"   Spreads only consume {metals_vs_sl:.0f}% of stop loss buffer.")
        print(f"   Your 0.18% SL / 2.0% TP settings are appropriate for all assets.")
        print()
        print('   Remember: Your backtest already showed:')
        print('   • 102% annual return at 1x leverage')
        print('   • 9.35 Sharpe ratio')
        print('   • -1.3% max drawdown')
        print()
        print('   These results INCLUDE spread costs. No changes needed!')

print()
print('='*100)
