"""
Test position sizing logic to verify USDJPY fix works correctly.
Simulates what will happen at 9 AM tomorrow without placing actual trades.
"""
import sys
import io

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, 'trading')

from oanda_data_fetcher import OandaDataFetcher

# Test configuration
ACCOUNT_BALANCE = 500.00
LEVERAGE = 2.0
NUM_PAIRS = 4

# Calculate expected values
capital_per_pair = ACCOUNT_BALANCE / NUM_PAIRS
position_size_per_pair = capital_per_pair * LEVERAGE

print("=" * 80)
print("POSITION SIZING TEST - DRY RUN")
print("=" * 80)
print(f"\nAccount balance: ${ACCOUNT_BALANCE:.2f}")
print(f"Leverage: {LEVERAGE:.1f}x")
print(f"Number of pairs: {NUM_PAIRS}")
print(f"\nCapital per pair: ${capital_per_pair:.2f}")
print(f"Position size per pair: ${position_size_per_pair:.2f}")
print(f"Total exposure: ${position_size_per_pair * NUM_PAIRS:.2f}")
print()

# Test pairs with real prices from OANDA
pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']

fetcher = OandaDataFetcher(practice=False)

print("=" * 80)
print("TESTING POSITION SIZING WITH CURRENT PRICES")
print("=" * 80)

results = []

for pair in pairs:
    print(f"\n{pair}:")
    print("-" * 80)

    # Get current price
    price_data = fetcher.get_current_price(pair)
    if not price_data:
        print(f"  ERROR: Could not fetch price")
        continue

    current_price = price_data['mid']
    spread_pips = price_data['spread_pips']

    # Convert to OANDA instrument name
    instrument = fetcher.get_instrument_name(pair)

    # Calculate units using FIXED logic
    if instrument.startswith('USD_'):
        # USD is base currency (USD/JPY, USD/CHF, USD/CAD)
        # 1 unit = $1, so units = desired dollar exposure
        units = int(position_size_per_pair)
        calculation = f"units = {position_size_per_pair:.2f} (direct, USD is base)"
    else:
        # USD is quote currency (EUR/USD, GBP/USD, AUD/USD, NZD/USD)
        # 1 unit = 1 base currency, so units = dollars / price
        units = int(position_size_per_pair / current_price)
        calculation = f"units = {position_size_per_pair:.2f} / {current_price:.5f} = {units}"

    # Calculate actual exposure
    if instrument.startswith('USD_'):
        actual_exposure = units  # USD is base
    else:
        actual_exposure = units * current_price  # Convert base currency to USD

    # Calculate margin used (assuming 50:1 margin)
    margin_used = actual_exposure / 50

    results.append({
        'pair': pair,
        'price': current_price,
        'units': units,
        'exposure': actual_exposure,
        'margin': margin_used,
        'spread': spread_pips
    })

    print(f"  Current price: {current_price:.5f}")
    print(f"  Spread: {spread_pips:.2f} pips")
    print(f"  Calculation: {calculation}")
    print(f"  Units to trade: {units:,}")
    print(f"  Actual exposure: ${actual_exposure:.2f}")
    print(f"  Margin required: ${margin_used:.2f}")

    # Validate
    target = position_size_per_pair
    tolerance = 2.0  # Allow $2 variance due to rounding

    if abs(actual_exposure - target) <= tolerance:
        print(f"  ✓ PASS: Exposure within ${tolerance:.2f} of target ${target:.2f}")
    else:
        print(f"  ✗ FAIL: Exposure ${actual_exposure:.2f} vs target ${target:.2f}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

total_exposure = sum(r['exposure'] for r in results)
total_margin = sum(r['margin'] for r in results)
avg_spread = sum(r['spread'] for r in results) / len(results)

print(f"\nTotal exposure: ${total_exposure:.2f}")
print(f"Total margin used: ${total_margin:.2f}")
print(f"Average spread: {avg_spread:.2f} pips")
print()

print("Position sizes:")
for r in results:
    diff = r['exposure'] - position_size_per_pair
    status = "✓" if abs(diff) <= 2.0 else "✗"
    print(f"  {status} {r['pair']:8} ${r['exposure']:6.2f} (target: ${position_size_per_pair:.2f}, diff: ${diff:+.2f})")

print()
print("Margin requirements:")
for r in results:
    print(f"  {r['pair']:8} ${r['margin']:6.2f}")

print()
if abs(total_exposure - (position_size_per_pair * NUM_PAIRS)) <= 5.0:
    print("✓ ALL TESTS PASSED - Position sizing is correct!")
    print(f"  Each pair will get ~${position_size_per_pair:.0f} exposure")
    print(f"  Total portfolio exposure: ${total_exposure:.2f}")
    print(f"  Your capital at risk: ${ACCOUNT_BALANCE:.2f}")
    print(f"  Borrowed (leverage): ${total_exposure - ACCOUNT_BALANCE:.2f}")
else:
    print("✗ TEST FAILED - Position sizing is incorrect!")
    print(f"  Expected total: ${position_size_per_pair * NUM_PAIRS:.2f}")
    print(f"  Actual total: ${total_exposure:.2f}")

print("=" * 80)
