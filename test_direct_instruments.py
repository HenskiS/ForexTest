"""
Test direct access to specific instruments to see what's actually available.
"""
import sys
import os
import io
import requests
from dotenv import load_dotenv

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

api_key = os.getenv('OANDA_API_KEY')
base_url = "https://api-fxtrade.oanda.com/v3"

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Instruments we know work
known_working = ['XAU_USD', 'XAG_USD', 'XPT_USD', 'XPD_USD', 'XCU_USD', 'BCO_USD']

# Instruments we want to test
test_instruments = [
    # Agricultural
    'SUGAR_USD', 'SGUSD', 'SUGAR', 'SB_USD',
    'COFFEE_USD', 'COFUSD', 'KC_USD',
    'CORN_USD', 'CORNUSD', 'ZC_USD',
    'WHEAT_USD', 'WHEATUSD', 'ZW_USD',
    'SOYBEAN_USD', 'SOYUSD', 'ZS_USD',

    # Indices
    'SPX500_USD', 'SP500_USD', 'US500_USD', 'SPX_USD',
    'NAS100_USD', 'NDAQ_USD', 'NDX_USD',
    'US30_USD', 'DJ30_USD', 'DJI_USD',
    'DE30_EUR', 'DAX_EUR', 'GER30_EUR',
    'UK100_GBP', 'FTSE_GBP', 'UKX_GBP',
    'JP225_USD', 'NKY_USD', 'NIKKEI_USD',
    'FR40_EUR', 'CAC_EUR', 'FRA40_EUR',

    # More energy
    'WTICO_USD', 'WTI_USD', 'CL_USD',
    'NATGAS_USD', 'NG_USD', 'NGAS_USD',

    # Other metals we haven't tried
    'XTI_USD',  # Titanium?
    'XNI_USD',  # Nickel?
]

print("="*80)
print("TESTING DIRECT INSTRUMENT ACCESS")
print("="*80)
print()

print("First, confirming known working instruments:")
print("-"*80)
for instrument in known_working:
    url = f"{base_url}/instruments/{instrument}/candles?count=1&granularity=D"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print(f"✓ {instrument:20} - WORKS")
        else:
            print(f"✗ {instrument:20} - FAILED ({response.status_code})")
    except Exception as e:
        print(f"✗ {instrument:20} - ERROR: {e}")

print()
print("Now testing potential new instruments:")
print("-"*80)

working_instruments = []
for instrument in test_instruments:
    url = f"{base_url}/instruments/{instrument}/candles?count=1&granularity=D"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print(f"✓ {instrument:20} - WORKS!")
            working_instruments.append(instrument)
        else:
            pass  # Don't print failures to reduce noise
    except Exception as e:
        pass

if working_instruments:
    print()
    print("="*80)
    print("NEWLY DISCOVERED WORKING INSTRUMENTS:")
    print("="*80)
    for inst in working_instruments:
        print(f"  {inst}")
    print()
    print(f"Total new instruments found: {len(working_instruments)}")
else:
    print()
    print("No new instruments found from test list.")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Known working: {len(known_working)}")
print(f"New discovered: {len(working_instruments)}")
print(f"Total available: {len(known_working) + len(working_instruments)}")
print("="*80)
