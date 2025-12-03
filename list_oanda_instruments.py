"""
Query OANDA API to list all available instruments.
This will help us find the correct names for Sugar, Coffee, indices, etc.
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
account_id = os.getenv('OANDA_ACCOUNT_ID')

if not api_key or not account_id:
    print("Error: Missing OANDA credentials")
    sys.exit(1)

# Use live endpoint
base_url = "https://api-fxtrade.oanda.com/v3"

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

print("="*80)
print("QUERYING OANDA FOR ALL AVAILABLE INSTRUMENTS")
print("="*80)
print()

# Get account instruments
url = f"{base_url}/accounts/{account_id}/instruments"

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()
    instruments = data.get('instruments', [])

    print(f"Found {len(instruments)} available instruments")
    print()

    # Categorize instruments
    commodities = []
    indices = []
    forex = []
    bonds = []
    metals = []

    for inst in instruments:
        name = inst['name']
        display_name = inst.get('displayName', '')
        inst_type = inst.get('type', '')

        if 'CFD' in inst_type:
            if any(x in display_name.upper() for x in ['SUGAR', 'COFFEE', 'CORN', 'WHEAT', 'SOYBEAN', 'COTTON']):
                commodities.append((name, display_name, inst_type))
            elif any(x in display_name.upper() for x in ['OIL', 'BRENT', 'WTI', 'GAS', 'NATURAL']):
                commodities.append((name, display_name, inst_type))
            elif any(x in display_name.upper() for x in ['SP500', 'S&P', 'DOW', 'NASDAQ', 'DAX', 'FTSE', 'NIKKEI', 'CAC']):
                indices.append((name, display_name, inst_type))
            elif any(x in display_name.upper() for x in ['BOND', 'BUND', 'TREASURY']):
                bonds.append((name, display_name, inst_type))
        elif any(x in name for x in ['XAU', 'XAG', 'XPT', 'XPD', 'XCU']):
            metals.append((name, display_name, inst_type))
        elif '_' in name and len(name) == 7:  # Forex pair format
            forex.append((name, display_name, inst_type))

    # Print categorized results
    if commodities:
        print("="*80)
        print("COMMODITIES (Agricultural & Energy)")
        print("="*80)
        for name, display, itype in sorted(commodities):
            print(f"  {name:20} | {display:40} | {itype}")

    if indices:
        print("\n" + "="*80)
        print("INDICES")
        print("="*80)
        for name, display, itype in sorted(indices):
            print(f"  {name:20} | {display:40} | {itype}")

    if metals:
        print("\n" + "="*80)
        print("METALS")
        print("="*80)
        for name, display, itype in sorted(metals):
            print(f"  {name:20} | {display:40} | {itype}")

    if bonds:
        print("\n" + "="*80)
        print("BONDS")
        print("="*80)
        for name, display, itype in sorted(bonds):
            print(f"  {name:20} | {display:40} | {itype}")

    # Search for specific instruments we're interested in
    print("\n" + "="*80)
    print("SEARCHING FOR SPECIFIC INSTRUMENTS")
    print("="*80)

    search_terms = ['SUGAR', 'COFFEE', 'S&P', 'SP500', 'SPX', 'DAX', 'CORN', 'WHEAT']

    for term in search_terms:
        matches = [inst for inst in instruments if term.upper() in inst.get('displayName', '').upper() or term.upper() in inst['name'].upper()]
        if matches:
            print(f"\n'{term}' matches:")
            for inst in matches:
                print(f"  {inst['name']:20} | {inst.get('displayName', 'N/A'):40} | {inst.get('type', 'N/A')}")
        else:
            print(f"\n'{term}' matches: None found")

    print("\n" + "="*80)
    print("TOTAL SUMMARY")
    print("="*80)
    print(f"  Forex pairs: {len(forex)}")
    print(f"  Metals: {len(metals)}")
    print(f"  Commodities: {len(commodities)}")
    print(f"  Indices: {len(indices)}")
    print(f"  Bonds: {len(bonds)}")
    print(f"  Total: {len(instruments)}")
    print("="*80)

except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {e.response.text}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
