"""
List ALL OANDA instruments available to your account.
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

base_url = "https://api-fxtrade.oanda.com/v3"

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

print("="*80)
print("ALL AVAILABLE INSTRUMENTS IN YOUR OANDA ACCOUNT")
print("="*80)
print()

url = f"{base_url}/accounts/{account_id}/instruments"

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()
    instruments = data.get('instruments', [])

    print(f"Total: {len(instruments)} instruments\n")

    # Print all instruments with details
    for inst in instruments:
        name = inst['name']
        display_name = inst.get('displayName', 'N/A')
        inst_type = inst.get('type', 'N/A')

        print(f"{name:15} | {display_name:45} | {inst_type}")

    print("\n" + "="*80)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
