"""
Fetch 4-hour forex data from Dukascopy Bank SA

Usage:
    python fetch_dukascopy_4hour.py EURUSD
    python fetch_dukascopy_4hour.py GBPUSD
"""

import sys
from datetime import datetime, timedelta
import pandas as pd

try:
    import dukascopy_python
except ImportError:
    print("ERROR: dukascopy-python not installed")
    print("Install with: pip install dukascopy-python")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python fetch_dukascopy_4hour.py <CURRENCY_PAIR>")
    print("Example: python fetch_dukascopy_4hour.py EURUSD")
    sys.exit(1)

CURRENCY_PAIR = sys.argv[1].upper()

# Map currency pairs to Dukascopy instrument constants
INSTRUMENT_MAP = {
    'EURUSD': 'INSTRUMENT_FX_MAJORS_EUR_USD',
    'GBPUSD': 'INSTRUMENT_FX_MAJORS_GBP_USD',
    'USDJPY': 'INSTRUMENT_FX_MAJORS_USD_JPY',
    'AUDUSD': 'INSTRUMENT_FX_MAJORS_AUD_USD',
    'USDCHF': 'INSTRUMENT_FX_MAJORS_USD_CHF',
    'USDCAD': 'INSTRUMENT_FX_MAJORS_USD_CAD',
    'NZDUSD': 'INSTRUMENT_FX_MAJORS_NZD_USD',
}

if CURRENCY_PAIR not in INSTRUMENT_MAP:
    print(f"ERROR: {CURRENCY_PAIR} not supported")
    print(f"Supported pairs: {', '.join(INSTRUMENT_MAP.keys())}")
    sys.exit(1)

print(f"{'='*80}")
print(f"FETCHING 4-HOUR DATA FROM DUKASCOPY - {CURRENCY_PAIR}")
print(f"{'='*80}\n")

# Get instrument constant
instrument_name = INSTRUMENT_MAP[CURRENCY_PAIR]
try:
    # Import from dukascopy_python.instruments
    from dukascopy_python import instruments
    instrument = getattr(instruments, instrument_name)
except AttributeError:
    print(f"ERROR: Could not find instrument {instrument_name}")
    sys.exit(1)

# Date range: from 2001 to present (24 years for sufficient training data)
end_date = datetime.now()
start_date = datetime(2001, 1, 1)

print(f"Instrument: {instrument_name}")
print(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
print(f"End Date:   {end_date.strftime('%Y-%m-%d')}")
print(f"Interval:   4 HOUR")
print(f"Offer Side: BID\n")

# Fetch data
print("Fetching data from Dukascopy...")
try:
    df = dukascopy_python.fetch(
        instrument,
        dukascopy_python.INTERVAL_HOUR_4,
        dukascopy_python.OFFER_SIDE_BID,
        start_date,
        end_date,
    )

    if df is None or df.empty:
        print("ERROR: No data returned from Dukascopy")
        sys.exit(1)

    print(f"Success! Fetched {len(df)} candles\n")

    # Reset index to get datetime as column
    df = df.reset_index()

    # Rename timestamp or index to 'date'
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})
    elif 'index' in df.columns:
        df = df.rename(columns={'index': 'date'})

    # Ensure we have the standard OHLCV columns
    expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in expected_cols:
        if col not in df.columns:
            print(f"WARNING: Missing column '{col}'")

    # Display info
    print("Data Summary:")
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Start:   {df['date'].min()}")
    print(f"  End:     {df['date'].max()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())

    # Save to CSV
    output_file = f'data/{CURRENCY_PAIR}_4hour_dukascopy.csv'
    df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"DATA SAVED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"File: {output_file}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nNext step: Add technical features using add_features.py or prepare_and_test_currency_pair.py")

except Exception as e:
    print(f"ERROR: Failed to fetch data from Dukascopy")
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
