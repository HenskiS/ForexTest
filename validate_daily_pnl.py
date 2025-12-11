"""
Automated Daily P&L Validation
Compares actual live trading P&L vs backtest predictions
Run daily at 6:01 AM PST (after position closes at 6:00 AM)
"""
import pandas as pd
import numpy as np
import sys
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Configuration
PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
SL_PCT = 0.0018
ACCOUNT = 500  # Default account size
LEVERAGE = 5.0  # Total leverage
N_PAIRS = 4
POSITION_SIZE_PER_PAIR = (ACCOUNT * LEVERAGE) / N_PAIRS

def get_target_date():
    """Get the date to validate (yesterday by default)"""
    if len(sys.argv) > 1:
        # Date provided as argument: python validate_daily_pnl.py 2025-12-10
        return sys.argv[1]
    else:
        # Default to yesterday
        yesterday = datetime.now() - timedelta(days=1)
        return yesterday.strftime('%Y-%m-%d')

def load_ohlc_data(pair, date_str):
    """Load OHLC data for a specific pair and date"""
    df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    try:
        candle = df.loc[date_str]
        return {
            'open': float(candle['open']),
            'high': float(candle['high']),
            'low': float(candle['low']),
            'close': float(candle['close']),
            'spread_pct': float(candle['spread_pct'])
        }
    except KeyError:
        return None

def get_oanda_transactions(date_str):
    """Fetch transactions from OANDA API for the target date"""
    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')

    if not api_key or not account_id:
        raise ValueError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set in .env")

    # Use live API
    base_url = "https://api-fxtrade.oanda.com/v3"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # Get transactions for the date range (add some buffer for timezone)
    target_date = pd.to_datetime(date_str)
    from_time = target_date.strftime('%Y-%m-%dT00:00:00Z')
    to_time = (target_date + timedelta(days=1)).strftime('%Y-%m-%dT23:59:59Z')

    url = f"{base_url}/accounts/{account_id}/transactions"
    params = {
        'from': from_time,
        'to': to_time,
        'type': 'ORDER_FILL'
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        result = response.json()
        return result.get('transactions', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching OANDA transactions: {e}")
        return None

def get_actual_trades(date_str):
    """Extract actual trades from OANDA CSV export for the target date"""
    # Read trade data from OANDA CSV export
    df = pd.read_csv('trade_data.csv')
    df['TRANSACTION DATE'] = pd.to_datetime(df['TRANSACTION DATE'], format='mixed')
    df['date'] = df['TRANSACTION DATE'].dt.date

    target_date = pd.to_datetime(date_str).date()

    # Filter to ORDER_FILL transactions on target date
    # Get only new position opens (MARKET_ORDER, not POSITION_CLOSEOUT)
    df_opens = df[
        (df['date'] == target_date) &
        (df['TRANSACTION TYPE'] == 'ORDER_FILL') &
        (df['DETAILS'].str.contains('MARKET_ORDER', na=False)) &
        (~df['DETAILS'].str.contains('POSITION_CLOSEOUT', na=False))
    ].copy()

    if len(df_opens) == 0:
        return {}

    # Get closes for the same date (stop loss, take profit, or manual closes)
    df_closes = df[
        (df['date'] == target_date) &
        (df['TRANSACTION TYPE'] == 'ORDER_FILL') &
        (
            df['DETAILS'].str.contains('STOP_LOSS_ORDER', na=False) |
            df['DETAILS'].str.contains('TAKE_PROFIT_ORDER', na=False) |
            (df['DETAILS'].str.contains('POSITION_CLOSEOUT', na=False))
        )
    ].copy()

    # Extract trades by pair
    trades = {}

    for _, row in df_opens.iterrows():
        instrument = row['INSTRUMENT']
        pair = instrument.replace('/', '')  # USD/JPY -> USDJPY

        direction = 'LONG' if row['DIRECTION'] == 'Buy' else 'SHORT'
        entry_price = float(row['PRICE'])
        entry_time = row['TRANSACTION DATE']

        # Find corresponding close for this pair that happens AFTER the open
        # Can be: STOP_LOSS, TAKE_PROFIT (same day), or POSITION_CLOSEOUT (next day at 6 AM)
        close_candidates = df_closes[
            (df_closes['INSTRUMENT'] == instrument) &
            (df_closes['TRANSACTION DATE'] > entry_time)
        ]

        if len(close_candidates) > 0:
            # Take the first close after the open
            close_row = close_candidates.iloc[0]
            exit_price = float(close_row['PRICE'])

            # P&L is in the PL column (negative for loss)
            actual_pnl = float(close_row['PL'])

            trades[pair] = {
                'signal': direction,
                'entry': entry_price,
                'exit': exit_price,
                'actual_pnl': actual_pnl
            }

    return trades

def calculate_backtest_pnl(pair, data, signal):
    """Calculate what the backtest would predict for P&L"""
    entry_price = data['open']
    spread_pct = data['spread_pct']

    if signal == 'LONG':
        # LONG: Pay spread on entry
        actual_entry = entry_price * (1 + spread_pct)
        sl_price = entry_price * (1 - SL_PCT)

        # Check if low hit SL
        if data['low'] <= sl_price:
            # Stop loss hit
            exit_price = sl_price * (1 - spread_pct)
            pnl_pct = (exit_price / actual_entry) - 1
            exit_type = 'STOP_LOSS'
        else:
            # End of day close
            exit_price = data['close'] * (1 - spread_pct)
            pnl_pct = (exit_price / actual_entry) - 1
            exit_type = 'EOD_CLOSE'

    else:  # SHORT
        # SHORT: Pay spread on entry
        actual_entry = entry_price * (1 - spread_pct)
        sl_price = entry_price * (1 + SL_PCT)

        # Check if high hit SL
        if data['high'] >= sl_price:
            # Stop loss hit
            exit_price = sl_price * (1 + spread_pct)
            pnl_pct = (actual_entry / exit_price) - 1
            exit_type = 'STOP_LOSS'
        else:
            # End of day close
            exit_price = data['close'] * (1 + spread_pct)
            pnl_pct = (actual_entry / exit_price) - 1
            exit_type = 'EOD_CLOSE'

    backtest_pnl = POSITION_SIZE_PER_PAIR * pnl_pct
    return backtest_pnl, exit_type, sl_price

def main():
    date_str = get_target_date()

    print("=" * 80)
    print(f"DAILY P&L VALIDATION - {date_str}")
    print("=" * 80)
    print(f"Account: ${ACCOUNT}")
    print(f"Leverage: {LEVERAGE}x")
    print(f"Position size per pair: ${POSITION_SIZE_PER_PAIR:.2f}")
    print()

    # Load actual trades
    print("Loading actual trades from CSV...")
    actual_trades = get_actual_trades(date_str)

    if not actual_trades:
        print(f"ERROR: No trades found for {date_str}")
        print("Make sure trade_data.csv is updated with latest transactions.")
        sys.exit(1)

    print(f"Found {len(actual_trades)} trades")
    print()

    # Validate each trade
    total_backtest_pnl = 0
    total_actual_pnl = 0
    comparison_results = []

    for pair in PAIRS:
        if pair not in actual_trades:
            print(f"{pair}: NO TRADE")
            continue

        trade = actual_trades[pair]

        # Load OHLC data
        ohlc = load_ohlc_data(pair, date_str)
        if ohlc is None:
            print(f"{pair}: ERROR - No OHLC data for {date_str}")
            continue

        # Calculate backtest prediction
        backtest_pnl, exit_type, sl_price = calculate_backtest_pnl(pair, ohlc, trade['signal'])
        actual_pnl = trade['actual_pnl']

        # Compare
        diff = abs(backtest_pnl - actual_pnl)
        error_pct = (diff / abs(actual_pnl)) * 100 if actual_pnl != 0 else 0

        total_backtest_pnl += backtest_pnl
        total_actual_pnl += actual_pnl

        comparison_results.append({
            'pair': pair,
            'signal': trade['signal'],
            'exit_type': exit_type,
            'backtest_pnl': backtest_pnl,
            'actual_pnl': actual_pnl,
            'diff': diff,
            'error_pct': error_pct
        })

        # Print result
        status = "[OK]" if error_pct < 15 else "[WARN]" if error_pct < 25 else "[FAIL]"
        print(f"{pair} ({trade['signal']}):")
        print(f"  Backtest: ${backtest_pnl:+.2f} | Actual: ${actual_pnl:+.2f} | Diff: ${diff:.2f} ({error_pct:.1f}%) {status}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Backtest P&L: ${total_backtest_pnl:+.2f}")
    print(f"Total Actual P&L:   ${total_actual_pnl:+.2f}")
    print(f"Total Difference:   ${abs(total_backtest_pnl - total_actual_pnl):.2f}")

    overall_error = (abs(total_backtest_pnl - total_actual_pnl) / abs(total_actual_pnl)) * 100 if total_actual_pnl != 0 else 0
    print(f"Overall Error:      {overall_error:.1f}%")
    print()

    # Status
    if overall_error < 15:
        print("STATUS: [PASS] Backtest predictions are accurate")
        exit_code = 0
    elif overall_error < 25:
        print("STATUS: [WARNING] Backtest predictions show moderate deviation")
        exit_code = 0
    else:
        print("STATUS: [FAIL] Backtest predictions are significantly off")
        print("       Check: leverage settings, spread data, or entry/exit logic")
        exit_code = 1

    print("=" * 80)

    # Log to file for tracking
    log_file = 'daily_pnl_validation.log'
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().isoformat()} | {date_str} | ")
        f.write(f"Backtest: ${total_backtest_pnl:+.2f} | Actual: ${total_actual_pnl:+.2f} | ")
        f.write(f"Error: {overall_error:.1f}%\n")

    sys.exit(exit_code)

if __name__ == '__main__':
    main()
