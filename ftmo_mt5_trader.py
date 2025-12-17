"""
FTMO MetaTrader 5 Signal Reversal Trading Bot

For use with FTMO Challenge/Funded accounts via MetaTrader 5.

FTMO-Safe Configuration:
- 2.5% SL / 2.5% TP (independent per trade)
- 6.2% allocation per slot, 1.5x leverage
- 5 slots max per pair
- Signal reversal exit (close all on opposite signal)

Requirements:
- MetaTrader 5 terminal installed and running
- pip install MetaTrader5
- FTMO account credentials configured in MT5

Expected Performance:
- ~38% annual return
- ~8% max drawdown
- ~2.3% worst day
"""
import sys
import os
import argparse
from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd

# Try to import MT5 - will fail on systems without it
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("WARNING: MetaTrader5 not installed. Install with: pip install MetaTrader5")
    print("MT5 only works on Windows with MetaTrader 5 terminal installed.")


class FTMOConfig:
    """FTMO-safe trading configuration"""

    # Risk Parameters
    STOP_LOSS_PCT = 0.025      # 2.5% stop loss
    TAKE_PROFIT_PCT = 0.025    # 2.5% take profit

    # Position Sizing
    ALLOCATION_PER_SLOT = 0.062  # 6.2% per slot
    LEVERAGE = 1.5              # Effective leverage target
    MAX_SLOTS = 5               # 5 slots max per pair

    # Signal Parameters
    PERCENTILE_LOWER = 10
    PERCENTILE_UPPER = 90
    BUFFER_SIZE = 200
    BUFFER_WARMUP = 50

    # MT5 Symbol mapping (FTMO uses standard MT5 symbols)
    PAIRS = {
        'EURUSD': 'EURUSD',
        'GBPUSD': 'GBPUSD',
        'AUDUSD': 'AUDUSD',
        'USDJPY': 'USDJPY',
        'EURJPY': 'EURJPY',
        'USDCAD': 'USDCAD',
        'USDCHF': 'USDCHF',
        'NZDUSD': 'NZDUSD'
    }


class PredictionBuffer:
    """Manages prediction buffer for signal generation"""

    def __init__(self, pair, buffer_file=None):
        self.pair = pair
        self.buffer_file = buffer_file or f'ftmo_buffer_{pair}.pkl'
        self.predictions = []
        self.load()

    def load(self):
        if os.path.exists(self.buffer_file):
            try:
                with open(self.buffer_file, 'rb') as f:
                    self.predictions = pickle.load(f)
                print(f"  Loaded {len(self.predictions)} predictions for {self.pair}")
            except:
                self.predictions = []

    def save(self):
        with open(self.buffer_file, 'wb') as f:
            pickle.dump(self.predictions, f)

    def add(self, prediction):
        self.predictions.append(prediction)
        if len(self.predictions) > FTMOConfig.BUFFER_SIZE:
            self.predictions = self.predictions[-FTMOConfig.BUFFER_SIZE:]
        self.save()

    def get_signal(self, prediction):
        """Generate signal based on prediction and buffer percentiles"""
        if len(self.predictions) < FTMOConfig.BUFFER_WARMUP:
            return 0

        lower = np.percentile(self.predictions, FTMOConfig.PERCENTILE_LOWER)
        upper = np.percentile(self.predictions, FTMOConfig.PERCENTILE_UPPER)

        if prediction >= upper:
            return 1  # Long
        elif prediction <= lower:
            return -1  # Short
        return 0  # Neutral


class MT5Client:
    """MetaTrader 5 trading client"""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.connected = False

        if not MT5_AVAILABLE:
            print("MT5 not available - running in simulation mode")
            return

        if not self.dry_run:
            if not mt5.initialize():
                print(f"MT5 initialization failed: {mt5.last_error()}")
                return
            self.connected = True

            # Print account info
            account_info = mt5.account_info()
            if account_info:
                print(f"\nMT5 Connected:")
                print(f"  Account: {account_info.login}")
                print(f"  Server: {account_info.server}")
                print(f"  Balance: ${account_info.balance:,.2f}")
                print(f"  Equity: ${account_info.equity:,.2f}")

    def shutdown(self):
        if self.connected:
            mt5.shutdown()

    def get_account_balance(self):
        if self.dry_run or not self.connected:
            return 100000  # Mock balance
        account = mt5.account_info()
        return account.balance if account else None

    def get_current_price(self, symbol):
        if self.dry_run or not self.connected:
            # Return mock price for dry run
            mock_prices = {
                'EURUSD': 1.05000, 'GBPUSD': 1.27000, 'AUDUSD': 0.64000,
                'USDJPY': 150.000, 'EURJPY': 157.500, 'USDCAD': 1.43000,
                'USDCHF': 0.89000, 'NZDUSD': 0.58000
            }
            price = mock_prices.get(symbol, 1.0)
            return {'bid': price - 0.0001, 'ask': price + 0.0001, 'mid': price}

        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return {'bid': tick.bid, 'ask': tick.ask, 'mid': (tick.bid + tick.ask) / 2}
        return None

    def get_symbol_info(self, symbol):
        if self.dry_run or not self.connected:
            # Mock symbol info
            if 'JPY' in symbol:
                return {'digits': 3, 'point': 0.001, 'trade_contract_size': 100000}
            return {'digits': 5, 'point': 0.00001, 'trade_contract_size': 100000}

        info = mt5.symbol_info(symbol)
        if info:
            return {
                'digits': info.digits,
                'point': info.point,
                'trade_contract_size': info.trade_contract_size
            }
        return None

    def get_positions(self, symbol=None):
        """Get open positions"""
        if self.dry_run or not self.connected:
            return []

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        return list(positions) if positions else []

    def place_order(self, symbol, order_type, volume, sl_price, tp_price):
        """Place a market order with SL and TP"""
        if self.dry_run:
            print(f"  [DRY RUN] Would place {order_type} {volume:.2f} lots {symbol}")
            print(f"    SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            return {'success': True, 'ticket': 0, 'price': self.get_current_price(symbol)['mid']}

        if not self.connected:
            return {'success': False, 'error': 'Not connected'}

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {'success': False, 'error': f'Symbol {symbol} not found'}

        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        price = mt5.symbol_info_tick(symbol)
        if price is None:
            return {'success': False, 'error': 'Failed to get price'}

        if order_type == 'BUY':
            trade_type = mt5.ORDER_TYPE_BUY
            entry_price = price.ask
        else:
            trade_type = mt5.ORDER_TYPE_SELL
            entry_price = price.bid

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': volume,
            'type': trade_type,
            'price': entry_price,
            'sl': sl_price,
            'tp': tp_price,
            'deviation': 20,
            'magic': 123456,
            'comment': 'FTMO Signal Reversal',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {'success': True, 'ticket': result.order, 'price': entry_price}
        else:
            return {'success': False, 'error': f'Order failed: {result.retcode}'}

    def close_position(self, ticket):
        """Close a position by ticket"""
        if self.dry_run:
            print(f"  [DRY RUN] Would close position {ticket}")
            return True

        if not self.connected:
            return False

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False

        pos = position[0]

        if pos.type == mt5.POSITION_TYPE_BUY:
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(pos.symbol).bid
        else:
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(pos.symbol).ask

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': pos.symbol,
            'volume': pos.volume,
            'type': trade_type,
            'position': ticket,
            'price': price,
            'deviation': 20,
            'magic': 123456,
            'comment': 'Close FTMO position',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def close_all_symbol_positions(self, symbol):
        """Close all positions for a symbol"""
        positions = self.get_positions(symbol)
        for pos in positions:
            self.close_position(pos.ticket)

    def get_historical_data(self, symbol, count=700):
        """Get historical daily OHLC data"""
        if self.dry_run or not self.connected:
            # Try to load from local CSV files
            csv_file = f'data/{symbol}_1day_oanda.csv'
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                return df.tail(count)
            else:
                print(f"  No local data for {symbol}")
                return pd.DataFrame()

        # Fetch from MT5
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['date'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('date')
        df = df.rename(columns={'tick_volume': 'volume'})

        return df[['open', 'high', 'low', 'close', 'volume']]


class SlotManager:
    """Manages slots for a single pair"""

    def __init__(self, pair):
        self.pair = pair
        self.state_file = f'ftmo_slots_{pair}.pkl'
        self.slots = []  # List of {ticket, direction, entry_price, entry_date}
        self.direction = 0
        self.load()

    def load(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.slots = state.get('slots', [])
                    self.direction = state.get('direction', 0)
            except:
                pass

    def save(self):
        with open(self.state_file, 'wb') as f:
            pickle.dump({'slots': self.slots, 'direction': self.direction}, f)

    def slot_count(self):
        return len(self.slots)

    def can_add_slot(self, signal):
        if len(self.slots) >= FTMOConfig.MAX_SLOTS:
            return False
        if self.direction != 0 and self.direction != signal:
            return False
        # One entry per day
        today = datetime.now().date()
        for slot in self.slots:
            if slot.get('entry_date') and slot['entry_date'].date() == today:
                return False
        return True

    def add_slot(self, ticket, direction, entry_price):
        self.slots.append({
            'ticket': ticket,
            'direction': direction,
            'entry_price': entry_price,
            'entry_date': datetime.now()
        })
        self.direction = direction
        self.save()

    def clear_all(self):
        self.slots = []
        self.direction = 0
        self.save()

    def get_tickets(self):
        return [s['ticket'] for s in self.slots]


class ANNModel:
    """ANN model with live training capability"""

    # ANN Hyperparameters (same as Sleep Well)
    ANN_PARAMS = {
        'hidden_layer_sizes': (13, 20, 31),
        'activation': 'tanh',
        'solver': 'sgd',
        'learning_rate_init': 0.001,
        'momentum': 0.4,
        'batch_size': 64,
        'max_iter': 20,
        'alpha': 0.0001,
        'learning_rate': 'adaptive',
        'random_state': 42,
        'verbose': False,
        'warm_start': False,
        'early_stopping': False
    }

    TRAIN_WINDOW = 378

    FEATURES = [
        'momentum', 'avg_price', 'range', 'ohlc',
        'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
        'macd', 'macd_signal', 'macd_hist',
        'adx', 'plus_di', 'minus_di',
        'rsi', 'stoch_k', 'stoch_d',
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
        'atr', 'volume_sma',
        'close_to_high', 'close_to_low',
        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_10'
    ]

    def __init__(self, pair):
        self.pair = pair
        self.model = None
        self.scaler = None

    def calculate_features(self, df):
        """Calculate technical indicators"""
        df = df.copy()

        # Basic features
        df['momentum'] = df['close'].pct_change()
        df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['range'] = df['high'] - df['low']
        df['ohlc'] = df['avg_price']

        # EMAs
        for period in [10, 20, 50, 100, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = pd.concat([df['high'] - df['low'],
                       abs(df['high'] - df['close'].shift()),
                       abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df['adx'] = dx.rolling(window=14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Stochastic
        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # Bollinger Bands
        middle = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = middle + (std * 2)
        df['bb_middle'] = middle
        df['bb_lower'] = middle - (std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        # ATR
        df['atr'] = atr

        # Volume SMA (use 1 if no volume)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
        else:
            df['volume_sma'] = 1.0

        # Price position
        df['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['close_to_low'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)

        # Lagged returns
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = df['close'].pct_change(lag)

        # Target
        df['target'] = df['close'].pct_change(1).shift(-1)

        return df

    def train_and_predict(self, df):
        """Train model on data and return prediction for latest row"""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import MinMaxScaler

        # Calculate features
        df = self.calculate_features(df)
        df = df.dropna(subset=self.FEATURES + ['target'])

        if len(df) < self.TRAIN_WINDOW + 10:
            print(f"  Not enough data: {len(df)} rows")
            return 0.0

        # Use last TRAIN_WINDOW rows for training, predict on final row
        train_data = df.iloc[-(self.TRAIN_WINDOW + 1):-1]
        predict_row = df.iloc[[-1]]

        # Scale features
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(train_data[self.FEATURES])
        y_train = train_data['target'].values

        X_predict = self.scaler.transform(predict_row[self.FEATURES])

        # Train model
        self.model = MLPRegressor(**self.ANN_PARAMS)
        self.model.fit(X_train, y_train)

        # Predict
        prediction = self.model.predict(X_predict)[0]
        return prediction


class FTMOTrader:
    """Main FTMO trading bot"""

    def __init__(self, pairs, dry_run=False):
        self.pairs = pairs
        self.dry_run = dry_run

        print("\nInitializing FTMO Trader...")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print(f"Pairs: {', '.join(pairs)}")

        # Initialize MT5 client
        self.client = MT5Client(dry_run=dry_run)

        # Initialize per-pair components
        self.buffers = {}
        self.slots = {}
        self.models = {}

        print("\nLoading models and buffers:")
        for pair in pairs:
            self.buffers[pair] = PredictionBuffer(pair)
            self.slots[pair] = SlotManager(pair)
            self.models[pair] = ANNModel(pair)

    def run(self):
        """Execute trading cycle"""
        print(f"\n{'='*70}")
        print(f"FTMO SIGNAL REVERSAL - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        # Check if market is open (forex hours)
        now = datetime.now()
        # Skip weekends
        if now.weekday() >= 5:
            print("\nMarket closed (weekend)")
            return

        # Get account balance
        balance = self.client.get_account_balance()
        print(f"\nAccount Balance: ${balance:,.2f}")

        capital_per_slot = balance * FTMOConfig.ALLOCATION_PER_SLOT
        position_size = capital_per_slot * FTMOConfig.LEVERAGE
        print(f"Position size per slot: ${position_size:,.2f}")

        # Process each pair
        signals = {}

        print(f"\n{'='*70}")
        print("TRAINING MODELS & GENERATING SIGNALS")
        print(f"{'='*70}")

        for pair in self.pairs:
            mt5_symbol = FTMOConfig.PAIRS[pair]
            print(f"\n{pair}:")

            # Fetch historical data
            df = self.client.get_historical_data(mt5_symbol, count=700)

            if df.empty:
                print(f"  No data available - skipping")
                signals[pair] = 0
                continue

            print(f"  Data: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

            # Train model and get prediction
            prediction = self.models[pair].train_and_predict(df)
            print(f"  Prediction: {prediction:.6f}")

            # Add to buffer and get signal
            self.buffers[pair].add(prediction)
            signal = self.buffers[pair].get_signal(prediction)
            signals[pair] = signal

            signal_str = 'LONG' if signal == 1 else ('SHORT' if signal == -1 else 'NEUTRAL')
            slots = self.slots[pair].slot_count()
            print(f"  Signal: {signal_str}, Slots: {slots}/{FTMOConfig.MAX_SLOTS}")

        # Check for signal reversals (exit all slots on opposite signal)
        print(f"\n{'='*70}")
        print("CHECKING EXITS")
        print(f"{'='*70}")

        for pair in self.pairs:
            slot_mgr = self.slots[pair]
            signal = signals[pair]

            if slot_mgr.slot_count() > 0 and signal != 0 and signal != slot_mgr.direction:
                print(f"\n{pair}: SIGNAL REVERSAL - closing {slot_mgr.slot_count()} slots")

                # Close all positions for this symbol
                mt5_symbol = FTMOConfig.PAIRS[pair]
                self.client.close_all_symbol_positions(mt5_symbol)
                slot_mgr.clear_all()

        # Process new entries
        print(f"\n{'='*70}")
        print("PROCESSING ENTRIES")
        print(f"{'='*70}")

        for pair in self.pairs:
            slot_mgr = self.slots[pair]
            signal = signals[pair]

            if signal == 0:
                continue

            if not slot_mgr.can_add_slot(signal):
                continue

            mt5_symbol = FTMOConfig.PAIRS[pair]
            price_data = self.client.get_current_price(mt5_symbol)

            if not price_data:
                print(f"{pair}: Failed to get price")
                continue

            current_price = price_data['mid']

            # Calculate lot size based on position value
            # Standard lot = 100,000 units
            symbol_info = self.client.get_symbol_info(mt5_symbol)
            contract_size = symbol_info.get('trade_contract_size', 100000) if symbol_info else 100000

            # For JPY pairs, price is ~150, for others ~1
            if 'JPY' in pair:
                lots = position_size / (current_price * 100)  # JPY pairs
            else:
                lots = position_size / (current_price * contract_size)

            lots = round(lots, 2)  # Round to 2 decimals
            lots = max(0.01, lots)  # Minimum lot size

            # Calculate SL and TP prices
            if signal == 1:  # Long
                sl_price = current_price * (1 - FTMOConfig.STOP_LOSS_PCT)
                tp_price = current_price * (1 + FTMOConfig.TAKE_PROFIT_PCT)
                order_type = 'BUY'
            else:  # Short
                sl_price = current_price * (1 + FTMOConfig.STOP_LOSS_PCT)
                tp_price = current_price * (1 - FTMOConfig.TAKE_PROFIT_PCT)
                order_type = 'SELL'

            print(f"\n{pair}: {order_type} {lots:.2f} lots @ {current_price:.5f}")
            print(f"  SL: {sl_price:.5f}, TP: {tp_price:.5f}")

            result = self.client.place_order(mt5_symbol, order_type, lots, sl_price, tp_price)

            if result['success']:
                slot_mgr.add_slot(result.get('ticket', 0), signal, result.get('price', current_price))
                print(f"  [OK] Order placed")
            else:
                print(f"  [FAIL] {result.get('error', 'Unknown error')}")

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        total_slots = sum(s.slot_count() for s in self.slots.values())
        max_slots = len(self.pairs) * FTMOConfig.MAX_SLOTS
        print(f"Active slots: {total_slots}/{max_slots}")

        for pair in self.pairs:
            slot_mgr = self.slots[pair]
            if slot_mgr.slot_count() > 0:
                dir_str = 'LONG' if slot_mgr.direction == 1 else 'SHORT'
                print(f"  {pair}: {dir_str} - {slot_mgr.slot_count()}/{FTMOConfig.MAX_SLOTS}")

    def shutdown(self):
        self.client.shutdown()


def main():
    parser = argparse.ArgumentParser(description='FTMO MT5 Signal Reversal Trader')
    parser.add_argument('--dry-run', action='store_true', help='Simulate only, no real trades')
    parser.add_argument('--pairs', nargs='+', default=list(FTMOConfig.PAIRS.keys()),
                        help='Pairs to trade')
    args = parser.parse_args()

    print("="*70)
    print("FTMO METATRADER 5 SIGNAL REVERSAL TRADER")
    print("="*70)
    print(f"\nFTMO-Safe Parameters:")
    print(f"  SL/TP: {FTMOConfig.STOP_LOSS_PCT*100:.1f}% / {FTMOConfig.TAKE_PROFIT_PCT*100:.1f}%")
    print(f"  Allocation: {FTMOConfig.ALLOCATION_PER_SLOT*100:.1f}% per slot")
    print(f"  Leverage: {FTMOConfig.LEVERAGE}x")
    print(f"  Max slots: {FTMOConfig.MAX_SLOTS} per pair")

    try:
        trader = FTMOTrader(
            pairs=[p.upper() for p in args.pairs],
            dry_run=args.dry_run
        )
        trader.run()
        trader.shutdown()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
