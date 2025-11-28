"""
Test script for new modular trading infrastructure.

Tests all trading modules to ensure imports work and basic functionality is correct.
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("TESTING TRADING MODULES")
print("="*80)

# Test 1: Config module
print("\n1. Testing trading/config.py...")
try:
    from trading.config import TradingConfig

    print(f"   Stop Loss: {TradingConfig.STOP_LOSS_PCT*100:.2f}%")
    print(f"   Take Profit: {TradingConfig.TAKE_PROFIT_PCT*100:.2f}%")
    print(f"   Train Window: {TradingConfig.TRAIN_WINDOW_SIZE} days")
    print(f"   Technical Features: {len(TradingConfig.TECHNICAL_FEATURES)} indicators")
    print(f"   Default Pairs: {', '.join(TradingConfig.DEFAULT_PAIRS)}")
    print(f"   Leverage: {TradingConfig.LEVERAGE}:1")
    print("   [OK] Config module loaded successfully")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

# Test 2: Market utils module
print("\n2. Testing trading/market_utils.py...")
try:
    from trading.market_utils import is_forex_market_open, calculate_technical_features

    # Test market hours check
    is_open = is_forex_market_open()
    print(f"   Forex market currently open: {is_open}")

    # Test technical indicators on sample data
    sample_data = pd.DataFrame({
        'open': np.random.uniform(1.08, 1.09, 100),
        'high': np.random.uniform(1.09, 1.10, 100),
        'low': np.random.uniform(1.07, 1.08, 100),
        'close': np.random.uniform(1.08, 1.09, 100)
    })

    df_with_features = calculate_technical_features(sample_data)
    print(f"   Sample data: {len(sample_data)} rows")
    print(f"   Features calculated: {len([c for c in df_with_features.columns if c not in ['open', 'high', 'low', 'close']])} indicators")
    print("   [OK] Market utils module working")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Notifications module
print("\n3. Testing trading/notifications.py...")
try:
    from trading.notifications import NotificationService

    notifier = NotificationService()
    print(f"   SMS enabled: {notifier.sms_enabled}")
    print(f"   Telegram enabled: {notifier.telegram_enabled}")
    print("   [OK] Notification module loaded")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: OANDA client module
print("\n4. Testing trading/oanda_client.py...")
try:
    from trading.oanda_client import OandaClient

    # Don't actually create client (requires credentials)
    print("   [OK] OANDA client module imported successfully")
    print("   (Skipping actual API connection test)")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 5: Position manager module
print("\n5. Testing trading/position_manager.py...")
try:
    from trading.position_manager import PositionManager

    # Create test position manager
    pm = PositionManager('EURUSD')
    print(f"   Pair: {pm.pair}")
    print(f"   Has position: {pm.has_position()}")
    print(f"   State file: {pm.state_file}")

    # Test opening/closing position (doesn't actually trade)
    pm.open_position(direction=1, entry_price=1.08500, position_size=250.0)
    print(f"   Test position opened: {pm.has_position()}")

    # Test P&L calculation
    pnl = pm.calculate_pnl(exit_price=1.08700)
    print(f"   Test P&L: {pnl['pnl_pct']:.2f}% (${pnl['pnl_dollars']:.2f})")

    pm.close_position()
    print(f"   Test position closed: {not pm.has_position()}")

    print("   [OK] Position manager module working")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Trading models module
print("\n6. Testing trading/trading_models.py...")
try:
    from trading.trading_models import TradingModel

    # Create test model (doesn't train yet)
    model = TradingModel('EURUSD')
    print(f"   Pair: {model.pair}")
    print(f"   Prediction buffer size: {len(model.prediction_buffer)}")
    print(f"   Hyperparameters loaded: {len(model.best_params)} params")

    # Test prediction buffer operations
    model.add_prediction_to_buffer(0.00123)
    print(f"   Added test prediction to buffer: {len(model.prediction_buffer)} predictions")

    print("   [OK] Trading models module working")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Main package imports
print("\n7. Testing trading package imports...")
try:
    from trading import (
        TradingConfig,
        OandaClient,
        PositionManager,
        TradingModel,
        is_forex_market_open,
        calculate_technical_features,
        NotificationService
    )

    print("   [OK] All imports working from trading package")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Summary
print("\n"+"="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nModular trading infrastructure is ready to use!")
print("\nNext steps:")
print("  1. Create single-pair strategy (strategies/single_pair_strategy.py)")
print("  2. Create multi-pair strategy (strategies/multi_pair_strategy.py)")
print("  3. Refactor oanda_production_trader.py to use new modules")
