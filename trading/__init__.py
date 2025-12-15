"""
Trading Infrastructure Modules

This package contains the core trading components:
- oanda_client: OANDA API communication
- position_manager: Position tracking and state management
- trading_models: XGBoost model training and prediction
- market_utils: Market hours and technical indicators
- notifications: Trading notifications (Discord, Telegram)
- config: Trading parameters and configuration
"""

from .config import TradingConfig
from .oanda_client import OandaClient
from .position_manager import PositionManager
from .trading_models import TradingModel
from .market_utils import is_forex_market_open, calculate_technical_features
from .notifications import NotificationService

__all__ = [
    'TradingConfig',
    'OandaClient',
    'PositionManager',
    'TradingModel',
    'is_forex_market_open',
    'calculate_technical_features',
    'NotificationService'
]
