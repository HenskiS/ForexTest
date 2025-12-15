"""
Notification Service for OANDA Trading Bot
Supports SMS (Twilio) and Telegram notifications
"""
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class NotificationService:
    """Send trade notifications via SMS and/or Telegram"""

    def __init__(self):
        """Initialize notification service with credentials from .env"""
        # Twilio configuration (for SMS)
        self.twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.twilio_from_number = os.getenv('TWILIO_FROM_NUMBER')
        self.twilio_to_number = os.getenv('TWILIO_TO_NUMBER')

        # Telegram configuration
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Check which services are configured
        self.sms_enabled = all([
            self.twilio_account_sid,
            self.twilio_auth_token,
            self.twilio_from_number,
            self.twilio_to_number
        ])

        self.telegram_enabled = all([
            self.telegram_bot_token,
            self.telegram_chat_id
        ])

        if not self.sms_enabled and not self.telegram_enabled:
            print("WARNING: No notification services configured")
            print("   Add credentials to .env to enable notifications")

    def send_sms(self, message):
        """Send SMS via Twilio"""
        if not self.sms_enabled:
            return False

        try:
            from twilio.rest import Client
            client = Client(self.twilio_account_sid, self.twilio_auth_token)

            message_obj = client.messages.create(
                body=message,
                from_=self.twilio_from_number,
                to=self.twilio_to_number
            )

            print(f"SMS sent (SID: {message_obj.sid})")
            return True

        except ImportError:
            print("Twilio package not installed. Run: pip install twilio")
            return False
        except Exception as e:
            print(f"Failed to send SMS: {e}")
            return False

    def send_telegram(self, message):
        """Send message via Telegram"""
        if not self.telegram_enabled:
            return False

        try:
            import requests

            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }

            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()

            print(f"Telegram message sent")
            return True

        except Exception as e:
            print(f"Failed to send Telegram message: {e}")
            return False

    def notify_trade_entry(self, pair, direction, entry_price, position_size, stop_loss, take_profit):
        """Send notification when entering a trade"""
        message = f"""
*TRADE OPENED* - {pair}

Direction: {direction}
Entry Price: {entry_price:.5f}
Position Size: ${position_size:,.2f}
Stop Loss: {stop_loss:.5f}
Take Profit: {take_profit:.5f}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()

        # Send via all enabled channels
        if self.sms_enabled:
            # SMS version (shorter)
            sms_message = f"TRADE OPENED: {direction} {pair} @ {entry_price:.5f} | Size: ${position_size:,.0f}"
            self.send_sms(sms_message)

        if self.telegram_enabled:
            self.send_telegram(message)

    def notify_trade_exit(self, pair, direction, entry_price, exit_price, pnl_pct, pnl_dollars, exit_reason, days_held):
        """Send notification when exiting a trade"""
        # Determine if win or loss
        outcome = "WIN" if pnl_pct > 0 else "LOSS"
        pnl_sign = "+" if pnl_pct > 0 else ""

        message = f"""
*TRADE CLOSED* - {pair} - {outcome}

Direction: {direction}
Entry: {entry_price:.5f}
Exit: {exit_price:.5f}

P&L: {pnl_sign}{pnl_pct:.2f}% (${pnl_sign}{pnl_dollars:,.2f})
Reason: {exit_reason}
Days Held: {days_held}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()

        # Send via all enabled channels
        if self.sms_enabled:
            # SMS version (shorter)
            sms_message = f"TRADE CLOSED: {direction} {pair} | P&L: {pnl_sign}{pnl_pct:.2f}% (${pnl_sign}{pnl_dollars:,.0f}) | {exit_reason}"
            self.send_sms(sms_message)

        if self.telegram_enabled:
            self.send_telegram(message)

    def notify_error(self, error_message):
        """Send notification for errors"""
        message = f"""
*BOT ERROR*

{error_message}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()

        if self.sms_enabled:
            self.send_sms(f"BOT ERROR: {error_message}")

        if self.telegram_enabled:
            self.send_telegram(message)
