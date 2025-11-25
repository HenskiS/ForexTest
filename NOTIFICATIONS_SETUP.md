# Notification Setup Guide

Your trading bot now supports real-time notifications via **SMS (Twilio)** and/or **Telegram** when trades are opened and closed.

## Features

✅ Get notified when trades are **opened** (entry price, position size, stops)
✅ Get notified when trades are **closed** with **P&L in both % and dollars**
✅ Support for both SMS and Telegram (use one or both!)

---

## Option 1: SMS Notifications via Twilio

### Step 1: Sign up for Twilio
1. Go to [https://www.twilio.com/try-twilio](https://www.twilio.com/try-twilio)
2. Sign up for a free trial account ($15 credit)
3. Verify your phone number

### Step 2: Get your credentials
1. From your Twilio Console Dashboard, copy:
   - **Account SID**
   - **Auth Token**
2. Get a Twilio phone number:
   - Click "Get a Trial Number" or buy a number
   - Copy the phone number (format: +1234567890)

### Step 3: Add to .env file
```bash
# Twilio SMS Configuration
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_FROM_NUMBER=+1234567890
TWILIO_TO_NUMBER=+1234567890
```

### Step 4: Install Twilio package
```bash
pip install twilio
```

---

## Option 2: Telegram Notifications

### Step 1: Create a Telegram Bot
1. Open Telegram and search for **@BotFather**
2. Send `/newbot` command
3. Follow prompts to name your bot
4. Copy the **Bot Token** (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Step 2: Get your Chat ID
1. Search for **@userinfobot** on Telegram
2. Start a chat and it will send you your **Chat ID** (looks like: `123456789`)

Alternatively, you can:
1. Send a message to your bot
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Look for `"chat":{"id":123456789}`

### Step 3: Add to .env file
```bash
# Telegram Configuration
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

---

## Testing Notifications

Run the test script to verify your setup:

```bash
python notification_service.py
```

This will send test notifications via all configured channels.

**Example output:**
```
Testing Notification Service...
======================================================================

SMS enabled: True
Telegram enabled: True

Sending test notifications...
✓ SMS sent (SID: SMxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)
✓ Telegram message sent
```

---

## Notification Examples

### Trade Entry (LONG)
**Telegram:**
```
🚀 TRADE OPENED - EURUSD

Direction: LONG
Entry Price: 1.08550
Position Size: $10,000.00
Stop Loss: 1.08116
Take Profit: 1.09635

Time: 2025-11-24 14:30:00
```

**SMS:**
```
TRADE OPENED: LONG EURUSD @ 1.08550 | Size: $10000
```

### Trade Exit (WIN)
**Telegram:**
```
📊 TRADE CLOSED - EURUSD - WIN ✅

Direction: LONG
Entry: 1.08550
Exit: 1.09200

P&L: +0.60% (+$60.00)
Reason: TAKE_PROFIT
Days Held: 3

Time: 2025-11-27 10:15:00
```

**SMS:**
```
TRADE CLOSED: LONG EURUSD | P&L: +0.60% (+$60) | TAKE_PROFIT
```

### Trade Exit (LOSS)
**Telegram:**
```
📊 TRADE CLOSED - GBPUSD - LOSS ❌

Direction: SHORT
Entry: 1.26400
Exit: 1.26900

P&L: -0.40% (-$40.00)
Reason: STOP_LOSS
Days Held: 1

Time: 2025-11-24 18:45:00
```

---

## Cost Considerations

### Twilio (SMS)
- **Free Trial**: $15 credit (≈500 messages)
- **After trial**: ~$0.0075-0.0079 per SMS in US
- **Estimated cost**: ~$0.50-1.00/month for typical forex trading (1-2 trades/week)

### Telegram
- **Cost**: 100% FREE forever
- **No limits** on messages
- **Recommended** if you want zero ongoing costs

---

## Troubleshooting

### No notifications received
1. Check `.env` file has correct credentials
2. Run `python notification_service.py` to test
3. Check console output for error messages

### SMS not working
- Verify phone numbers include country code (e.g., +1 for US)
- Confirm Twilio trial phone is verified
- Check Twilio console for error logs

### Telegram not working
- Ensure you've started a chat with your bot first
- Verify Chat ID is correct (no spaces or extra characters)
- Check bot token is complete and correct

---

## Optional: Disable Notifications

Simply remove or comment out the notification credentials in `.env`:

```bash
# TWILIO_ACCOUNT_SID=...
# TELEGRAM_BOT_TOKEN=...
```

The bot will continue to work normally without sending notifications.

---

## Security Note

⚠️ **Never commit your `.env` file to git!**

Your `.env` file contains sensitive credentials. Make sure it's listed in `.gitignore`:

```bash
# .gitignore
.env
```
