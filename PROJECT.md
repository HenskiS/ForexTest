# LSTM Forex Trading Study

## Goal
Recreate the LSTM-based forex trading strategy from *"Predictive modeling of foreign exchange trading signals using machine learning techniques"* (Enkhbayar & Ślepaczuk, Expert Systems With Applications, 2025).

## Progress

### ✅ Step 1: Data Loading & Exploration (Complete)
- Loaded EURUSD daily data (6,743 observations, Jan 2000 - Nov 2025)
- Matches study's full timeframe (2000-2023) + additional recent data
- Calculated simple and log returns
- Compared statistics with study's Table 1
- Confirmed non-normal distribution (JB test)

### 🔄 Next Steps
- Step 2: Feature Engineering (technical indicators: EMA, MACD, RSI, etc.)
- Step 3: Target variable preparation
- Step 4: Walk-forward optimization
- Step 5: LSTM model training
- Step 6: Signal generation
- Step 7: Backtesting with transaction costs
- Step 8: Performance evaluation

## Key Details
- **Data**: EURUSD daily, 6,743 observations (2000-2025) — exceeds study's 6,171 obs
- **Framework**: Keras 3.0 + PyTorch backend
- **Notebook**: forex_lstm_study.ipynb
