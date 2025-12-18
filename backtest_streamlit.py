"""
Sleep Well Strategy Backtest Dashboard
Matches exact production mechanics: 8 pairs, 10/90 percentiles, 2.5% independent SL, 5-day hold
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Sleep Well Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants - MUST MATCH PRODUCTION
ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']
BUFFER_SIZE = 200
BUFFER_WARMUP = 50
LOWER_PCT = 10
UPPER_PCT = 90
HOLD_DAYS = 5
SL_PCT = 0.025  # 2.5% independent stop loss
ALLOC_PER_SLOT = 0.225  # 22.5% per slot
MAX_SLOTS = 5


@st.cache_data(ttl=3600)
def load_all_data():
    """Load all prediction and price data"""
    pair_data = {}

    for pair in ALL_PAIRS:
        try:
            with open(f'optimized_ann_predictions/predictions_{pair}.pkl', 'rb') as f:
                data = pickle.load(f)

            df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # Load spread data
            try:
                spread_df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
                spread_df['date'] = pd.to_datetime(spread_df['date'])
                spread_df = spread_df.set_index('date')
                df = df.join(spread_df[['spread_pct']], how='left')
            except FileNotFoundError:
                df['spread_pct'] = 0.00025

            df['spread_pct'] = df['spread_pct'].fillna(0.00025)

            pair_data[pair] = {
                'predictions': data['predictions'],
                'test_indices': data['test_indices'],
                'df': df
            }
        except Exception as e:
            st.warning(f"Could not load {pair}: {e}")

    return pair_data


def run_backtest(pair_data, start_date=None, end_date=None, leverage=2.0):
    """
    Run backtest with EXACT production mechanics:
    - 10/90 percentile thresholds
    - 2.5% INDEPENDENT stop loss (each trade has its own stop)
    - 5-day hold period
    - Up to 5 slots per pair in same direction
    """
    all_trades = []
    daily_pnl = {}

    for pair in ALL_PAIRS:
        if pair not in pair_data:
            continue

        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']

        prediction_buffer = []
        slots = []  # List of {entry_price, entry_idx, entry_date, direction, size}

        for idx, (pred_idx, prediction) in enumerate(zip(test_indices, predictions)):
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer = prediction_buffer[-BUFFER_SIZE:]

            if len(prediction_buffer) < BUFFER_WARMUP:
                continue

            current_idx = pred_idx + 1
            if current_idx >= len(df):
                continue

            current_date = df.index[current_idx]

            # Filter by date range (handle timezone)
            current_date_naive = current_date.tz_localize(None) if current_date.tzinfo else current_date
            if start_date and current_date_naive < pd.Timestamp(start_date):
                continue
            if end_date and current_date_naive > pd.Timestamp(end_date):
                continue

            # Skip weekends
            if current_date.dayofweek >= 5:
                continue

            current_row = df.iloc[current_idx]
            spread = current_row['spread_pct']

            # Check INDEPENDENT stop losses FIRST (each slot has its own stop)
            # This must be checked before time exits to match test_dca_stop_fixed.py
            if slots:
                direction = slots[0]['direction']
                still_open = []
                for slot in slots:
                    # Each slot has its own stop at 2.5% from its entry
                    if direction == 1:
                        sl_price = slot['entry_price'] * (1 - SL_PCT)
                        sl_hit = current_row['low'] <= sl_price
                    else:
                        sl_price = slot['entry_price'] * (1 + SL_PCT)
                        sl_hit = current_row['high'] >= sl_price

                    if sl_hit:
                        # Only THIS slot exits
                        if direction == 1:
                            exit_price = sl_price * (1 - spread)
                            pnl = (exit_price / slot['entry_price']) - 1
                        else:
                            exit_price = sl_price * (1 + spread)
                            pnl = (slot['entry_price'] / exit_price) - 1

                        all_trades.append({
                            'pair': pair,
                            'entry_date': slot['entry_date'],
                            'exit_date': current_date,
                            'direction': direction,
                            'pnl': pnl,
                            'exit_type': 'SL'
                        })

                        if slot['entry_date'] not in daily_pnl:
                            daily_pnl[slot['entry_date']] = 0.0
                        daily_pnl[slot['entry_date']] += ALLOC_PER_SLOT * pnl
                    else:
                        still_open.append(slot)

                slots = still_open

            # Process time exits AFTER stop losses
            remaining_slots = []
            for slot in slots:
                days_held = current_idx - slot['entry_idx']

                if days_held >= HOLD_DAYS:
                    # Time exit at open
                    direction = slot['direction']
                    if direction == 1:
                        exit_price = current_row['open'] * (1 - spread)
                        pnl = (exit_price / slot['entry_price']) - 1
                    else:
                        exit_price = current_row['open'] * (1 + spread)
                        pnl = (slot['entry_price'] / exit_price) - 1

                    all_trades.append({
                        'pair': pair,
                        'entry_date': slot['entry_date'],
                        'exit_date': current_date,
                        'direction': direction,
                        'pnl': pnl,
                        'exit_type': 'TIME'
                    })

                    # Add to daily PnL on entry date
                    if slot['entry_date'] not in daily_pnl:
                        daily_pnl[slot['entry_date']] = 0.0
                    daily_pnl[slot['entry_date']] += ALLOC_PER_SLOT * pnl
                else:
                    remaining_slots.append(slot)

            slots = remaining_slots

            # Check for new signal
            lower_thresh = np.percentile(prediction_buffer, LOWER_PCT)
            upper_thresh = np.percentile(prediction_buffer, UPPER_PCT)

            if prediction >= upper_thresh:
                signal = 1
            elif prediction <= lower_thresh:
                signal = -1
            else:
                signal = 0

            if signal != 0:
                # Check if we can add this position
                can_add = True
                if slots and slots[0]['direction'] != signal:
                    can_add = False  # Can't add opposite direction
                if len(slots) >= MAX_SLOTS:
                    can_add = False  # Max slots reached

                if can_add:
                    entry_price = current_row['open']
                    if signal == 1:
                        actual_entry = entry_price * (1 + spread)
                    else:
                        actual_entry = entry_price * (1 - spread)

                    # Check if THIS slot's independent stop would be hit on entry day
                    if signal == 1:
                        sl_price = actual_entry * (1 - SL_PCT)
                        entry_day_sl_hit = current_row['low'] <= sl_price
                    else:
                        sl_price = actual_entry * (1 + SL_PCT)
                        entry_day_sl_hit = current_row['high'] >= sl_price

                    if entry_day_sl_hit:
                        # Only THIS new slot stops out - existing slots unaffected
                        if signal == 1:
                            exit_price = sl_price * (1 - spread)
                            pnl = (exit_price / actual_entry) - 1
                        else:
                            exit_price = sl_price * (1 + spread)
                            pnl = (actual_entry / exit_price) - 1

                        all_trades.append({
                            'pair': pair,
                            'entry_date': current_date,
                            'exit_date': current_date,
                            'direction': signal,
                            'pnl': pnl,
                            'exit_type': 'SL_DAY0'
                        })

                        if current_date not in daily_pnl:
                            daily_pnl[current_date] = 0.0
                        daily_pnl[current_date] += ALLOC_PER_SLOT * pnl
                        # Existing slots continue running (independent stops)
                    else:
                        # Add new slot
                        slots.append({
                            'entry_price': actual_entry,
                            'entry_idx': current_idx,
                            'entry_date': current_date,
                            'direction': signal,
                            'size': 1
                        })

    return all_trades, daily_pnl


def calc_stats(trades, daily_pnl, leverage, starting_equity=1000):
    """Calculate performance statistics"""
    if not trades or not daily_pnl:
        return None

    # Sort daily returns
    returns = pd.Series(daily_pnl).sort_index()
    leveraged = returns * leverage

    # Build equity curve
    equity = [starting_equity]
    for r in leveraged:
        equity.append(equity[-1] * (1 + r))

    equity_arr = np.array(equity[1:])

    # Total return
    total_return = (equity_arr[-1] / starting_equity - 1) * 100

    # Annualized return (calendar days)
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0

    # Max drawdown
    cummax = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - cummax) / cummax * 100
    max_dd = drawdowns.min()
    max_dd_idx = drawdowns.argmin()
    max_dd_date = returns.index[max_dd_idx]

    # Sharpe ratio
    sharpe = (leveraged.mean() / leveraged.std()) * np.sqrt(252) if leveraged.std() > 0 else 0

    # Win rate
    wins = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = wins / len(trades) * 100 if trades else 0

    # Avg win/loss
    winning_trades = [t['pnl'] for t in trades if t['pnl'] > 0]
    losing_trades = [t['pnl'] for t in trades if t['pnl'] <= 0]
    avg_win = np.mean(winning_trades) * 100 if winning_trades else 0
    avg_loss = np.mean(losing_trades) * 100 if losing_trades else 0

    # Exit type breakdown
    sl_exits = sum(1 for t in trades if 'SL' in t['exit_type'])
    time_exits = sum(1 for t in trades if t['exit_type'] == 'TIME')

    # Daily stats
    winning_days = (leveraged > 0).sum()
    losing_days = (leveraged < 0).sum()

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_dd': max_dd,
        'max_dd_date': max_dd_date,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_trades': len(trades),
        'sl_exits': sl_exits,
        'time_exits': time_exits,
        'winning_days': winning_days,
        'losing_days': losing_days,
        'equity_curve': equity_arr,
        'dates': returns.index.tolist(),
        'daily_returns': leveraged.values,
        'years': years
    }


# Main app
st.title("📈 Sleep Well Strategy Backtest")
st.caption("8 Pairs | 10/90 Percentiles | 2.5% Independent SL | 5-Day Hold")

# Load data
with st.spinner("Loading prediction data..."):
    pair_data = load_all_data()

if not pair_data:
    st.error("No data loaded. Check prediction files.")
    st.stop()

# Get date range from data
sample_pair = list(pair_data.keys())[0]
sample_df = pair_data[sample_pair]['df']
test_indices = pair_data[sample_pair]['test_indices']
min_date = sample_df.index[test_indices[0]].date()
max_date = sample_df.index[test_indices[-1]].date()

# Sidebar controls
st.sidebar.header("Parameters")

# Date range
st.sidebar.subheader("Date Range")
date_presets = {
    "Last 1 Year": 365,
    "Last 2 Years": 730,
    "Last 3 Years": 1095,
    "Last 5 Years": 1825,
    "All Data": None,
    "Custom": "custom"
}

preset = st.sidebar.selectbox("Period", list(date_presets.keys()), index=1)

if date_presets[preset] == "custom":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("From", value=max_date - timedelta(days=730), min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)
elif date_presets[preset] is None:
    start_date = min_date
    end_date = max_date
else:
    end_date = max_date
    start_date = max_date - timedelta(days=date_presets[preset])

# Leverage
st.sidebar.subheader("Risk Settings")
leverage = st.sidebar.slider("Leverage", 0.5, 5.0, 2.0, 0.5, format="%.1fx")
starting_equity = st.sidebar.number_input("Starting Equity ($)", 100, 100000, 1000, 100)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Strategy Config:**
- Pairs: {len(ALL_PAIRS)}
- Thresholds: {LOWER_PCT}/{UPPER_PCT} percentile
- Stop Loss: {SL_PCT*100:.1f}% (independent)
- Hold: {HOLD_DAYS} days
- Allocation: {ALLOC_PER_SLOT*100:.1f}% per slot
""")

# Run backtest
with st.spinner("Running backtest..."):
    trades, daily_pnl = run_backtest(pair_data, start_date, end_date, leverage)
    stats = calc_stats(trades, daily_pnl, leverage, starting_equity)

if stats is None:
    st.warning("No trades in selected period.")
    st.stop()

# Display metrics
st.markdown(f"**Period:** {stats['dates'][0].date()} to {stats['dates'][-1].date()} ({stats['years']:.1f} years)")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Annual Return", f"{stats['annual_return']:.1f}%")
with col2:
    st.metric("Max Drawdown", f"{stats['max_dd']:.1f}%")
with col3:
    st.metric("Sharpe Ratio", f"{stats['sharpe']:.2f}")
with col4:
    st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
with col5:
    st.metric("Total Trades", f"{stats['total_trades']:,}")

# Charts
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Equity Curve", "Daily P&L"),
    vertical_spacing=0.12,
    row_heights=[0.6, 0.4]
)

# Equity curve
fig.add_trace(
    go.Scatter(
        x=stats['dates'],
        y=stats['equity_curve'],
        mode='lines',
        name='Equity',
        line=dict(color='#00ff41', width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Equity: $%{y:,.2f}<extra></extra>"
    ),
    row=1, col=1
)

# Max DD marker
fig.add_trace(
    go.Scatter(
        x=[stats['max_dd_date']],
        y=[stats['equity_curve'][stats['dates'].index(stats['max_dd_date'])]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='x'),
        name=f'Max DD: {stats["max_dd"]:.1f}%',
        hovertemplate=f"Max Drawdown: {stats['max_dd']:.1f}%<extra></extra>"
    ),
    row=1, col=1
)

# Daily P&L bars
colors = ['green' if r > 0 else 'red' for r in stats['daily_returns']]
fig.add_trace(
    go.Bar(
        x=stats['dates'],
        y=stats['daily_returns'] * starting_equity,
        marker_color=colors,
        name='Daily P&L',
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>P&L: $%{y:.2f}<extra></extra>"
    ),
    row=2, col=1
)

fig.update_layout(
    height=700,
    showlegend=False,
    hovermode='x unified',
    plot_bgcolor='#0d0208',
    paper_bgcolor='#0d0208',
    font=dict(color='#00ff41')
)

fig.update_xaxes(gridcolor='#1a1a1a', zerolinecolor='#1a1a1a')
fig.update_yaxes(gridcolor='#1a1a1a', zerolinecolor='#1a1a1a')
fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
fig.update_yaxes(title_text="P&L ($)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# Detailed stats
with st.expander("Detailed Statistics"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Performance**")
        st.write(f"Total Return: {stats['total_return']:.1f}%")
        st.write(f"Annual Return: {stats['annual_return']:.1f}%")
        st.write(f"Max Drawdown: {stats['max_dd']:.1f}%")
        st.write(f"Max DD Date: {stats['max_dd_date'].date()}")
        st.write(f"Sharpe Ratio: {stats['sharpe']:.2f}")

    with col2:
        st.markdown("**Trades**")
        st.write(f"Total Trades: {stats['total_trades']:,}")
        st.write(f"Win Rate: {stats['win_rate']:.1f}%")
        st.write(f"Avg Win: +{stats['avg_win']:.2f}%")
        st.write(f"Avg Loss: {stats['avg_loss']:.2f}%")
        st.write(f"Profit Factor: {abs(stats['avg_win']/stats['avg_loss']) if stats['avg_loss'] != 0 else 0:.2f}")

    with col3:
        st.markdown("**Exit Types**")
        st.write(f"Time Exits: {stats['time_exits']} ({100*stats['time_exits']/stats['total_trades']:.0f}%)")
        st.write(f"Stop Loss Exits: {stats['sl_exits']} ({100*stats['sl_exits']/stats['total_trades']:.0f}%)")
        st.write(f"Winning Days: {stats['winning_days']}")
        st.write(f"Losing Days: {stats['losing_days']}")

# Yearly breakdown
with st.expander("Yearly Performance"):
    trades_df = pd.DataFrame(trades)
    trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year

    yearly_data = []
    for year in sorted(trades_df['year'].unique()):
        year_trades = trades_df[trades_df['year'] == year]
        year_pnl = {k: v for k, v in daily_pnl.items() if k.year == year}

        if year_pnl:
            returns = pd.Series(year_pnl).sort_index() * leverage
            equity = [1000]
            for r in returns:
                equity.append(equity[-1] * (1 + r))

            year_return = (equity[-1] / 1000 - 1) * 100
            cummax = np.maximum.accumulate(equity)
            max_dd = ((np.array(equity) - cummax) / cummax).min() * 100
            win_rate = (year_trades['pnl'] > 0).mean() * 100
            sl_exits = sum('SL' in t for t in year_trades['exit_type'])

            yearly_data.append({
                'Year': year,
                'Return': f"{year_return:.1f}%",
                'Max DD': f"{max_dd:.1f}%",
                'Win Rate': f"{win_rate:.1f}%",
                'Trades': len(year_trades),
                'SL Exits': sl_exits
            })

    st.dataframe(pd.DataFrame(yearly_data), use_container_width=True, hide_index=True)

# Per-pair breakdown
with st.expander("Per-Pair Performance"):
    trades_df = pd.DataFrame(trades)

    pair_data_list = []
    for pair in ALL_PAIRS:
        pair_trades = trades_df[trades_df['pair'] == pair]
        if len(pair_trades) > 0:
            win_rate = (pair_trades['pnl'] > 0).mean() * 100
            avg_pnl = pair_trades['pnl'].mean() * 100
            sl_exits = sum('SL' in t for t in pair_trades['exit_type'])

            pair_data_list.append({
                'Pair': pair,
                'Trades': len(pair_trades),
                'Win Rate': f"{win_rate:.1f}%",
                'Avg P&L': f"{avg_pnl:.3f}%",
                'SL Exits': sl_exits
            })

    st.dataframe(pd.DataFrame(pair_data_list), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Backtest uses exact production mechanics: 8 pairs, 10/90 percentiles, 2.5% independent SL per trade, 5-day hold, 22.5% allocation per slot")
