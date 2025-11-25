"""
Analyze backtest performance metrics including drawdown, Sharpe ratio, etc.
"""
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='Backtest trades CSV file')
args = parser.parse_args()

# Load trades
trades_df = pd.read_csv(args.file)
trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

print(f"\n{'='*80}")
print(f"BACKTEST PERFORMANCE ANALYSIS")
print(f"{'='*80}")
print(f"File: {args.file}")
print(f"Total Trades: {len(trades_df)}")
print(f"Date Range: {trades_df['entry_date'].min().date()} to {trades_df['exit_date'].max().date()}")

# Calculate equity curve
starting_equity = 1000
equity_curve = [starting_equity]
equity_dates = [trades_df['entry_date'].iloc[0]]

for _, trade in trades_df.iterrows():
    current_equity = equity_curve[-1]
    new_equity = current_equity * (1 + trade['net_return_pct'] / 100)
    equity_curve.append(new_equity)
    equity_dates.append(trade['exit_date'])

equity_df = pd.DataFrame({
    'date': equity_dates,
    'equity': equity_curve
})

# Calculate drawdown
equity_df['peak'] = equity_df['equity'].cummax()
equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100

max_drawdown = equity_df['drawdown'].min()
max_dd_date = equity_df.loc[equity_df['drawdown'].idxmin(), 'date']

# Calculate longest drawdown period
equity_df['is_peak'] = equity_df['equity'] == equity_df['peak']
equity_df['peak_number'] = equity_df['is_peak'].cumsum()

dd_periods = []
for peak_num in equity_df['peak_number'].unique():
    period_data = equity_df[equity_df['peak_number'] == peak_num]
    if len(period_data) > 1:
        dd_periods.append({
            'days': (period_data['date'].max() - period_data['date'].min()).days,
            'start': period_data['date'].min(),
            'end': period_data['date'].max()
        })

if dd_periods:
    longest_dd = max(dd_periods, key=lambda x: x['days'])
else:
    longest_dd = {'days': 0, 'start': None, 'end': None}

# Calculate returns statistics
total_return_pct = (equity_curve[-1] - starting_equity) / starting_equity * 100
years = (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25
annual_return_pct = ((equity_curve[-1] / starting_equity) ** (1 / years) - 1) * 100

# Win/Loss statistics
wins = trades_df[trades_df['outcome'] == 'WIN']
losses = trades_df[trades_df['outcome'] == 'LOSS']
win_rate = len(wins) / len(trades_df) * 100

avg_win = wins['net_return_pct'].mean() if len(wins) > 0 else 0
avg_loss = losses['net_return_pct'].mean() if len(losses) > 0 else 0
largest_win = wins['net_return_pct'].max() if len(wins) > 0 else 0
largest_loss = losses['net_return_pct'].min() if len(losses) > 0 else 0

# Profit factor
total_profit = wins['net_return_pct'].sum() if len(wins) > 0 else 0
total_loss = abs(losses['net_return_pct'].sum()) if len(losses) > 0 else 1
profit_factor = total_profit / total_loss if total_loss > 0 else 0

# Sharpe ratio (annualized, assuming risk-free rate = 0)
returns = trades_df['net_return_pct'].values
avg_return = returns.mean()
std_return = returns.std()

# Estimate trading days per year based on actual trades
total_days = (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days
trades_per_year = len(trades_df) / years

if std_return > 0:
    sharpe_ratio = (avg_return * trades_per_year) / (std_return * np.sqrt(trades_per_year))
else:
    sharpe_ratio = 0

# Calmar ratio (annual return / max drawdown)
calmar_ratio = abs(annual_return_pct / max_drawdown) if max_drawdown != 0 else 0

# Consecutive wins/losses
consecutive_wins = 0
consecutive_losses = 0
max_consecutive_wins = 0
max_consecutive_losses = 0

for outcome in trades_df['outcome']:
    if outcome == 'WIN':
        consecutive_wins += 1
        consecutive_losses = 0
        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
    else:
        consecutive_losses += 1
        consecutive_wins = 0
        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

# Print results
print(f"\n{'='*80}")
print(f"RETURNS")
print(f"{'='*80}")
print(f"Starting Equity: ${starting_equity:,.2f}")
print(f"Final Equity: ${equity_curve[-1]:,.2f}")
print(f"Total Return: {total_return_pct:.2f}%")
print(f"Annual Return: {annual_return_pct:.2f}%")
print(f"Years: {years:.2f}")

print(f"\n{'='*80}")
print(f"RISK METRICS")
print(f"{'='*80}")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print(f"Max DD Date: {max_dd_date.date()}")
print(f"Longest Drawdown: {longest_dd['days']} days")
if longest_dd['start']:
    print(f"  From: {longest_dd['start'].date()} to {longest_dd['end'].date()}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Calmar Ratio: {calmar_ratio:.2f}")

print(f"\n{'='*80}")
print(f"TRADE STATISTICS")
print(f"{'='*80}")
print(f"Total Trades: {len(trades_df)}")
print(f"Wins: {len(wins)} ({win_rate:.1f}%)")
print(f"Losses: {len(losses)} ({100-win_rate:.1f}%)")
print(f"")
print(f"Average Win: {avg_win:.2f}%")
print(f"Average Loss: {avg_loss:.2f}%")
print(f"Largest Win: {largest_win:.2f}%")
print(f"Largest Loss: {largest_loss:.2f}%")
print(f"")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "Win/Loss Ratio: N/A")
print(f"")
print(f"Max Consecutive Wins: {max_consecutive_wins}")
print(f"Max Consecutive Losses: {max_consecutive_losses}")

print(f"\n{'='*80}")
print(f"RISK-ADJUSTED RETURNS")
print(f"{'='*80}")
print(f"Return/Drawdown: {abs(annual_return_pct/max_drawdown):.2f}x" if max_drawdown != 0 else "Return/Drawdown: N/A")
print(f"Avg Trade Return: {avg_return:.2f}%")
print(f"Std Dev Trade Return: {std_return:.2f}%")
print(f"Trades per Year: {trades_per_year:.0f}")

# Save equity curve
equity_output = args.file.replace('_trades_', '_equity_')
equity_df[['date', 'equity', 'drawdown']].to_csv(equity_output, index=False)
print(f"\nEquity curve saved to: {equity_output}")
