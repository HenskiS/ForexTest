"""
Analyze backtest performance by year.
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

print(f"\n{'='*120}")
print(f"YEARLY PERFORMANCE ANALYSIS")
print(f"{'='*120}")
print(f"File: {args.file}")
print(f"Total Trades: {len(trades_df)}")
print(f"Date Range: {trades_df['entry_date'].min().date()} to {trades_df['exit_date'].max().date()}")

# Add year column based on exit date
trades_df['year'] = trades_df['exit_date'].dt.year

# Calculate equity curve for drawdown calculation
starting_equity = 1000
equity_curve = [starting_equity]
equity_dates = [trades_df['entry_date'].iloc[0]]
equity_years = [trades_df['entry_date'].iloc[0].year]

for _, trade in trades_df.iterrows():
    current_equity = equity_curve[-1]
    new_equity = current_equity * (1 + trade['net_return_pct'] / 100)
    equity_curve.append(new_equity)
    equity_dates.append(trade['exit_date'])
    equity_years.append(trade['exit_date'].year)

equity_df = pd.DataFrame({
    'date': equity_dates,
    'equity': equity_curve,
    'year': equity_years
})

# Calculate cumulative peak for drawdown
equity_df['peak'] = equity_df['equity'].cummax()
equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100

# Group by year and calculate metrics
yearly_results = []

for year in sorted(trades_df['year'].unique()):
    year_trades = trades_df[trades_df['year'] == year].copy()
    year_equity = equity_df[equity_df['year'] == year].copy()

    if len(year_trades) == 0:
        continue

    # Get starting and ending equity for the year
    year_start_idx = equity_df[equity_df['year'] == year].index[0]
    year_end_idx = equity_df[equity_df['year'] == year].index[-1]

    if year_start_idx > 0:
        start_equity = equity_df.loc[year_start_idx - 1, 'equity']
    else:
        start_equity = starting_equity

    end_equity = equity_df.loc[year_end_idx, 'equity']

    # Calculate year return
    year_return = (end_equity - start_equity) / start_equity * 100

    # Win/Loss stats
    wins = year_trades[year_trades['outcome'] == 'WIN']
    losses = year_trades[year_trades['outcome'] == 'LOSS']
    win_rate = len(wins) / len(year_trades) * 100 if len(year_trades) > 0 else 0

    avg_win = wins['net_return_pct'].mean() if len(wins) > 0 else 0
    avg_loss = losses['net_return_pct'].mean() if len(losses) > 0 else 0

    # Sharpe ratio for the year
    returns = year_trades['net_return_pct'].values
    avg_return = returns.mean()
    std_return = returns.std()

    trades_in_year = len(year_trades)
    if std_return > 0 and trades_in_year > 1:
        sharpe_ratio = (avg_return * trades_in_year) / (std_return * np.sqrt(trades_in_year))
    else:
        sharpe_ratio = 0

    # Max drawdown in the year
    max_dd = year_equity['drawdown'].min()

    yearly_results.append({
        'Year': year,
        'Return (%)': year_return,
        'Trades': len(year_trades),
        'Win Rate (%)': win_rate,
        'Avg Win (%)': avg_win,
        'Avg Loss (%)': avg_loss,
        'Sharpe': sharpe_ratio,
        'Max DD (%)': max_dd,
        'End Equity': end_equity
    })

# Create results dataframe
results_df = pd.DataFrame(yearly_results)

# Display results
print(f"\n{'='*120}")
print("YEAR-BY-YEAR PERFORMANCE")
print(f"{'='*120}")
print(results_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

# Summary statistics
print(f"\n{'='*120}")
print("SUMMARY STATISTICS")
print(f"{'='*120}")
print(f"Best Year: {results_df.loc[results_df['Return (%)'].idxmax(), 'Year']:.0f} ({results_df['Return (%)'].max():.2f}%)")
print(f"Worst Year: {results_df.loc[results_df['Return (%)'].idxmin(), 'Year']:.0f} ({results_df['Return (%)'].min():.2f}%)")
print(f"Avg Annual Return: {results_df['Return (%)'].mean():.2f}%")
print(f"Median Annual Return: {results_df['Return (%)'].median():.2f}%")
print(f"Std Dev Annual Return: {results_df['Return (%)'].std():.2f}%")
print(f"")
print(f"Avg Win Rate: {results_df['Win Rate (%)'].mean():.2f}%")
print(f"Avg Sharpe Ratio: {results_df['Sharpe'].mean():.2f}")
print(f"Worst Max DD: {results_df['Max DD (%)'].min():.2f}%")
print(f"")
print(f"Positive Years: {len(results_df[results_df['Return (%)'] > 0])}/{len(results_df)} ({len(results_df[results_df['Return (%)'] > 0])/len(results_df)*100:.1f}%)")

# Save results
output_file = args.file.replace('_trades_', '_yearly_')
results_df.to_csv(output_file, index=False)
print(f"\nYearly results saved to: {output_file}")
