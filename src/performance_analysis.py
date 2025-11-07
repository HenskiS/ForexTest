"""
Performance Analysis and Visualization Module
Provides comprehensive analysis and visualization of backtest results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from datetime import datetime
import os


class PerformanceAnalyzer:
    """Analyzes and visualizes backtest performance"""

    def __init__(self, result, strategy_name="Strategy"):
        """
        Initialize analyzer

        Args:
            result: BacktestResult object
            strategy_name: Name of the strategy
        """
        self.result = result
        self.strategy_name = strategy_name
        self.metrics = result.metrics

    def print_summary(self):
        """Print performance summary to console"""
        print("\n" + "="*70)
        print(f"BACKTEST RESULTS: {self.strategy_name}")
        print("="*70)

        if self.metrics['total_trades'] == 0:
            print("\nNo trades executed.")
            return

        # Format metrics for display
        summary_data = [
            ["Total Trades", self.metrics['total_trades']],
            ["Winning Trades", f"{self.metrics['winning_trades']} ({self.metrics['win_rate']:.2f}%)"],
            ["Losing Trades", self.metrics['losing_trades']],
            ["", ""],
            ["Total Return", f"${self.metrics['total_return']:.2f} ({self.metrics['total_return_pct']:.2f}%)"],
            ["Final Capital", f"${self.metrics['final_capital']:.2f}"],
            ["Max Drawdown", f"{self.metrics['max_drawdown']:.2f}%"],
            ["", ""],
            ["Average Win", f"${self.metrics['avg_win']:.2f}"],
            ["Average Loss", f"${self.metrics['avg_loss']:.2f}"],
            ["Profit Factor", f"{self.metrics['profit_factor']:.2f}"],
            ["Sharpe Ratio", f"{self.metrics['sharpe_ratio']:.2f}"],
        ]

        print("\n" + tabulate(summary_data, tablefmt="simple"))
        print("="*70 + "\n")

    def plot_equity_curve(self, save_path=None):
        """Plot equity curve"""
        if self.result.equity_curve is None or self.result.equity_curve.empty:
            print("No equity data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Equity curve
        equity = self.result.equity_curve
        ax1.plot(equity['date'], equity['equity'], linewidth=2, label='Equity')
        ax1.fill_between(equity['date'], equity['equity'],
                         alpha=0.3)
        ax1.axhline(y=self.metrics['final_capital'], color='g',
                   linestyle='--', alpha=0.5, label='Final Capital')
        ax1.set_title(f'{self.strategy_name} - Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        equity['cummax'] = equity['equity'].cummax()
        equity['drawdown'] = (equity['equity'] - equity['cummax']) / equity['cummax'] * 100

        ax2.fill_between(equity['date'], equity['drawdown'], 0,
                        color='red', alpha=0.3)
        ax2.plot(equity['date'], equity['drawdown'], color='darkred', linewidth=1)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Equity curve saved to {save_path}")
        else:
            plt.savefig(os.path.join('results', f'{self.strategy_name}_equity_curve.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()

    def plot_trade_analysis(self, save_path=None):
        """Plot trade analysis"""
        if not self.result.trades:
            print("No trades to analyze")
            return

        df_trades = pd.DataFrame([vars(t) for t in self.result.trades])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. P&L Distribution
        ax1.hist(df_trades['pnl'], bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_title('P&L Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # 2. Cumulative P&L
        df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
        ax2.plot(range(len(df_trades)), df_trades['cumulative_pnl'],
                linewidth=2, color='green')
        ax2.fill_between(range(len(df_trades)), df_trades['cumulative_pnl'],
                         alpha=0.3, color='green')
        ax2.set_title('Cumulative P&L', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L ($)')
        ax2.grid(True, alpha=0.3)

        # 3. Win/Loss by Direction
        direction_pnl = df_trades.groupby('direction')['pnl'].agg(['sum', 'count', 'mean'])
        colors = ['green' if x > 0 else 'red' for x in direction_pnl['sum']]
        ax3.bar(direction_pnl.index, direction_pnl['sum'], color=colors, alpha=0.7)
        ax3.set_title('Total P&L by Direction', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Direction')
        ax3.set_ylabel('Total P&L ($)')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Win Rate Over Time (rolling)
        window = min(20, len(df_trades) // 5) if len(df_trades) > 20 else len(df_trades)
        df_trades['is_win'] = df_trades['pnl'] > 0
        df_trades['rolling_win_rate'] = df_trades['is_win'].rolling(window=window, min_periods=1).mean() * 100
        ax4.plot(range(len(df_trades)), df_trades['rolling_win_rate'],
                linewidth=2, color='blue')
        ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax4.set_title(f'Rolling Win Rate ({window}-trade window)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_ylim([0, 100])
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trade analysis saved to {save_path}")
        else:
            plt.savefig(os.path.join('results', f'{self.strategy_name}_trade_analysis.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()

    def get_monthly_returns(self):
        """Calculate monthly returns"""
        if not self.result.trades:
            return pd.DataFrame()

        df_trades = pd.DataFrame([vars(t) for t in self.result.trades])
        df_trades['month'] = pd.to_datetime(df_trades['exit_time']).dt.to_period('M')

        monthly = df_trades.groupby('month').agg({
            'pnl': 'sum',
            'entry_time': 'count'
        }).rename(columns={'entry_time': 'num_trades'})

        monthly['return_pct'] = (monthly['pnl'].cumsum() / 10000) * 100  # Assuming 10k start

        return monthly

    def export_trades(self, filename=None):
        """Export trades to CSV"""
        if not self.result.trades:
            print("No trades to export")
            return

        df_trades = pd.DataFrame([vars(t) for t in self.result.trades])

        if filename is None:
            filename = os.path.join('results', f'{self.strategy_name}_trades.csv')

        df_trades.to_csv(filename, index=False)
        print(f"Trades exported to {filename}")

    def generate_full_report(self):
        """Generate complete performance report"""
        self.print_summary()

        if self.result.trades:
            print("\nGenerating visualizations...")
            self.plot_equity_curve()
            self.plot_trade_analysis()
            self.export_trades()

            # Monthly returns
            monthly = self.get_monthly_returns()
            if not monthly.empty:
                print("\n" + "="*70)
                print("MONTHLY PERFORMANCE")
                print("="*70)
                print(monthly.to_string())
                print("="*70 + "\n")


class StrategyComparison:
    """Compare multiple strategy results"""

    def __init__(self, results_dict):
        """
        Initialize comparison

        Args:
            results_dict: Dictionary mapping strategy names to BacktestResult objects
        """
        self.results = results_dict

    def compare_metrics(self):
        """Compare key metrics across strategies"""
        comparison_data = []

        for name, result in self.results.items():
            metrics = result.metrics
            comparison_data.append([
                name,
                metrics['total_trades'],
                f"{metrics['win_rate']:.1f}%",
                f"{metrics['total_return_pct']:.2f}%",
                f"{metrics['max_drawdown']:.2f}%",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics['profit_factor']:.2f}"
            ])

        headers = ['Strategy', 'Trades', 'Win Rate', 'Return', 'Max DD', 'Sharpe', 'PF']
        print("\n" + "="*100)
        print("STRATEGY COMPARISON")
        print("="*100)
        print(tabulate(comparison_data, headers=headers, tablefmt='grid'))
        print("="*100 + "\n")

    def plot_comparison(self, save_path=None):
        """Plot comparison of equity curves"""
        fig, ax = plt.subplots(figsize=(14, 8))

        for name, result in self.results.items():
            if result.equity_curve is not None and not result.equity_curve.empty:
                equity = result.equity_curve
                ax.plot(equity['date'], equity['equity'], linewidth=2, label=name)

        ax.set_title('Strategy Comparison - Equity Curves', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join('results', 'strategy_comparison.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()
