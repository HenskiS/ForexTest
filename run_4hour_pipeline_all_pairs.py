"""
Complete 4-hour pipeline for multiple currency pairs.

This script:
1. Adds features to raw 4-hour data
2. Trains XGBoost models with walk-forward validation
3. Backtests with advanced exits
4. Generates comparison report

Usage:
    python run_4hour_pipeline_all_pairs.py
    python run_4hour_pipeline_all_pairs.py --pairs EURUSD GBPUSD USDJPY
    python run_4hour_pipeline_all_pairs.py --skip-training  # Skip if models exist
"""

import sys
import subprocess
import argparse
import json
import pandas as pd
from datetime import datetime

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pairs', nargs='+', default=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
                    help='Currency pairs to process')
parser.add_argument('--skip-features', action='store_true',
                    help='Skip feature engineering if already done')
parser.add_argument('--skip-training', action='store_true',
                    help='Skip training if models already exist')
parser.add_argument('--n-iter', type=int, default=20,
                    help='Number of hyperparameter iterations for training')
args = parser.parse_args()

PAIRS = [p.upper() for p in args.pairs]
SKIP_FEATURES = args.skip_features
SKIP_TRAINING = args.skip_training
N_ITER = args.n_iter

print("="*80)
print("4-HOUR MULTI-PAIR PIPELINE")
print("="*80)
print(f"Pairs: {', '.join(PAIRS)}")
print(f"Skip features: {SKIP_FEATURES}")
print(f"Skip training: {SKIP_TRAINING}")
print(f"Training iterations: {N_ITER}")
print("="*80)


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}")
    print()

    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nERROR: Command failed with exit code {result.returncode}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    return result.returncode == 0


# Track results
pipeline_results = {
    'pairs': {},
    'timestamp': datetime.now().isoformat()
}

for pair in PAIRS:
    print(f"\n{'#'*80}")
    print(f"# PROCESSING {pair}")
    print(f"{'#'*80}")

    pair_result = {
        'pair': pair,
        'features_added': False,
        'model_trained': False,
        'backtest_completed': False
    }

    # Step 1: Add features
    if not SKIP_FEATURES:
        success = run_command(
            f'python add_features_4hour.py {pair}',
            f'[1/3] Adding features to {pair} 4-hour data'
        )
        pair_result['features_added'] = success

        if not success:
            print(f"WARNING: Failed to add features for {pair}, skipping...")
            pipeline_results['pairs'][pair] = pair_result
            continue
    else:
        print(f"\n[1/3] Skipping feature engineering for {pair} (--skip-features)")
        pair_result['features_added'] = True

    # Step 2: Train model
    if not SKIP_TRAINING:
        success = run_command(
            f'python train_xgboost_4hour.py --pair {pair} --n_iter {N_ITER}',
            f'[2/3] Training XGBoost model for {pair}'
        )
        pair_result['model_trained'] = success

        if not success:
            print(f"WARNING: Failed to train model for {pair}, skipping backtest...")
            pipeline_results['pairs'][pair] = pair_result
            continue
    else:
        print(f"\n[2/3] Skipping training for {pair} (--skip-training)")
        pair_result['model_trained'] = True

    # Step 3: Backtest
    success = run_command(
        f'python backtest_advanced_exits_4hour.py --pair {pair} --target target_1day_return',
        f'[3/3] Backtesting {pair} with advanced exits'
    )
    pair_result['backtest_completed'] = success

    if not success:
        print(f"WARNING: Backtest failed for {pair}")

    pipeline_results['pairs'][pair] = pair_result

# Generate comparison report
print(f"\n{'='*80}")
print("GENERATING MULTI-PAIR COMPARISON REPORT")
print(f"{'='*80}")

successful_pairs = [p for p, r in pipeline_results['pairs'].items() if r['backtest_completed']]

if len(successful_pairs) == 0:
    print("\nERROR: No pairs completed successfully")
    sys.exit(1)

print(f"\nSuccessful pairs: {', '.join(successful_pairs)}")

# Load all backtest results
all_results = {}
for pair in successful_pairs:
    result_file = f'advanced_exits_4hour_{pair}_target_1day_return_results.json'
    try:
        with open(result_file, 'r') as f:
            all_results[pair] = json.load(f)
        print(f"  ✓ Loaded {pair} results: {len(all_results[pair])} configs")
    except FileNotFoundError:
        print(f"  ✗ Missing {pair} results file: {result_file}")

# Create comparison table
print(f"\n{'='*80}")
print("MULTI-PAIR COMPARISON - BEST CONFIGURATIONS")
print(f"{'='*80}\n")

comparison_data = []
for pair, results in all_results.items():
    # Find best config by Sharpe ratio
    best = max(results, key=lambda x: x['sharpe_ratio'])
    comparison_data.append({
        'Pair': pair,
        'Config': best['config'][:40],  # Truncate for display
        'Annual Return': best['annualized_return'],
        'Monthly Return': best['annualized_return'] / 12,
        'Sharpe': best['sharpe_ratio'],
        'Max DD': best['max_drawdown'],
        'Win Rate': best['win_rate'],
        'Trades/Yr': best['trades_per_year'],
        'Final Capital': best['final_capital']
    })

# Create DataFrame for nice formatting
df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values('Sharpe', ascending=False)

print(df_comparison.to_string(index=False))

# Calculate portfolio metrics (equal weight)
print(f"\n{'='*80}")
print("PORTFOLIO METRICS (Equal Weight)")
print(f"{'='*80}")

total_capital = sum(r['Final Capital'] for r in comparison_data)
avg_annual = sum(r['Annual Return'] for r in comparison_data) / len(comparison_data)
avg_monthly = avg_annual / 12
avg_sharpe = sum(r['Sharpe'] for r in comparison_data) / len(comparison_data)
worst_dd = min(r['Max DD'] for r in comparison_data)

print(f"\nStarting Capital:      ${1000 * len(successful_pairs):,.0f} (${1000} per pair)")
print(f"Final Capital:         ${total_capital:,.2f}")
print(f"Total Return:          {((total_capital / (1000 * len(successful_pairs))) - 1) * 100:.2f}%")
print(f"Avg Annual Return:     {avg_annual * 100:.2f}%")
print(f"Avg Monthly Return:    {avg_monthly * 100:.2f}%")
print(f"Avg Sharpe Ratio:      {avg_sharpe:.3f}")
print(f"Worst Max Drawdown:    {worst_dd * 100:.2f}%")

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'pairs_processed': successful_pairs,
    'best_configs': comparison_data,
    'portfolio_metrics': {
        'starting_capital': 1000 * len(successful_pairs),
        'final_capital': total_capital,
        'total_return': ((total_capital / (1000 * len(successful_pairs))) - 1),
        'avg_annual_return': avg_annual,
        'avg_monthly_return': avg_monthly,
        'avg_sharpe_ratio': avg_sharpe,
        'worst_max_drawdown': worst_dd
    }
}

summary_file = '4hour_multi_pair_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n{'='*80}")
print(f"Summary saved to: {summary_file}")
print(f"{'='*80}")

# Display individual pair details
print(f"\n{'='*80}")
print("INDIVIDUAL PAIR DETAILS")
print(f"{'='*80}")

for pair_data in comparison_data:
    print(f"\n{pair_data['Pair']}:")
    print(f"  Config:          {pair_data['Config']}")
    print(f"  Annual Return:   {pair_data['Annual Return'] * 100:6.2f}%")
    print(f"  Monthly Return:  {pair_data['Monthly Return'] * 100:6.2f}%")
    print(f"  Sharpe Ratio:    {pair_data['Sharpe']:6.3f}")
    print(f"  Max Drawdown:    {pair_data['Max DD'] * 100:6.2f}%")
    print(f"  Win Rate:        {pair_data['Win Rate'] * 100:6.2f}%")
    print(f"  Trades/Year:     {pair_data['Trades/Yr']:6.1f}")
    print(f"  $1000 → ${pair_data['Final Capital']:,.2f}")

print(f"\n{'='*80}")
print("PIPELINE COMPLETE!")
print(f"{'='*80}")
print(f"\nResults:")
print(f"  Pairs processed: {len(successful_pairs)}/{len(PAIRS)}")
print(f"  Avg monthly return: {avg_monthly * 100:.2f}%")
print(f"  Portfolio final capital: ${total_capital:,.2f}")
print(f"\nNext steps:")
print(f"  - Review individual pair results in advanced_exits_4hour_<PAIR>_target_1day_return_results.json")
print(f"  - Consider portfolio allocation strategies")
print(f"  - Test on additional pairs or timeframes")
print(f"{'='*80}")
