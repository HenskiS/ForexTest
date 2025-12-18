"""
Generate full historical predictions for backtesting.

This script generates predictions for the entire historical dataset using
a rolling window approach. These predictions are used by test_dca_stop_fixed.py
for backtesting the strategy.

This is OPTIONAL - only needed if you want to run backtests.
Production trading only needs the prediction buffers from initialize_oanda_buffer_ann.py

Safe to run multiple times (idempotent) - will overwrite existing files.
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
from trading import TradingConfig
from trading.trading_models import TradingModel
from trading.market_utils import calculate_technical_features


def generate_predictions_for_pair(pair, model_type='ann', max_predictions=None, update_mode=False, force=False):
    """
    Generate full historical predictions for a pair.

    Args:
        pair: Currency pair
        model_type: 'ann' or 'xgboost'
        max_predictions: Maximum number of predictions to generate (None = all)
        update_mode: If True, only generate predictions for new data since last run
        force: If True, regenerate even if we already have enough predictions

    Returns:
        tuple: (success, error_message)
    """
    print(f"\n{'='*100}")
    print(f"GENERATING PREDICTIONS FOR {pair} ({model_type.upper()})")
    print(f"{'='*100}\n")

    # Load historical data
    data_file = f'data/{pair}_1day_oanda.csv'
    if not os.path.exists(data_file):
        return False, f"Data file not found: {data_file}"

    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    print(f"Loaded {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

    # Calculate features
    print("Calculating technical features...")
    df_with_features = calculate_technical_features(df.copy())
    df_with_features['target_1day_return'] = df_with_features['close'].pct_change(1).shift(-1)
    df_clean = df_with_features.dropna(subset=TradingConfig.TECHNICAL_FEATURES)

    print(f"Clean data: {len(df_clean)} days with valid features")

    # Check for existing predictions
    output_dir = 'optimized_ann_predictions' if model_type == 'ann' else 'optimized_xgboost_predictions'
    output_file = f'{output_dir}/predictions_{pair}.pkl'

    existing_predictions = []
    existing_test_indices = []
    start_idx = TradingConfig.TRAIN_WINDOW_SIZE

    # Load existing predictions if they exist
    if os.path.exists(output_file):
        try:
            with open(output_file, 'rb') as f:
                existing_data = pickle.load(f)
            existing_predictions = existing_data['predictions']
            existing_test_indices = existing_data['test_indices']
            print(f"Found {len(existing_predictions)} existing predictions")
        except Exception as e:
            print(f"Warning: Could not load existing predictions: {e}")
            print("Starting fresh generation")

    # Initialize model (verbose=False to reduce output during batch generation)
    model = TradingModel(pair, model_type=model_type, verbose=False)

    # Generate predictions using rolling window
    train_window = TradingConfig.TRAIN_WINDOW_SIZE
    predictions = []
    test_indices = []

    # Determine how many new predictions to generate
    end_idx = len(df_clean)

    if max_predictions is not None:
        # Check if we already have enough predictions
        if not force and len(existing_predictions) >= max_predictions:
            print(f"[SKIP] Already have {len(existing_predictions)} predictions (requested {max_predictions})")
            print(f"       Use --force to regenerate")
            return True, None

        # If we have some predictions but not enough, only generate what's needed
        if not force and len(existing_predictions) > 0:
            # Keep existing predictions and only generate the difference
            predictions = list(existing_predictions)
            test_indices = list(existing_test_indices)
            # Calculate how many more we need (work backwards from most recent)
            needed = max_predictions - len(existing_predictions)
            start_idx = max(train_window, end_idx - needed)
            print(f"Have {len(existing_predictions)} predictions, need {needed} more to reach {max_predictions}")
        else:
            # Force regeneration or no existing predictions
            # Work backwards from the end (most recent data)
            start_idx = max(train_window, end_idx - max_predictions)
            if force and len(existing_predictions) > 0:
                print(f"Force mode: Regenerating all {max_predictions} predictions")
    elif update_mode:
        # Update mode: continue from where we left off
        if existing_test_indices:
            predictions = list(existing_predictions)
            test_indices = list(existing_test_indices)
            start_idx = max(existing_test_indices) + 1
            print(f"Update mode: Continuing from index {start_idx}")
        else:
            print("Update mode: No existing predictions, generating all")
    else:
        # Full mode: generate all predictions
        if not force and len(existing_predictions) > 0:
            print(f"Already have {len(existing_predictions)} predictions")
            print(f"Use --force to regenerate all")
            return True, None

    new_predictions_count = end_idx - start_idx

    if new_predictions_count <= 0:
        print(f"No new predictions to generate (already up to date)")
        return True, None

    print(f"\nGenerating {new_predictions_count} predictions with {train_window}-day rolling window...")
    print(f"This will take approximately {new_predictions_count // 30} minutes\n")

    # Start predictions after we have enough data for training
    for i in range(start_idx, end_idx):
        # We need at least train_window + extra margin for features/targets
        # Skip if we don't have enough historical data
        if i < train_window:
            continue

        # Train on full historical data up to current day (excluding current day)
        # The model.train() function will handle selecting the appropriate training window
        df_train = df_clean.iloc[:i].copy()

        # Train model
        try:
            _, _, _ = model.train(df_train)

            # Generate prediction for current day
            prediction = model.predict(df_clean.iloc[:i])
            predictions.append(prediction)
            test_indices.append(i)

            # Progress indicator - only print every 100 predictions or at the end
            if (i - start_idx + 1) % 100 == 0 or (i - start_idx + 1) == new_predictions_count:
                progress = ((i - start_idx + 1) / new_predictions_count) * 100
                print(f"  Progress: {progress:.1f}% ({i - start_idx + 1}/{new_predictions_count})", flush=True)

        except Exception as e:
            print(f"  Warning: Failed to generate prediction for index {i}: {e}")
            continue

    print(f"\nGenerated {len(predictions)} total predictions ({len(predictions) - len(existing_predictions)} new)")

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump({
            'predictions': predictions,
            'test_indices': test_indices,
            'pair': pair,
            'model_type': model_type
        }, f)

    print(f"Predictions saved to: {output_file}")
    print(f"[OK] {pair} complete")

    return True, None


def generate_all_predictions(model_type='ann', max_predictions=None, update_mode=False, force=False):
    """
    Generate predictions for all pairs.

    Args:
        model_type: 'ann' or 'xgboost'
        max_predictions: Maximum number of predictions per pair (None = all)
        update_mode: If True, only generate new predictions since last run
        force: If True, regenerate even if predictions already exist

    Returns:
        dict: Results for each pair {pair: {'success': bool, 'error': str}}
    """
    pairs = TradingConfig.DEFAULT_PAIRS

    print("="*100)
    print(f"GENERATING BACKTEST PREDICTIONS ({model_type.upper()} MODEL)")
    print("="*100)
    print(f"\nPairs: {', '.join(pairs)}")
    print(f"Model: {model_type.upper()}")

    if update_mode:
        print(f"Mode: UPDATE (only generate new predictions)")
    elif max_predictions:
        print(f"Max predictions per pair: {max_predictions}")
    else:
        print(f"Mode: FULL (entire historical dataset)")

    if force:
        print(f"Force mode: Will regenerate existing predictions")

    print(f"\nNote: Used by backtest scripts (test_dca_stop_fixed.py, etc.)")
    print(f"      NOT needed for production trading (use initialize_oanda_buffer_ann.py)")
    print()

    results = {}

    for i, pair in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Processing {pair}...")

        success, error = generate_predictions_for_pair(pair, model_type, max_predictions, update_mode, force)
        results[pair] = {'success': success, 'error': error}

        if not success:
            print(f"  [FAIL] {error}")

    # Summary
    print("\n" + "="*100)
    print("PREDICTION GENERATION SUMMARY")
    print("="*100)

    successful = sum(1 for r in results.values() if r['success'])
    failed = len(pairs) - successful

    print(f"\nSuccessful: {successful}/{len(pairs)}")
    print(f"Failed: {failed}/{len(pairs)}")

    if successful > 0:
        print("\n[OK] Successfully generated:")
        for pair, result in results.items():
            if result['success']:
                output_dir = 'optimized_ann_predictions' if model_type == 'ann' else 'optimized_xgboost_predictions'
                print(f"  - {pair}: {output_dir}/predictions_{pair}.pkl")

    if failed > 0:
        print("\n[FAIL] Failed to generate:")
        for pair, result in results.items():
            if not result['success']:
                print(f"  - {pair}: {result['error']}")

    print()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate backtest predictions for all pairs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all predictions (full historical dataset)
  python generate_backtest_predictions.py

  # Update existing predictions with new data only
  python generate_backtest_predictions.py --update

  # Generate only 500 predictions per pair (will skip if already have 500+)
  python generate_backtest_predictions.py --max-predictions 500

  # Generate 500 predictions, adding to existing (e.g., have 250, will generate 250 more)
  python generate_backtest_predictions.py --max-predictions 500

  # Force regenerate 500 predictions even if we already have them
  python generate_backtest_predictions.py --max-predictions 500 --force

  # Use XGBoost model instead of ANN
  python generate_backtest_predictions.py --model xgboost
        """
    )
    parser.add_argument('--model', type=str, default='ann', choices=['ann', 'xgboost'],
                       help='Model type (default: ann)')
    parser.add_argument('--max-predictions', type=int, default=None,
                       help='Maximum predictions per pair (default: all available data)')
    parser.add_argument('--update', action='store_true',
                       help='Update mode: only generate new predictions since last run')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration even if predictions already exist')
    args = parser.parse_args()

    results = generate_all_predictions(
        model_type=args.model,
        max_predictions=args.max_predictions,
        update_mode=args.update,
        force=args.force
    )

    # Exit with error code if any failed
    failed_count = sum(1 for r in results.values() if not r['success'])
    if failed_count > 0:
        sys.exit(1)

    sys.exit(0)
