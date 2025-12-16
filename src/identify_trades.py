#!/usr/bin/env python3
"""
identify_trades.py

Identifies current potential trading opportunities using a trained model.

This script:
  1. Loads the trained model and feature list
  2. For each ticker in the universe, loads the latest feature data
  3. Generates predictions using the model
  4. Filters and ranks opportunities based on prediction probability
  5. Outputs a report of potential trades with key metrics
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.stop_loss_policy import StopLossConfig, calculate_stop_loss_pct

# —— CONFIGURATION —— #
DATA_DIR = PROJECT_ROOT / "data" / "features_labeled"
MODEL_DIR = PROJECT_ROOT / "models"
TICKERS_CSV = PROJECT_ROOT / "data" / "tickers" / "sp500_tickers.csv"
DEFAULT_MODEL = MODEL_DIR / "xgb_classifier_selected_features.pkl"
MIN_PROBABILITY = 0.5  # Minimum prediction probability to consider a trade


def load_model(model_path: Path) -> Tuple:
    """
    Load the trained model, feature list, and scaler from a pickle file.

    Args:
        model_path: Path to the model pickle file.

    Returns:
        Tuple of (model, feature_list, scaler, features_to_scale, features_to_keep).
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    data = joblib.load(model_path)
    if isinstance(data, dict):
        model = data.get("model")
        features = data.get("features", [])
        scaler = data.get("scaler")
        features_to_scale = data.get("features_to_scale", [])
        features_to_keep = data.get("features_to_keep", [])
    else:
        model = data
        features = []
        scaler = None
        features_to_scale = []
        features_to_keep = []
    
    return model, features, scaler, features_to_scale, features_to_keep


def get_recommended_filters() -> Dict[str, Tuple[str, float]]:
    """
    Get recommended entry filters based on stop-loss analysis.
    
    These filters are based on top findings from stop-loss analysis:
    - Candle body percentage (effect size: -0.725)
    - Close position in range (effect size: -0.646)
    - Weekly RSI (effect size: 0.603)
    - Market correlation (effect size: -0.497)
    - Volatility regime (effect size: -0.464)
    
    Returns:
        Dict of {feature_name: (operator, threshold)}
    """
    return {
        'candle_body_pct': ('>', -10.0),  # Avoid very bearish candles
        'close_position_in_range': ('>', 0.40),  # Price in upper portion of range
        'weekly_rsi_14w': ('<', 42.0),  # Avoid overbought weekly RSI
        'market_correlation_20d': ('>', 0.60),  # Only when moving with market
        'volatility_regime': ('>', 60.0),  # Prefer higher volatility
    }


def apply_entry_filters(
    feature_vector: pd.Series,
    filters: Dict[str, Tuple[str, float]]
) -> bool:
    """
    Apply entry filters to a feature vector.
    
    Args:
        feature_vector: Series with feature values
        filters: Dict of {feature_name: (operator, threshold)} where operator is '>', '<', '>=', '<='
    
    Returns:
        True if all filters pass, False otherwise
    """
    if not filters:
        return True
    
    for feature, (operator, threshold) in filters.items():
        if feature not in feature_vector.index:
            # Feature not available - skip this filter (or return False if strict)
            continue
        
        feature_value = feature_vector[feature]
        
        # Handle NaN values
        if pd.isna(feature_value):
            return False
        
        if operator == '>':
            if not (feature_value > threshold):
                return False
        elif operator == '<':
            if not (feature_value < threshold):
                return False
        elif operator == '>=':
            if not (feature_value >= threshold):
                return False
        elif operator == '<=':
            if not (feature_value <= threshold):
                return False
        else:
            # Unknown operator - skip
            continue
    
    return True


def load_latest_features(ticker: str, data_dir: Path, features: List[str]) -> pd.Series:
    """
    Load the latest feature vector for a ticker.

    Args:
        ticker: Ticker symbol.
        data_dir: Directory containing feature Parquet files.
        features: List of feature names to extract.

    Returns:
        Series with the latest feature values, or None if not available.
    """
    file_path = data_dir / f"{ticker}.parquet"
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_parquet(file_path)
        if df.empty:
            return None
        
        # Get the most recent row
        latest = df.iloc[-1]
        
        # Extract only the features we need
        available_features = [f for f in features if f in latest.index]
        if len(available_features) < len(features) * 0.8:  # Require at least 80% of features
            return None
        
        feature_vector = latest[available_features]
        return feature_vector
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None


def get_current_price(ticker: str) -> float:
    """
    Get the current price for a ticker using yfinance.

    Args:
        ticker: Ticker symbol.

    Returns:
        Current price, or None if unavailable.
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.history(period="1d")
        if not info.empty:
            return float(info['Close'].iloc[-1])
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
    return None


def identify_opportunities(
    model,
    features: List[str],
    data_dir: Path,
    tickers: List[str],
    min_probability: float = MIN_PROBABILITY,
    top_n: int = 20,
    scaler=None,
    features_to_scale: List[str] = None,
    entry_filters: Optional[Dict[str, Tuple[str, float]]] = None,
    stop_loss_config: Optional[StopLossConfig] = None
) -> pd.DataFrame:
    """
    Identify trading opportunities across a universe of tickers.

    Args:
        model: Trained model for prediction.
        features: List of feature names the model expects.
        data_dir: Directory containing feature Parquet files.
        tickers: List of ticker symbols to analyze.
        min_probability: Minimum prediction probability to consider.
        top_n: Maximum number of opportunities to return.

    Returns:
        DataFrame with columns: ticker, probability, current_price, date.
    """
    opportunities = []
    
    print(f"Analyzing {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(tickers)} tickers...")
        
        # Load latest features
        feature_vector = load_latest_features(ticker, data_dir, features)
        if feature_vector is None:
            continue
        
        # Apply entry filters if provided (before making prediction)
        if entry_filters:
            if not apply_entry_filters(feature_vector, entry_filters):
                continue  # Skip this ticker if filters don't pass
        
        # Align features with model expectations
        feature_df = pd.DataFrame([feature_vector])
        available_features = [f for f in features if f in feature_df.columns]
        
        if len(available_features) < len(features) * 0.8:
            continue
        
        # Prepare feature matrix
        X = feature_df[available_features]
        
        # Handle missing features by filling with 0 (or median if available)
        for f in features:
            if f not in X.columns:
                X[f] = 0.0
        
        X = X[features]  # Ensure correct order
        
        # Apply scaling if scaler is provided
        if scaler is not None and features_to_scale:
            X_scaled = X.copy()
            # Only scale features that were scaled during training
            scale_cols = [f for f in features_to_scale if f in X.columns]
            if scale_cols:
                X_scaled[scale_cols] = scaler.transform(X[scale_cols])
            X = X_scaled
        
        # Make prediction
        try:
            proba = model.predict_proba(X)[0, 1]  # Probability of positive class
            
            if proba >= min_probability:
                current_price = get_current_price(ticker)
                
                # Calculate adaptive stop-loss if configured
                stop_loss_pct = None
                if stop_loss_config is not None and stop_loss_config.mode in ["adaptive_atr", "swing_atr"]:
                    try:
                        # Use the feature_vector (not X, which may be scaled) to get atr14_normalized
                        # Pass entry_price for swing_atr mode
                        entry_price_val = current_price if current_price is not None else None
                        stop_loss_pct, _ = calculate_stop_loss_pct(feature_vector, stop_loss_config, entry_price=entry_price_val)
                    except Exception as e:
                        # If stop calculation fails, leave as None
                        pass
                
                opportunities.append({
                    'ticker': ticker,
                    'probability': proba,
                    'current_price': current_price,
                    'stop_loss_pct': stop_loss_pct,
                    'date': datetime.now().strftime('%Y-%m-%d')
                })
        except Exception as e:
            print(f"Error predicting for {ticker}: {e}")
            continue
    
    if not opportunities:
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by probability
    df = pd.DataFrame(opportunities)
    df = df.sort_values('probability', ascending=False)
    df = df.head(top_n)
    
    return df


def main():
    """
    Main entry point for trade identification.
    """
    parser = argparse.ArgumentParser(
        description="Identify current trading opportunities using trained model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to trained model pickle file"
    )
    parser.add_argument(
        "--tickers-file",
        type=str,
        default=str(TICKERS_CSV),
        help="Path to CSV file with ticker symbols"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory containing feature Parquet files"
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=MIN_PROBABILITY,
        help="Minimum prediction probability to consider (0.0-1.0)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Maximum number of opportunities to return"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (optional)"
    )
    parser.add_argument(
        "--use-recommended-filters",
        action="store_true",
        default=False,
        help="Use recommended filters from stop-loss analysis (default: False)"
    )
    parser.add_argument(
        "--custom-filter",
        action="append",
        nargs=3,
        metavar=("FEATURE", "OPERATOR", "THRESHOLD"),
        help="Add custom filter: --custom-filter feature_name > 0.5 (can be used multiple times)"
    )
    parser.add_argument(
        "--stop-loss-mode",
        type=str,
        choices=["constant", "adaptive_atr", "swing_atr"],
        default=None,
        help="Stop-loss mode: 'constant' (fixed), 'adaptive_atr' (ATR-based), or 'swing_atr' (swing low + ATR buffer). If specified, calculates and displays recommended stop-loss for each trade."
    )
    parser.add_argument(
        "--atr-stop-k",
        type=float,
        default=1.8,
        help="ATR multiplier for adaptive stops (default: 1.8). Used for adaptive_atr and swing_atr fallback."
    )
    parser.add_argument(
        "--atr-stop-min-pct",
        type=float,
        default=0.04,
        help="Minimum stop distance for adaptive stops (default: 0.04 = 4%%). Used for adaptive_atr and swing_atr."
    )
    parser.add_argument(
        "--atr-stop-max-pct",
        type=float,
        default=0.10,
        help="Maximum stop distance for adaptive stops (default: 0.10 = 10%%). Used for adaptive_atr and swing_atr."
    )
    parser.add_argument(
        "--swing-lookback-days",
        type=int,
        default=10,
        help="Days to look back for swing low (default: 10). Only used when --stop-loss-mode=swing_atr."
    )
    parser.add_argument(
        "--swing-atr-buffer-k",
        type=float,
        default=0.75,
        help="ATR multiplier for swing_atr buffer (default: 0.75). Only used when --stop-loss-mode=swing_atr."
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, features, scaler, features_to_scale, _ = load_model(Path(args.model))
    print(f"Model loaded. Using {len(features)} features.")
    if scaler is not None:
        print(f"Scaler loaded. Will scale {len(features_to_scale)} features during prediction.")
    
    # Load tickers
    tickers_df = pd.read_csv(args.tickers_file, header=None)
    tickers = tickers_df.iloc[:, 0].astype(str).tolist()
    print(f"Loaded {len(tickers)} tickers from {args.tickers_file}")
    
    # Build entry filters
    entry_filters = {}
    
    if args.use_recommended_filters:
        entry_filters.update(get_recommended_filters())
        print(f"\nUsing recommended filters from stop-loss analysis:")
        for feat, (op, val) in sorted(entry_filters.items()):
            print(f"  {feat} {op} {val}")
    
    # Add custom filters
    if args.custom_filter:
        for feature, operator, threshold_str in args.custom_filter:
            try:
                threshold = float(threshold_str)
                if operator not in ['>', '<', '>=', '<=']:
                    print(f"Warning: Invalid operator '{operator}'. Must be one of: >, <, >=, <=")
                    continue
                entry_filters[feature] = (operator, threshold)
                print(f"  Added custom filter: {feature} {operator} {threshold}")
            except ValueError:
                print(f"Warning: Invalid threshold '{threshold_str}' for {feature}. Skipping.")
    
    if entry_filters:
        print(f"\nTotal filters applied: {len(entry_filters)}")
    else:
        print("\nNo entry filters applied")
    
    # Create stop-loss configuration if adaptive mode is specified
    stop_loss_config = None
    if args.stop_loss_mode in ["adaptive_atr", "swing_atr"]:
        stop_loss_config = StopLossConfig(
            mode=args.stop_loss_mode,
            atr_stop_k=args.atr_stop_k,
            atr_stop_min_pct=args.atr_stop_min_pct,
            atr_stop_max_pct=args.atr_stop_max_pct,
            swing_lookback_days=args.swing_lookback_days,
            swing_atr_buffer_k=args.swing_atr_buffer_k
        )
        mode_name = "Adaptive ATR" if args.stop_loss_mode == "adaptive_atr" else "Swing ATR"
        print(f"\n{mode_name} Stop-Loss Configuration:")
        if args.stop_loss_mode == "adaptive_atr":
            print(f"  ATR Stop K: {stop_loss_config.atr_stop_k}")
            print(f"  ATR Stop Range: {stop_loss_config.atr_stop_min_pct:.2%} - {stop_loss_config.atr_stop_max_pct:.2%}")
        else:
            print(f"  Swing Lookback Days: {stop_loss_config.swing_lookback_days}")
            print(f"  Swing ATR Buffer K: {stop_loss_config.swing_atr_buffer_k}")
            print(f"  Stop Range: {stop_loss_config.atr_stop_min_pct:.2%} - {stop_loss_config.atr_stop_max_pct:.2%}")
    
    # Identify opportunities
    print("\nIdentifying trading opportunities...")
    opportunities = identify_opportunities(
        model=model,
        features=features,
        data_dir=Path(args.data_dir),
        tickers=tickers,
        min_probability=args.min_probability,
        top_n=args.top_n,
        scaler=scaler,
        features_to_scale=features_to_scale,
        entry_filters=entry_filters if entry_filters else None,
        stop_loss_config=stop_loss_config
    )
    
    # Display results
    print("\n" + "="*80)
    print("TRADING OPPORTUNITIES")
    print("="*80)
    
    if opportunities.empty:
        print("\nNo trading opportunities found matching the criteria.")
    else:
        print(f"\nFound {len(opportunities)} opportunities:\n")
        
        # Format output for display
        display_df = opportunities.copy()
        
        # Format probability as percentage
        if 'probability' in display_df.columns:
            display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.1%}")
        
        # Format stop-loss percentage if present
        if 'stop_loss_pct' in display_df.columns:
            display_df['stop_loss_pct'] = display_df['stop_loss_pct'].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) and x is not None else "N/A"
            )
            # Rename column for better display
            display_df = display_df.rename(columns={'stop_loss_pct': 'stop_loss_%'})
        
        # Format current price
        if 'current_price' in display_df.columns:
            display_df['current_price'] = display_df['current_price'].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) and x is not None else "N/A"
            )
        
        print(display_df.to_string(index=False))
        
        # Save to file if requested (save original data with numeric values)
        if args.output:
            opportunities.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

