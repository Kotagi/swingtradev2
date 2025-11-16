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
from typing import List, Dict, Tuple
from datetime import datetime

# —— CONFIGURATION —— #
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "features_labeled"
MODEL_DIR = PROJECT_ROOT / "models"
TICKERS_CSV = PROJECT_ROOT / "data" / "tickers" / "sp500_tickers.csv"
DEFAULT_MODEL = MODEL_DIR / "xgb_classifier_selected_features.pkl"
MIN_PROBABILITY = 0.5  # Minimum prediction probability to consider a trade


def load_model(model_path: Path) -> Tuple:
    """
    Load the trained model and feature list from a pickle file.

    Args:
        model_path: Path to the model pickle file.

    Returns:
        Tuple of (model, feature_list).
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    data = joblib.load(model_path)
    if isinstance(data, dict):
        model = data.get("model")
        features = data.get("features", [])
    else:
        model = data
        features = []
    
    return model, features


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
    top_n: int = 20
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
        
        # Make prediction
        try:
            proba = model.predict_proba(X)[0, 1]  # Probability of positive class
            
            if proba >= min_probability:
                current_price = get_current_price(ticker)
                opportunities.append({
                    'ticker': ticker,
                    'probability': proba,
                    'current_price': current_price,
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
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, features = load_model(Path(args.model))
    print(f"Model loaded. Using {len(features)} features.")
    
    # Load tickers
    tickers_df = pd.read_csv(args.tickers_file, header=None)
    tickers = tickers_df.iloc[:, 0].astype(str).tolist()
    print(f"Loaded {len(tickers)} tickers from {args.tickers_file}")
    
    # Identify opportunities
    print("\nIdentifying trading opportunities...")
    opportunities = identify_opportunities(
        model=model,
        features=features,
        data_dir=Path(args.data_dir),
        tickers=tickers,
        min_probability=args.min_probability,
        top_n=args.top_n
    )
    
    # Display results
    print("\n" + "="*80)
    print("TRADING OPPORTUNITIES")
    print("="*80)
    
    if opportunities.empty:
        print("\nNo trading opportunities found matching the criteria.")
    else:
        print(f"\nFound {len(opportunities)} opportunities:\n")
        print(opportunities.to_string(index=False))
        
        # Save to file if requested
        if args.output:
            opportunities.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

