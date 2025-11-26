#!/usr/bin/env python3
"""
tune_threshold.py

Finds the optimal prediction threshold for the trained model by analyzing
precision-recall curves on the validation set.

Usage:
    python src/tune_threshold.py [--model MODEL_PATH] [--target-precision TARGET] [--target-recall TARGET]
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# Default paths
DEFAULT_MODEL = Path("models/xgb_classifier_selected_features.pkl")
DEFAULT_DATA_DIR = Path("data/features_labeled")
DEFAULT_TRAIN_END = "2022-12-31"
DEFAULT_VAL_END = "2023-12-31"


def load_model(model_path: Path) -> Tuple:
    """Load model and extract components."""
    data = joblib.load(model_path)
    model = data.get("model")
    features = data.get("features", [])
    scaler = data.get("scaler", None)
    features_to_scale = data.get("features_to_scale", [])
    features_to_keep = data.get("features_to_keep", features)
    
    return model, features, scaler, features_to_scale, features_to_keep


def load_validation_data(data_dir: Path, features: list, label_col: str,
                        train_end: str, val_end: str,
                        scaler=None, features_to_scale: list = None) -> Tuple:
    """Load and prepare validation data."""
    import glob
    
    # Load all feature files
    parquet_files = glob.glob(str(data_dir / "*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            if label_col in df.columns:
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
            continue
    
    if not dfs:
        raise ValueError(f"No valid data files found in {data_dir}")
    
    # Concatenate all data
    df_all = pd.concat(dfs, ignore_index=False)
    
    # Handle date index - check if index is datetime, or if there's a date column
    if isinstance(df_all.index, pd.DatetimeIndex):
        # Index is already datetime
        date_index = df_all.index
    elif 'date' in df_all.columns:
        # Convert date column to datetime and set as index
        df_all['date'] = pd.to_datetime(df_all['date'])
        df_all = df_all.set_index('date')
        date_index = df_all.index
    else:
        # No date information - can't filter by date
        raise ValueError("Data must have either a DatetimeIndex or a 'date' column")
    
    # Convert date strings to datetime for comparison
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    # Filter by date range
    df_all = df_all.sort_index()
    val_data = df_all[(date_index > train_end_dt) & (date_index <= val_end_dt)]
    
    if len(val_data) == 0:
        raise ValueError(f"No validation data found between {train_end} and {val_end}")
    
    # Prepare features
    available_features = [f for f in features if f in val_data.columns]
    missing_features = [f for f in features if f not in val_data.columns]
    
    if missing_features:
        print(f"Warning: {len(missing_features)} features missing from data. Filling with 0.")
        for f in missing_features:
            val_data[f] = 0.0
    
    X_val = val_data[features].copy()
    y_val = val_data[label_col].copy()
    
    # Handle infinities and NaNs
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.fillna(0.0)
    
    # Apply scaling if provided
    if scaler is not None and features_to_scale:
        scale_cols = [f for f in features_to_scale if f in X_val.columns]
        if scale_cols:
            X_val_scaled = X_val.copy()
            X_val_scaled[scale_cols] = scaler.transform(X_val[scale_cols])
            X_val = X_val_scaled
    
    # Remove rows with NaN labels
    valid_mask = ~y_val.isna()
    X_val = X_val[valid_mask]
    y_val = y_val[valid_mask]
    
    return X_val, y_val


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_precision: Optional[float] = None,
    target_recall: Optional[float] = None,
    optimize_for: str = "f1"
) -> Dict:
    """
    Find optimal threshold based on precision-recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        target_precision: Target precision (if specified, finds threshold closest to this)
        target_recall: Target recall (if specified, finds threshold closest to this)
        optimize_for: What to optimize for ("f1", "precision", "recall", "balanced")
    
    Returns:
        Dictionary with optimal threshold and metrics
    """
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find optimal threshold based on optimization strategy
    if target_precision is not None:
        # Find threshold closest to target precision
        idx = np.argmin(np.abs(precisions - target_precision))
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
        optimal_precision = precisions[idx]
        optimal_recall = recalls[idx]
        optimal_f1 = f1_scores[idx]
    elif target_recall is not None:
        # Find threshold closest to target recall
        idx = np.argmin(np.abs(recalls - target_recall))
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
        optimal_precision = precisions[idx]
        optimal_recall = recalls[idx]
        optimal_f1 = f1_scores[idx]
    elif optimize_for == "f1":
        # Maximize F1 score
        idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
        optimal_precision = precisions[idx]
        optimal_recall = recalls[idx]
        optimal_f1 = f1_scores[idx]
    elif optimize_for == "precision":
        # Maximize precision (at minimum recall of 0.3)
        valid_idx = recalls >= 0.3
        if valid_idx.any():
            idx = np.argmax(precisions[valid_idx])
            optimal_threshold = thresholds[valid_idx][idx] if idx < len(thresholds[valid_idx]) else thresholds[-1]
            optimal_precision = precisions[valid_idx][idx]
            optimal_recall = recalls[valid_idx][idx]
            optimal_f1 = f1_scores[valid_idx][idx]
        else:
            # Fallback to F1
            idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
            optimal_precision = precisions[idx]
            optimal_recall = recalls[idx]
            optimal_f1 = f1_scores[idx]
    else:  # balanced
        # Balance precision and recall (maximize harmonic mean)
        idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
        optimal_precision = precisions[idx]
        optimal_recall = recalls[idx]
        optimal_f1 = f1_scores[idx]
    
    # Calculate metrics at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    result = {
        "optimal_threshold": float(optimal_threshold),
        "precision": float(optimal_precision),
        "recall": float(optimal_recall),
        "f1_score": float(optimal_f1),
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1])
        },
        "total_predictions": int(len(y_true)),
        "positive_predictions": int(y_pred.sum()),
        "positive_rate": float(y_pred.mean())
    }
    
    return result


def analyze_threshold_range(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None
) -> pd.DataFrame:
    """Analyze metrics across a range of thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "positive_predictions": y_pred.sum(),
            "positive_rate": y_pred.mean()
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal prediction threshold for trained model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to trained model file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing feature Parquet files"
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label column name (auto-detected if not specified)"
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=DEFAULT_TRAIN_END,
        help="Training data end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default=DEFAULT_VAL_END,
        help="Validation data end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=None,
        help="Target precision (finds threshold closest to this)"
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="Target recall (finds threshold closest to this)"
    )
    parser.add_argument(
        "--optimize-for",
        type=str,
        choices=["f1", "precision", "recall", "balanced"],
        default="f1",
        help="What to optimize for (default: f1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for threshold results (optional)"
    )
    
    args = parser.parse_args()
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model, features, scaler, features_to_scale, features_to_keep = load_model(model_path)
    print(f"Model loaded. Using {len(features)} features.")
    
    # Auto-detect label column if not specified
    label_col = args.label_col
    if label_col is None:
        # Try to detect from data
        import glob
        sample_file = glob.glob(str(Path(args.data_dir) / "*.parquet"))[0]
        sample_df = pd.read_parquet(sample_file)
        label_cols = [col for col in sample_df.columns if col.startswith("label_")]
        if label_cols:
            label_col = label_cols[0]
            print(f"Auto-detected label column: {label_col}")
        else:
            raise ValueError("Could not auto-detect label column. Please specify --label-col")
    
    # Load validation data
    print(f"Loading validation data from {args.data_dir}...")
    X_val, y_val = load_validation_data(
        Path(args.data_dir),
        features,
        label_col,
        args.train_end,
        args.val_end,
        scaler,
        features_to_scale
    )
    print(f"Validation data: {len(X_val)} samples, {y_val.sum()} positive ({y_val.mean()*100:.2f}%)")
    
    # Get predictions
    print("Generating predictions...")
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Find optimal threshold
    print(f"\nFinding optimal threshold (optimize_for={args.optimize_for})...")
    result = find_optimal_threshold(
        y_val.values,
        y_proba,
        args.target_precision,
        args.target_recall,
        args.optimize_for
    )
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("="*70)
    print(f"Optimal Threshold: {result['optimal_threshold']:.4f}")
    print(f"Precision: {result['precision']:.4f} ({result['precision']*100:.2f}%)")
    print(f"Recall: {result['recall']:.4f} ({result['recall']*100:.2f}%)")
    print(f"F1 Score: {result['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {result['confusion_matrix']['true_negatives']:,}")
    print(f"  False Positives: {result['confusion_matrix']['false_positives']:,}")
    print(f"  False Negatives: {result['confusion_matrix']['false_negatives']:,}")
    print(f"  True Positives:  {result['confusion_matrix']['true_positives']:,}")
    print(f"\nTotal Predictions: {result['total_predictions']:,}")
    print(f"Positive Predictions: {result['positive_predictions']:,} ({result['positive_rate']*100:.2f}%)")
    
    # Analyze threshold range
    print("\n" + "="*70)
    print("THRESHOLD RANGE ANALYSIS")
    print("="*70)
    threshold_df = analyze_threshold_range(y_val.values, y_proba)
    print(threshold_df.to_string(index=False))
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    # Save to model metadata if possible
    metadata_path = model_path.parent / "training_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata["optimal_threshold"] = result['optimal_threshold']
        metadata["threshold_metrics"] = {
            "precision": result['precision'],
            "recall": result['recall'],
            "f1_score": result['f1_score']
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Threshold saved to model metadata: {metadata_path}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"Use threshold: {result['optimal_threshold']:.4f}")
    print(f"This gives you:")
    print(f"  - Precision: {result['precision']*100:.2f}% (reduces false positives)")
    print(f"  - Recall: {result['recall']*100:.2f}% (captures true signals)")
    print(f"\nTo use this threshold in backtesting:")
    print(f"  python src/swing_trade_app.py backtest --model-threshold {result['optimal_threshold']:.4f}")


if __name__ == "__main__":
    main()

