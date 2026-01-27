#!/usr/bin/env python3
"""
train_model.py

Enhanced training script for XGBoost classifier with:
- Hyperparameter tuning (RandomizedSearchCV)
- Early stopping with validation set
- Cross-validation for robust evaluation
- Comprehensive evaluation metrics
- Feature importance analysis
- Model versioning and metadata tracking

This script:
  1. Loads all labeled feature Parquet files from `data/features_labeled/`
  2. Filters features based on `config/train_features_v1.yaml` (or feature set config)
  3. Drops NaNs and raw-price/forward-return columns to prevent leakage
  4. Splits data into train/validation/test by date
  5. Trains baseline models (DummyClassifier & LogisticRegression)
  6. Performs hyperparameter tuning with RandomizedSearchCV
  7. Trains XGBClassifier with early stopping and class-imbalance handling
  8. Evaluates performance with comprehensive metrics
  9. Saves the trained model, feature list, and training metadata
  10. Displays feature importances and optionally SHAP diagnostics
"""

import argparse
import sys
import yaml
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import os

# Add project root and src to Python path
# Resolve to absolute path to ensure correct resolution when run as subprocess
_script_file = Path(__file__).resolve()
PROJECT_ROOT = _script_file.parent.parent
SRC_DIR = _script_file.parent
_project_root_str = str(PROJECT_ROOT)
_src_dir_str = str(SRC_DIR)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)
if _src_dir_str not in sys.path:
    sys.path.insert(0, _src_dir_str)

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from typing import Tuple, List, Dict, Optional

# Import labeling utility (used during training to calculate labels)
from utils.labeling import label_future_return

# Import feature set manager
try:
    from feature_set_manager import (
        get_feature_set_data_path,
        get_train_features_config_path,
        feature_set_exists,
        DEFAULT_FEATURE_SET
    )
    HAS_FEATURE_SET_MANAGER = True
except ImportError:
    HAS_FEATURE_SET_MANAGER = False
    DEFAULT_FEATURE_SET = "v3_New_Dawn"

# Try to import matplotlib for plotting (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ——— CONFIGURATION ————————————————————————————————————————————————
PROJECT_ROOT = Path.cwd()
# Default paths (will be overridden if feature_set is specified)
DATA_DIR     = PROJECT_ROOT / "data" / "features_labeled"  # expects .parquet files
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_OUT    = MODEL_DIR / "xgb_classifier_selected_features.pkl"
# Data split configuration
# Recommended split for 500+ tickers with data from 2008:
# - Training: 2010-2021 (12 years, includes COVID, excludes 2008-2009 crisis)
# - Validation: 2022-2023 (2 years, includes high volatility 2022 and recovery 2023)
# - Test: 2024+ (12+ months, ongoing for evaluation)
# 
# Rationale:
# - Excludes 2008-2009 financial crisis (unique event, market structure changed)
# - Includes 2020 COVID (learns from extreme volatility, similar events may occur)
# - Includes 2018-2019 trade wars (normal market dynamics)
# - 2-year validation provides robust tuning and tests across different regimes
# - 12+ month test set provides statistically meaningful evaluation
TRAIN_START  = "2010-01-01"      # Start date for training (excludes 2008-2009 crisis)
TRAIN_END    = "2021-12-31"      # Cut-off date (inclusive) for training
VAL_END      = "2023-12-31"      # Cut-off date (inclusive) for validation (validation = 2022-2023)
# 
# Alternative configurations (uncomment to use):
# Option 2: Exclude 2020 COVID (more conservative)
# TRAIN_START  = "2010-01-01"
# TRAIN_END    = "2019-12-31"      # Train on 2010-2019, 2021 (exclude 2020)
# VAL_END      = "2023-12-31"      # Validate on 2022-2023
# 
# Option 3: Maximum history (includes 2008-2009)
# TRAIN_START  = None              # Use all available data from 2008
# TRAIN_END    = "2021-12-31"
# VAL_END      = "2023-12-31"
# 
# Option 4: Recent focus (post-crisis only)
# TRAIN_START  = "2015-01-01"      # Post-crisis recovery period
# TRAIN_END    = "2021-12-31"
# VAL_END      = "2023-12-31"
# Label column configuration
# Default: auto-detect from data, or use --horizon/--label-col CLI args
LABEL_COL    = None              # Will be auto-detected or set via CLI
TRAIN_CFG    = PROJECT_ROOT / "config" / "train_features_v1.yaml"  # Default to v1
# —————————————————————————————————————————————————————————————————————————

# Hyperparameter search space for XGBoost
# Note: Lower n_estimators during tuning for faster search, then retrain with more
HYPERPARAMETER_SPACE = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1, 1.5, 2.0]
}

# Fast hyperparameter space (fewer options, lower n_estimators for quicker tuning)
HYPERPARAMETER_SPACE_FAST = {
    'n_estimators': [100, 200],  # Reduced from [100, 200, 300, 500]
    'max_depth': [4, 5, 6],  # Reduced from [3, 4, 5, 6, 7]
    'learning_rate': [0.05, 0.1],  # Reduced from [0.01, 0.05, 0.1, 0.2]
    'min_child_weight': [1, 3],  # Reduced from [1, 3, 5]
    'subsample': [0.8, 0.9],  # Reduced from [0.8, 0.9, 1.0]
    'colsample_bytree': [0.8, 0.9],  # Reduced from [0.8, 0.9, 1.0]
    'gamma': [0, 0.1],  # Reduced from [0, 0.1, 0.2]
    'reg_alpha': [0, 0.1],  # Reduced from [0, 0.1, 1.0]
    'reg_lambda': [1, 1.5]  # Reduced from [1, 1.5, 2.0]
}


def load_data() -> pd.DataFrame:
    """
    Load all ticker feature Parquet files into one concatenated DataFrame.

    Each Parquet in DATA_DIR must have a datetime index and feature columns.
    Adds a 'ticker' column to identify the source symbol.
    
    Excludes reference tickers (SPY and sector ETFs) from training data,
    as these are used only for calculating features on actual stocks.

    Raises:
        FileNotFoundError: if no Parquet files are found.

    Returns:
        DataFrame: concatenated data from all tickers (excluding reference tickers).
    """
    # Reference tickers to exclude from training (used only for calculating features on actual stocks)
    # SPY and sector ETFs should not be included in training data
    REFERENCE_TICKERS = {'SPY', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'XLC', 'XLRE'}
    
    parts: List[pd.DataFrame] = []
    excluded = []
    
    for f in DATA_DIR.glob("*.parquet"):
        ticker = f.stem.upper()
        if ticker in REFERENCE_TICKERS:
            excluded.append(f.stem)
            continue
        
        df = pd.read_parquet(f)
        df.index.name = "date"
        df["ticker"] = f.stem
        
        # Convert numeric columns to float32 immediately after loading (safety measure)
        # Feature pipeline already saves as float32, but this ensures consistency
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].astype(np.float32)
        
        parts.append(df)
    
    if excluded:
        print(f"Excluding {len(excluded)} reference ticker(s) from training data: {', '.join(excluded)}")
    
    if not parts:
        raise FileNotFoundError(f"No Parquet files found in {DATA_DIR} (after excluding reference tickers)")
    
    print(f"Loaded {len(parts)} tickers for training (excluded {len(excluded)} reference tickers)")
    return pd.concat(parts, axis=0)


def prepare(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Clean and split the raw DataFrame into X (features) and y (target).

    - Fills missing values with 0.0 (consistent with backtest/inference).
    - Removes forward-return columns (to prevent leakage).
    - Removes raw price columns: open/high/low/close/adj close.
    - Leaves only label and feature columns.

    Args:
        df: DataFrame with label column, raw prices, and feature columns.
        label_col: Name of the label column to use.

    Returns:
        X: DataFrame of selected feature columns.
        y: Series of the target label.
        feats: List of feature column names used.
    """
    # 1) Replace infinities with NaN, then fill NaNs with 0.0 (consistent with backtest)
    df_clean = df.replace([np.inf, -np.inf], np.nan).copy()
    
    # Count NaNs before filling for logging
    nan_counts = df_clean.isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        nan_pct = (total_nans / (len(df_clean) * len(df_clean.columns))) * 100
        if nan_pct > 5:  # Log if more than 5% NaN values
            print(f"Note: Found {nan_pct:.1f}% NaN values in data. Filling with 0.0 (consistent with inference).")
    
    # Fill NaNs with 0.0 (XGBoost can handle NaN, but filling is safer and consistent with backtest)
    df_clean = df_clean.fillna(0.0)

    # 2) Candidate feature names (exclude label & ticker)
    feats = [c for c in df_clean.columns if c not in [label_col, "ticker"]]

    # 3) Exclude any forward-return columns (labels)
    forward_return_cols = {"5d_return", "10d_return"}
    feats = [c for c in feats if c not in forward_return_cols]

    # 4) Exclude raw price columns to avoid leakage
    raw_price_cols = {"open", "high", "low", "close", "adj close"}
    feats = [c for c in feats if c not in raw_price_cols]

    # 5) Slice out X and y
    X = df_clean[feats].copy()
    y = df_clean[label_col]
    
    # 6) Clip extreme values to prevent overflow (clip to reasonable range)
    # Clip to ±1e6 to prevent overflow issues while preserving signal
    X = X.clip(lower=-1e6, upper=1e6)
    
    # 7) Final check: ensure no infinities remain (NaNs already filled)
    if X.isin([np.inf, -np.inf]).any().any():
        print("Warning: Found infinities after cleaning, replacing with 0.0")
        X = X.replace([np.inf, -np.inf], 0.0)
    
    # Verify no NaNs remain (shouldn't happen after fillna, but double-check)
    if X.isna().any().any():
        print("Warning: Found NaNs after filling, filling again with 0.0")
        X = X.fillna(0.0)

    return X, y, feats


def print_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, 
                             dataset_name: str = "Test") -> Dict:
    """
    Print comprehensive evaluation metrics.
    
    Returns:
        Dictionary of metrics for saving.
    """
    print(f"\n=== {dataset_name.upper()} SET METRICS ===")
    
    # Basic metrics
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    
    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:>8}")
    print(f"  False Positives: {fp:>8}")
    print(f"  False Negatives: {fn:>8}")
    print(f"  True Positives:  {tp:>8}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    
    # Additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"  Precision (Positive): {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  Specificity:          {specificity:.4f}")
    print(f"  Accuracy:             {(tp + tn) / len(y_true):.4f}")
    
    return {
        'roc_auc': float(auc),
        'average_precision': float(ap),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'accuracy': float((tp + tn) / len(y_true)),
        'confusion_matrix': cm.tolist()
    }


def display_feature_importance(model: XGBClassifier, features: List[str], top_n: int = 20):
    """Display feature importance rankings."""
    print(f"\n=== TOP {top_n} FEATURE IMPORTANCES ===")
    importances = model.feature_importances_
    fi = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12} {'Cumulative %':<12}")
    print("-" * 60)
    cumulative = 0
    total_importance = sum(importances)
    
    for rank, (feat, imp) in enumerate(fi[:top_n], 1):
        cumulative += imp
        pct = (imp / total_importance) * 100
        cum_pct = (cumulative / total_importance) * 100
        print(f"{rank:<6} {feat:<30} {imp:<12.6f} {cum_pct:<12.2f}")
    
    if len(fi) > top_n:
        print(f"\n... and {len(fi) - top_n} more features")
    
    return dict(fi)


def plot_training_curves(model: XGBClassifier, output_dir: Path, model_name: str = "xgb"):
    """
    Plot training curves from XGBoost evaluation results.
    
    Args:
        model: Trained XGBoost model with evaluation results
        output_dir: Directory to save plots
        model_name: Name prefix for plot files
    """
    if not HAS_MATPLOTLIB:
        return None
    
    # Check if model has evaluation results
    if not hasattr(model, 'evals_result_') or not model.evals_result_:
        return None
    
    try:
        evals_result = model.evals_result_
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Log Loss (if available)
        if 'validation_0' in evals_result and 'logloss' in evals_result['validation_0']:
            train_loss = evals_result['validation_0']['logloss']
            if 'validation_1' in evals_result and 'logloss' in evals_result['validation_1']:
                val_loss = evals_result['validation_1']['logloss']
            else:
                val_loss = None
            
            ax1 = axes[0]
            iterations = range(1, len(train_loss) + 1)
            ax1.plot(iterations, train_loss, label='Train Loss', linewidth=2)
            if val_loss:
                ax1.plot(iterations, val_loss, label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Iteration (Boosting Round)')
            ax1.set_ylabel('Log Loss')
            ax1.set_title('Training Progress: Log Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: AUC (if available) or Error
        ax2 = axes[1]
        if 'validation_0' in evals_result:
            # Try to find AUC or error metric
            metric_name = None
            for key in ['auc', 'aucpr', 'error', 'merror']:
                if key in evals_result['validation_0']:
                    metric_name = key
                    break
            
            if metric_name:
                train_metric = evals_result['validation_0'][metric_name]
                if 'validation_1' in evals_result and metric_name in evals_result['validation_1']:
                    val_metric = evals_result['validation_1'][metric_name]
                else:
                    val_metric = None
                
                iterations = range(1, len(train_metric) + 1)
                ax2.plot(iterations, train_metric, label=f'Train {metric_name.upper()}', linewidth=2)
                if val_metric:
                    ax2.plot(iterations, val_metric, label=f'Validation {metric_name.upper()}', linewidth=2)
                ax2.set_xlabel('Iteration (Boosting Round)')
                ax2.set_ylabel(metric_name.upper())
                ax2.set_title(f'Training Progress: {metric_name.upper()}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / f"{model_name}_training_curves.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {plot_file}")
        return str(plot_file)
    
    except Exception as e:
        print(f"Warning: Could not create training curves: {e}")
        return None


def plot_feature_importance_chart(model: XGBClassifier, features: List[str], 
                                  output_dir: Path, top_n: int = 20):
    """
    Create a bar chart of top feature importances.
    
    Args:
        model: Trained XGBoost model
        features: List of feature names
        output_dir: Directory to save plot
        top_n: Number of top features to plot
    """
    if not HAS_MATPLOTLIB:
        return None
    
    try:
        importances = model.feature_importances_
        fi = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        top_features = fi[:top_n]
        
        # Extract names and values
        feat_names = [f[0] for f in top_features]
        feat_importances = [f[1] for f in top_features]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(feat_names))
        
        ax.barh(y_pos, feat_importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_names)
        ax.invert_yaxis()  # Top feature at top
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / f"feature_importance_chart.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance chart saved to: {plot_file}")
        return str(plot_file)
    
    except Exception as e:
        print(f"Warning: Could not create feature importance chart: {e}")
        return None


def main() -> None:
    """
    Main routine to train and evaluate the model with enhanced features.
    """
    # Step 1: Parse CLI options
    parser = argparse.ArgumentParser(
        description="Train XGBoost model with hyperparameter tuning and comprehensive evaluation"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning (RandomizedSearchCV). Slower but better results."
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of iterations for hyperparameter search (default: 20)"
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Use cross-validation for hyperparameter tuning (slower but more robust)"
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Compute and print SHAP diagnostics (requires shap library)"
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Compute and save SHAP explanations as artifacts (default: False, but recommended for model interpretability)"
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping (train for full n_estimators)"
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Number of rounds without improvement before stopping (default: 50)"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate training curves and feature importance plots (requires matplotlib)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster hyperparameter tuning (reduced search space, fewer CV folds). ~3-5x faster but slightly less optimal."
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Number of CV folds (default: 3, or 2 if --fast). Lower = faster but less robust."
    )
    parser.add_argument(
        "--imbalance-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for class imbalance handling (default: 1.0). Increase to 2.0-3.0 to favor positive class more. Higher = more trades predicted."
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default=None,
        help="Training data start date (YYYY-MM-DD). Default: 2010-01-01 (excludes 2008-2009 financial crisis). Use None to include all available data from the beginning."
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=None,
        help="Training data end date (YYYY-MM-DD). Default: 2021-12-31 (includes COVID period 2020-2021)."
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default=None,
        help="Validation data end date (YYYY-MM-DD). Default: 2023-12-31 (provides 2-year validation period 2022-2023)."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Trade horizon in days (e.g., 5, 30). Used to auto-detect label column (label_{horizon}d). If not specified, will auto-detect from data."
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label column name (e.g., 'label_5d', 'label_30d'). If not specified, will auto-detect from data."
    )
    parser.add_argument(
        "--return-threshold",
        type=float,
        default=None,
        help="Return threshold used for labeling (as decimal, e.g., 0.05 for 5%%). Used for metadata tracking only."
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default=None,
        help=f"Feature set name (e.g., 'v1', 'v2'). If specified, automatically sets data directory and train config. Default: '{DEFAULT_FEATURE_SET}'"
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=None,
        help="Custom model output file path (e.g., 'models/my_custom_model.pkl'). If not specified, uses default naming based on feature set or 'xgb_classifier_selected_features.pkl'"
    )
    args = parser.parse_args()

    start_time = time.time()

    # Step 1.5: Handle feature set configuration
    global DATA_DIR, TRAIN_CFG, MODEL_OUT
    if args.feature_set:
        if not HAS_FEATURE_SET_MANAGER:
            raise ImportError("Feature set manager not available. Cannot use --feature-set argument.")
        if not feature_set_exists(args.feature_set):
            raise ValueError(f"Feature set '{args.feature_set}' does not exist. Use 'python src/manage_feature_sets.py list' to see available sets.")
        
        # Update paths based on feature set
        DATA_DIR = get_feature_set_data_path(args.feature_set)
        TRAIN_CFG = get_train_features_config_path(args.feature_set)
        
        # Update model output path to include feature set name (unless custom path specified)
        if args.model_output is None:
            MODEL_OUT = MODEL_DIR / f"xgb_classifier_selected_features_{args.feature_set}.pkl"
        
        print(f"\n=== USING FEATURE SET: {args.feature_set} ===")
        print(f"Data directory: {DATA_DIR}")
        print(f"Train config: {TRAIN_CFG}")
        print(f"Model output: {MODEL_OUT}")
        print("=" * 70)
    elif HAS_FEATURE_SET_MANAGER:
        # Use default feature set
        DATA_DIR = get_feature_set_data_path(DEFAULT_FEATURE_SET)
        TRAIN_CFG = get_train_features_config_path(DEFAULT_FEATURE_SET)
    
    # Handle custom model output path (overrides feature set naming)
    if args.model_output:
        # If relative path, assume it's in models/ directory
        if not Path(args.model_output).is_absolute():
            MODEL_OUT = MODEL_DIR / args.model_output
        else:
            MODEL_OUT = Path(args.model_output)
        # Ensure .pkl extension
        if not MODEL_OUT.suffix:
            MODEL_OUT = MODEL_OUT.with_suffix('.pkl')
        print(f"\n=== USING CUSTOM MODEL OUTPUT ===")
        print(f"Model output: {MODEL_OUT}")
        print("=" * 70)

    # Step 2: Load & prepare data
    print("Loading data...")
    df_all = load_data()
    
    # Step 2.5: Calculate labels based on horizon and return_threshold
    # Labels are now calculated during training, not during feature engineering
    # (label_future_return is imported at the top of the file)
    
    # Determine horizon and threshold
    horizon = args.horizon if args.horizon else 20  # Default to 20 if not specified
    return_threshold = args.return_threshold if args.return_threshold is not None else 0.15
    
    print(f"Calculating labels with horizon={horizon} days, threshold={return_threshold:.2%}")
    
    # Find close and high columns (case-insensitive)
    # Prefer 'close' over 'adj close' for consistency with 'high' (both split-adjusted only)
    close_col = None
    high_col = None
    for col in df_all.columns:
        # Prefer 'close' (split-adjusted) over 'adj close' (split+dividend adjusted)
        # This ensures consistency with 'high' column which is also split-adjusted only
        if col.lower() == 'close':
            close_col = col
        elif col.lower() == 'adj close' and close_col is None:
            # Fallback to adj close if regular close not found
            close_col = col
        if col.lower() == 'high':
            high_col = col
    
    if not close_col:
        raise ValueError("Could not find 'close' or 'adj close' column in data")
    if not high_col:
        raise ValueError("Could not find 'high' column in data")
    
    # Log which column is being used
    if close_col.lower() == 'adj close':
        print(f"Warning: Using 'adj close' instead of 'close' for labeling. This may cause inconsistencies with 'high' column.")
    else:
        print(f"Using '{close_col}' column for labeling (consistent with 'high' column)")
    
    # Ensure close and high columns are numeric (convert from string if needed)
    if df_all[close_col].dtype == 'object':
        print(f"Converting {close_col} column to numeric...")
        df_all[close_col] = pd.to_numeric(df_all[close_col], errors='coerce')
    if df_all[high_col].dtype == 'object':
        print(f"Converting {high_col} column to numeric...")
        df_all[high_col] = pd.to_numeric(df_all[high_col], errors='coerce')
    
    # Calculate labels using the labeling utility
    label_col = "training_label"  # Generic name, not horizon-specific
    df_all = label_future_return(
        df_all,
        close_col=close_col,
        high_col=high_col,
        horizon=horizon,
        threshold=return_threshold,
        label_name=label_col
    )
    
    print(f"Labels calculated and stored in column: {label_col}")
    
    X_all, y_all, all_feats = prepare(df_all, label_col)

    # Step 3: Load training feature flags
    if not TRAIN_CFG.exists():
        raise FileNotFoundError(f"Training config not found: {TRAIN_CFG}")
    train_cfg = yaml.safe_load(TRAIN_CFG.read_text(encoding="utf-8")) or {}
    flags = train_cfg.get("features", {})

    # Step 4: Determine which features are enabled
    enabled_feats = {name for name, flag in flags.items() if flag == 1}
    feats = [f for f in all_feats if f in enabled_feats]
    if not feats:
        raise ValueError("No features selected; check train_features.yaml")

    # Step 5: Slice X to only use selected features
    X = X_all[feats]
    y = y_all

    # Sanity checks
    print("\n=== SANITY CHECKS ===")
    print(f"X shape: {X.shape}")
    print("Label distribution (positive class %):")
    print(y.value_counts(normalize=True))
    print(f"Any NaNs in X? {X.isna().any().any()} (should be False - NaNs filled with 0.0)")
    print(f"Number of features: {len(feats)}")
    print(f"Feature names: {', '.join(feats[:10])}{'...' if len(feats) > 10 else ''}")
    print(f"Date range: {X.index.min()} to {X.index.max()}")
    print("=" * 70)

    # Step 6: Train/Validation/Test split by date
    dates = pd.to_datetime(X.index)
    # Allow override of split dates via CLI
    train_start_date = args.train_start if args.train_start else TRAIN_START
    train_end_date = args.train_end if args.train_end else TRAIN_END
    val_end_date = args.val_end if args.val_end else VAL_END
    
    train_cutoff = pd.to_datetime(train_end_date)
    val_cutoff = pd.to_datetime(val_end_date)
    
    # Build train mask: from train_start (or beginning) to train_end
    train_mask = dates <= train_cutoff
    if train_start_date:
        train_start = pd.to_datetime(train_start_date)
        train_mask = train_mask & (dates >= train_start)
    
    val_mask = (dates > train_cutoff) & (dates <= val_cutoff)
    test_mask = dates > val_cutoff
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Convert to float32 to reduce memory usage (50% reduction)
    # This helps prevent ArrayMemoryError during parallel hyperparameter tuning
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    print(f"\n=== DATA SPLITS ===")
    print(f"Training set:   {len(X_train):,} samples ({X_train.index.min()} to {X_train.index.max()})")
    print(f"Validation set: {len(X_val):,} samples ({X_val.index.min()} to {X_val.index.max()})")
    print(f"Test set:       {len(X_test):,} samples ({X_test.index.min()} to {X_test.index.max()})")
    print(f"Training positive class: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"Validation positive class: {y_val.sum():,} ({y_val.mean()*100:.2f}%)")
    print(f"Test positive class: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

    # Step 7: Baseline DummyClassifier
    print("\n=== TRAINING BASELINE MODELS ===")
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    dummy_proba = dummy.predict_proba(X_test)[:, 1]
    dummy_auc = roc_auc_score(y_test, dummy_proba)
    print(f"DummyClassifier Test AUC: {dummy_auc:.4f}")

    # Step 8: Baseline LogisticRegression
    # Note: XGBoost doesn't require feature scaling, so we train on raw features
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    print(f"LogisticRegression Test AUC: {lr_auc:.4f}")

    # Step 9: Compute scale_pos_weight for XGBoost
    pos = y_train.sum()
    neg = len(y_train) - pos
    base_scale_pos_weight = neg / pos
    
    # Allow tuning of class imbalance handling
    # Default: 1.0x (standard), can increase to 2.0x-3.0x to favor positive class more
    # Higher values = model more likely to predict positive class (trades)
    imbalance_multiplier = args.imbalance_multiplier
    scale_pos_weight = base_scale_pos_weight * imbalance_multiplier
    
    print(f"\nClass imbalance ratio (neg/pos): {base_scale_pos_weight:.2f}")
    if imbalance_multiplier != 1.0:
        print(f"Imbalance multiplier: {imbalance_multiplier:.2f}x")
        print(f"Adjusted scale_pos_weight: {scale_pos_weight:.2f}")

    # Step 10: Hyperparameter tuning or use defaults
    print("\n=== TRAINING XGBOOST CLASSIFIER ===")
    
    if args.tune:
        print(f"Performing hyperparameter tuning ({args.n_iter} iterations)...")
        
        # Optimize parallelization: During CV, use fewer threads per XGBoost model
        # so RandomizedSearchCV can parallelize better across folds/candidates
        # Get number of CPU cores
        n_cores = os.cpu_count() or 4
        
        # During hyperparameter search: Use 1-2 threads per XGBoost model
        # This allows RandomizedSearchCV to use more cores for parallelizing across candidates
        # After finding best params, we'll use full parallelization for final training
        xgb_threads_during_search = max(1, n_cores // 4)  # Use 1/4 of cores per model during search
        
        print(f"Parallelization: Using {xgb_threads_during_search} threads per XGBoost model during search")
        print(f"  (RandomizedSearchCV will parallelize across {n_cores} cores)")
        
        # Base parameters for hyperparameter search
        base_params = {
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'eval_metric': 'logloss',
            'tree_method': 'hist',  # Memory-efficient histogram method
            'n_jobs': xgb_threads_during_search  # Reduced for better CV parallelization
        }
        
        # Create base model
        base_model = XGBClassifier(**base_params)
        
        # Note: Early stopping doesn't work with RandomizedSearchCV
        # CV already provides validation, so we skip early stopping during tuning
        # Early stopping will be used when training the final model below
        
        # Choose hyperparameter space (fast mode uses smaller space)
        param_space = HYPERPARAMETER_SPACE_FAST if args.fast else HYPERPARAMETER_SPACE
        
        # Choose CV strategy
        if args.cv_folds is not None:
            # User-specified CV folds
            if args.cv:
                cv = TimeSeriesSplit(n_splits=args.cv_folds)
                print(f"Using TimeSeriesSplit cross-validation ({args.cv_folds} folds)")
            else:
                cv = args.cv_folds
                print(f"Using standard {args.cv_folds}-fold cross-validation")
        elif args.fast:
            # Fast mode: use 2 folds instead of 3
            if args.cv:
                cv = TimeSeriesSplit(n_splits=2)
                print("Using TimeSeriesSplit cross-validation (2 folds) [FAST MODE]")
            else:
                cv = 2
                print("Using standard 2-fold cross-validation [FAST MODE]")
        else:
            # Normal mode: use 3 folds
            if args.cv:
                cv = TimeSeriesSplit(n_splits=3)
                print("Using TimeSeriesSplit cross-validation (3 folds)")
            else:
                cv = 3
                print("Using standard 3-fold cross-validation")
        
        # Randomized search (no early stopping during CV)
        # verbose=2 shows progress for each fit, verbose=1 shows only completion
        total_fits = args.n_iter * (cv if isinstance(cv, int) else cv.n_splits)
        est_time = "5-10" if args.fast else "10-20"
        print(f"Starting hyperparameter search: {args.n_iter} candidates × {cv if isinstance(cv, int) else cv.n_splits} folds = {total_fits} total fits")
        if args.fast:
            print(f"FAST MODE: Reduced search space. This may take {est_time} minutes. Progress updates will appear below...\n")
        else:
            print(f"This may take {est_time} minutes. Progress updates will appear below...\n")
        
        search = RandomizedSearchCV(
            base_model,
            param_space,
            n_iter=args.n_iter,
            cv=cv,
            scoring='roc_auc',
            n_jobs=2,  # Limited to 2 workers to prevent memory errors (Windows spawn backend creates full copies)
            random_state=42,
            verbose=2  # Show progress for each fit
        )
        
        # Use only training set for hyperparameter tuning (CV will split internally)
        # We'll use the validation set for early stopping after finding best params
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        print(f"\nBest hyperparameters:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV score: {search.best_score_:.4f}")
        
        # Now retrain the best model on the full training set with early stopping on validation set
        print("\nRetraining best model on full training set with early stopping...")
        # Create a new model with the best parameters
        best_params = best_model.get_params()
        # Remove parameters that shouldn't be passed to constructor
        best_params.pop('n_estimators', None)  # We'll set this explicitly if needed
        best_params.pop('early_stopping_rounds', None)  # Handle separately
        
        # Use full parallelization for final training (no CV overhead)
        best_params['n_jobs'] = -1  # Use all cores for final training
        # Ensure tree_method is set (preserve from base_params if present, otherwise set to 'hist')
        if 'tree_method' not in best_params:
            best_params['tree_method'] = 'hist'  # Memory-efficient histogram method
        print("Retraining with full parallelization (all CPU cores)...")
        
        # Use early stopping on validation set if available
        # Note: In XGBoost 2.1.0+, early_stopping_rounds must be in constructor
        if not args.no_early_stop and len(X_val) > 0:
            # Set early_stopping_rounds in constructor (XGBoost 2.1.0+)
            best_params['early_stopping_rounds'] = args.early_stopping_rounds
            model = XGBClassifier(**best_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
            if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                print(f"Early stopping: best iteration {model.best_iteration} (score: {model.best_score:.4f})")
        else:
            # Still capture eval history for plots
            # Ensure tree_method is set (preserve from base_params if present, otherwise set to 'hist')
            if 'tree_method' not in best_params:
                best_params['tree_method'] = 'hist'  # Memory-efficient histogram method
            model = XGBClassifier(**best_params)
            if len(X_val) > 0:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
        
    else:
        print("Using default hyperparameters (use --tune for hyperparameter optimization)")
        
        # Default parameters (improved from original)
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            tree_method='hist',  # Memory-efficient histogram method
            n_jobs=-1
        )
        
        # Fit with early stopping if validation set exists
        # Note: In XGBoost 2.1.0+, early_stopping_rounds must be in constructor
        if not args.no_early_stop and len(X_val) > 0:
            print("Training with early stopping on validation set...")
            model.set_params(early_stopping_rounds=args.early_stopping_rounds)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
            if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                print(f"Stopped at {model.best_iteration} iterations (best score: {model.best_score:.4f})")
        else:
            # Still use eval_set to capture training history for plots
            if len(X_val) > 0:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)

    # Step 11: Evaluate on validation and test sets
    val_proba = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else None
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = model.predict(X_test)
    
    metrics = {}
    if val_proba is not None:
        val_pred = model.predict(X_val)
        metrics['validation'] = print_evaluation_metrics(y_val, val_pred, val_proba, "Validation")
    
    metrics['test'] = print_evaluation_metrics(y_test, test_pred, test_proba, "Test")
    
    # Compare to baselines
    test_auc = metrics['test']['roc_auc']
    print(f"\n=== MODEL COMPARISON ===")
    print(f"DummyClassifier AUC:     {dummy_auc:.4f}")
    print(f"LogisticRegression AUC:  {lr_auc:.4f}")
    print(f"XGBClassifier AUC:      {test_auc:.4f}")
    print(f"Improvement over LR:     {test_auc - lr_auc:+.4f}")

    # Step 12: Feature importance
    feature_importances = display_feature_importance(model, feats, top_n=20)
    
    # Step 12.5: Export all feature importances to CSV
    fi_csv_file = MODEL_DIR / "feature_importances_all.csv"
    fi_df = pd.DataFrame([
        {'feature': feat, 'importance': imp, 'rank': rank + 1}
        for rank, (feat, imp) in enumerate(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
    ])
    fi_df['importance_pct'] = (fi_df['importance'] / fi_df['importance'].sum() * 100).round(4)
    fi_df['cumulative_pct'] = fi_df['importance_pct'].cumsum().round(2)
    fi_df.to_csv(fi_csv_file, index=False)
    print(f"\nAll {len(fi_df)} feature importances exported to: {fi_csv_file}")
    
    # Step 12.5: Generate plots if requested
    plot_files = {}
    if args.plots:
        print("\n=== GENERATING PLOTS ===")
        if HAS_MATPLOTLIB:
            plot_files['training_curves'] = plot_training_curves(model, MODEL_DIR, "xgb")
            plot_files['feature_importance'] = plot_feature_importance_chart(model, feats, MODEL_DIR, top_n=20)
        else:
            print("Matplotlib not available. Install with: pip install matplotlib")

    # Step 13: Save the trained model, feature list, and metadata
    training_metadata = {
        'training_date': datetime.now().isoformat(),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)) if len(X_val) > 0 else 0,
        'test_samples': int(len(X_test)),
        'n_features': len(feats),
        'features': feats,
        'hyperparameters': model.get_params(),
        'metrics': metrics,
        'feature_importances': {k: float(v) for k, v in feature_importances.items()},
        'baseline_auc': {
            'dummy': float(dummy_auc),
            'logistic_regression': float(lr_auc)
        },
        'class_imbalance_ratio': float(scale_pos_weight),
        'train_date_range': [str(X_train.index.min()), str(X_train.index.max())],
        'test_date_range': [str(X_test.index.min()), str(X_test.index.max())],
        'plot_files': plot_files if plot_files else None,
        'return_threshold': args.return_threshold,  # Store return threshold for metadata
        'horizon': args.horizon,  # Store horizon for metadata
        'label_col': label_col,  # Store label column for metadata
        'feature_set': args.feature_set if hasattr(args, 'feature_set') and args.feature_set else DEFAULT_FEATURE_SET  # Store feature set used
    }
    
    # Add early stopping info if available
    if hasattr(model, 'best_iteration'):
        training_metadata['early_stopping'] = {
            'early_stopping_rounds': args.early_stopping_rounds,
            'best_iteration': int(model.best_iteration),
            'best_score': float(model.best_score) if hasattr(model, 'best_score') else None
        }
    
    # Save model with metadata
    # Note: XGBoost doesn't require feature scaling, so features are used as-is
    model_data = {
        "model": model,
        "features": feats,
        "metadata": training_metadata
    }
    joblib.dump(model_data, MODEL_OUT)
    
    # Also save metadata as JSON for easy inspection (update if SHAP was computed)
    metadata_file = MODEL_DIR / "training_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Model saved to: {MODEL_OUT}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Total training time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    # Step 14: Optional SHAP explanations (using new SHAP service)
    shap_artifacts_path = None
    if args.shap:
        print("\n=== COMPUTING SHAP EXPLANATIONS ===")
        try:
            # Import SHAP service (it's in the same src/ directory)
            from shap_service import SHAPService
            
            # Generate model ID from model filename (without extension)
            model_id = MODEL_OUT.stem
            
            # Initialize SHAP service
            shap_service = SHAPService()
            
            # Use validation set for SHAP (out-of-sample, more realistic)
            # Fall back to test set if validation is too small
            if len(X_val) >= 100:
                shap_data = X_val
                shap_labels = y_val
                data_split_name = "validation"
            elif len(X_test) >= 100:
                shap_data = X_test
                shap_labels = y_test
                data_split_name = "test"
            else:
                # Use training set if validation/test are too small (not ideal but better than nothing)
                shap_data = X_train
                shap_labels = y_train
                data_split_name = "training"
                print("Warning: Validation/test sets too small, using training set for SHAP (less ideal)")
            
            # Compute SHAP
            result = shap_service.compute_shap(
                model=model,
                X_data=shap_data,
                y_data=shap_labels,
                features=feats,
                model_id=model_id,
                sample_size=1000,  # Default sample size
                use_stratified=True,
                data_split=data_split_name
            )
            
            if result["success"]:
                shap_artifacts_path = result["artifacts_path"]
                print(f"✓ {result['message']}")
                print(f"  Data split used: {data_split_name}")
                print(f"  Samples computed: {result['metadata']['sample_size']}")
                
                # Store SHAP artifacts path in training metadata
                training_metadata['shap_artifacts_path'] = str(shap_artifacts_path)
                training_metadata['shap_metadata'] = result['metadata']
                
                # Update model data and save
                model_data['metadata'] = training_metadata
                joblib.dump(model_data, MODEL_OUT)
                
                # Update metadata JSON file
                with open(metadata_file, 'w') as f:
                    json.dump(training_metadata, f, indent=2)
            else:
                print(f"✗ SHAP computation failed: {result['message']}")
                
        except ImportError:
            print("SHAP service not available. Ensure shap_service.py is in the src/ directory.")
        except Exception as e:
            print(f"Error computing SHAP: {str(e)}")
            import traceback
            traceback.print_exc()

    # Step 15: Optional SHAP diagnostics (legacy, for backward compatibility)
    if args.diagnostics:
        print("\n=== SHAP DIAGNOSTICS ===")
        try:
            import shap
            # Sample up to 5k rows for SHAP to speed up computation
            sample_size = min(5000, X_train.shape[0])
            X_sample = X_train.sample(n=sample_size, random_state=42)
            shap_explainer = shap.TreeExplainer(model)
            shap_values = shap_explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            shap_fi = sorted(zip(feats, mean_abs_shap), key=lambda x: x[1], reverse=True)
            print(f"SHAP feature importances (mean absolute) for sample of {sample_size} rows:")
            print(f"{'Rank':<6} {'Feature':<30} {'SHAP Importance':<18}")
            print("-" * 54)
            for rank, (f, imp) in enumerate(shap_fi[:20], 1):
                print(f"{rank:<6} {f:<30} {imp:<18.6f}")
            
            # Export all SHAP importances to CSV
            shap_csv_file = MODEL_DIR / "shap_importances_all.csv"
            shap_df = pd.DataFrame([
                {'feature': feat, 'shap_importance': imp, 'rank': rank + 1}
                for rank, (feat, imp) in enumerate(shap_fi)
            ])
            shap_df['shap_importance_pct'] = (shap_df['shap_importance'] / shap_df['shap_importance'].sum() * 100).round(4)
            shap_df['cumulative_pct'] = shap_df['shap_importance_pct'].cumsum().round(2)
            shap_df.to_csv(shap_csv_file, index=False)
            print(f"\nAll {len(shap_df)} SHAP importances exported to: {shap_csv_file}")
            
            # Also save SHAP values to metadata and update metadata file
            shap_importances_dict = {feat: float(imp) for feat, imp in shap_fi}
            training_metadata['shap_importances'] = shap_importances_dict
            
            # Update the model data with SHAP importances
            model_data['metadata'] = training_metadata
            joblib.dump(model_data, MODEL_OUT)
            
            # Update metadata JSON file
            with open(metadata_file, 'w') as f:
                json.dump(training_metadata, f, indent=2)
            print(f"Metadata updated with SHAP importances: {metadata_file}")
            
        except ImportError:
            print("SHAP library not installed. Install with: pip install shap")


if __name__ == "__main__":
    main()
