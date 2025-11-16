#!/usr/bin/env python3
"""
train_model.py

Train and evaluate an XGBoost classifier to predict multi-day positive returns,
now loading feature data from Parquet files.

This script:
  1. Loads all labeled feature Parquet files from `data/features_labeled/`
  2. Filters features based on `config/train_features.yaml`
  3. Drops NaNs and raw‐price/forward‐return columns to prevent leakage
  4. Splits data into train/test by date
  5. Trains baseline DummyClassifier & LogisticRegression (with scaling)
  6. Trains an XGBClassifier with class-imbalance handling
  7. Evaluates performance (ROC AUC, classification report)
  8. Saves the trained model and feature list to `models/`
  9. Optionally computes feature importances & SHAP diagnostics
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from typing import Tuple, List

# ——— CONFIGURATION ————————————————————————————————————————————————
PROJECT_ROOT = Path.cwd()
DATA_DIR     = PROJECT_ROOT / "data" / "features_labeled"  # expects .parquet files
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_OUT    = MODEL_DIR / "xgb_classifier_selected_features.pkl"
TRAIN_END    = "2022-12-31"      # cut-off date (inclusive) for training
LABEL_COL    = "label_5d"        # target column name
TRAIN_CFG    = PROJECT_ROOT / "config" / "train_features.yaml"
# —————————————————————————————————————————————————————————————————————————

def load_data() -> pd.DataFrame:
    """
    Load all ticker feature Parquet files into one concatenated DataFrame.

    Each Parquet in DATA_DIR must have a datetime index and feature columns.
    Adds a 'ticker' column to identify the source symbol.

    Raises:
        FileNotFoundError: if no Parquet files are found.

    Returns:
        DataFrame: concatenated data from all tickers.
    """
    parts: List[pd.DataFrame] = []
    for f in DATA_DIR.glob("*.parquet"):
        df = pd.read_parquet(f)
        df.index.name = "date"
        df["ticker"] = f.stem
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No Parquet files found in {DATA_DIR}")
    return pd.concat(parts, axis=0)


def prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Clean and split the raw DataFrame into X (features) and y (target).

    - Drops rows with missing values.
    - Removes forward-return columns (to prevent leakage).
    - Removes raw price columns: open/high/low/close/adj close.
    - Leaves only label and feature columns.

    Args:
        df: DataFrame with LABEL_COL, raw prices, and feature columns.

    Returns:
        X: DataFrame of selected feature columns.
        y: Series of the target label.
        feats: List of feature column names used.
    """
    # 1) Replace infinities with NaN, then drop rows with missing values
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna().copy()

    # 2) Candidate feature names (exclude label & ticker)
    feats = [c for c in df_clean.columns if c not in [LABEL_COL, "ticker"]]

    # 3) Exclude any forward-return columns (labels)
    forward_return_cols = {"5d_return", "10d_return"}
    feats = [c for c in feats if c not in forward_return_cols]

    # 4) Exclude raw price columns to avoid leakage
    raw_price_cols = {"open", "high", "low", "close", "adj close"}
    feats = [c for c in feats if c not in raw_price_cols]

    # 5) Slice out X and y
    X = df_clean[feats].copy()
    y = df_clean[LABEL_COL]
    
    # 6) Clip extreme values to prevent overflow (clip to reasonable range)
    # Clip to ±1e6 to prevent overflow issues while preserving signal
    X = X.clip(lower=-1e6, upper=1e6)
    
    # 7) Final check: ensure no infinities or NaNs remain
    if X.isin([np.inf, -np.inf]).any().any():
        print("Warning: Found infinities after cleaning, replacing with NaN")
        X = X.replace([np.inf, -np.inf], np.nan)
    
    if X.isna().any().any():
        print("Warning: Found NaNs after cleaning, dropping rows")
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]

    return X, y, feats


def main() -> None:
    """
    Main routine to train and evaluate the model.

    Workflow:
      1. Parse command-line arguments
      2. Load all data
      3. Prepare X, y and all_feats list
      4. Load train_features.yaml to pick enabled features
      5. Filter X by those features
      6. Print sanity checks
      7. Split into train/test by TRAIN_END date
      8. Fit DummyClassifier & LogisticRegression (with scaling) baselines
      9. Fit XGBClassifier with class-imbalance handling
     10. Evaluate and print metrics
     11. Serialize model + feature list
     12. Optionally compute feature importances & SHAP diagnostics on a sample
    """
    # Step 1: Parse CLI options
    parser = argparse.ArgumentParser(description="Train model with optional diagnostics")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Compute and print feature importances and SHAP diagnostics"
    )
    args = parser.parse_args()

    # Step 2: Load & prepare data
    df_all = load_data()
    X_all, y_all, all_feats = prepare(df_all)

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
    print("=== SANITY CHECKS ===")
    print("X shape:", X.shape)
    print("Label distribution (positive class %):")
    print(y.value_counts(normalize=True))
    print("Any NaNs in X? ", X.isna().any().any())
    print("Number of features:", len(feats))
    print("Feature names:", ", ".join(feats))
    print("Date range:", X.index.min(), "to", X.index.max())
    print("======================\n")

    # Step 6: Train/test split by date
    dates = pd.to_datetime(X.index)
    cutoff = pd.to_datetime(TRAIN_END)
    train_mask = dates <= cutoff
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[~train_mask], y[~train_mask]

    # Step 7: Baseline DummyClassifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_proba = dummy.predict_proba(X_test)[:, 1]
    print("DummyClassifier AUC:", roc_auc_score(y_test, dummy_proba))

    # Step 8: Baseline LogisticRegression (with scaling)
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    )
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    print("LogisticRegression AUC:", roc_auc_score(y_test, lr_proba))

    # Step 9: Compute scale_pos_weight for XGBoost
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos

    # Step 10: Train XGBoost classifier
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # Step 11: Evaluate on test set
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"XGBClassifier Test ROC AUC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    # Step 12: Save the trained model and feature list
    joblib.dump({"model": model, "features": feats}, MODEL_OUT)
    print(f"\nModel and feature list saved to: {MODEL_OUT}")

    # Step 13: Optional diagnostics
    if args.diagnostics:
        print("\n—— FEATURE IMPORTANCE DIAGNOSTICS ——")
        # 13.1 Model-derived importances
        importances = model.feature_importances_
        fi = sorted(zip(feats, importances), key=lambda x: x[1], reverse=True)
        print("Model feature importances:")
        for f, imp in fi:
            print(f"  {f}: {imp:.4f}")

        # 13.2 SHAP-based importances (mean absolute) on a sample
        try:
            import shap
            # Sample up to 10k rows for SHAP to speed up computation
            sample_size = min(10000, X_train.shape[0])
            X_sample = X_train.sample(n=sample_size, random_state=42)
            shap_explainer = shap.TreeExplainer(model)
            shap_values = shap_explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            shap_fi = sorted(zip(feats, mean_abs_shap), key=lambda x: x[1], reverse=True)
            print(f"\nSHAP feature importances (mean absolute) for sample of {sample_size} rows:")
            for f, imp in shap_fi:
                print(f"  {f}: {imp:.4f}")
        except ImportError:
            print("SHAP library not installed; skipping SHAP diagnostics.")


if __name__ == "__main__":
    main()
