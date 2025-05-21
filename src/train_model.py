#!/usr/bin/env python3
"""
train_model.py

Train and evaluate an XGBoost classifier to predict multi-day positive returns.

This script:
  1. Loads all labeled feature CSVs from `data/features_labeled/`
  2. Filters features based on `config/train_features.yaml`
  3. Drops any NaNs and raw‐price/forward‐return columns to avoid leakage
  4. Splits data into train/test by date
  5. Runs baseline Dummy and LogisticRegression models
  6. Trains an XGBoost classifier (with class‐imbalance handling)
  7. Evaluates performance (ROC AUC, classification report)
  8. Saves the trained model and feature list to `models/`

Usage:
    python src/train_model.py
"""

import yaml
import pandas as pd
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from typing import Tuple, List

# ——— CONFIGURATION ————————————————————————————————————————————————
PROJECT_ROOT = Path.cwd()
DATA_DIR     = PROJECT_ROOT / "data" / "features_labeled"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)  # ensure models/ exists
MODEL_OUT    = MODEL_DIR / "xgb_classifier_selected_features.pkl"
TRAIN_END    = "2022-12-31"      # cut-off date (inclusive) for training
LABEL_COL    = "label_5d"        # target column name in CSVs
TRAIN_CFG    = PROJECT_ROOT / "config" / "train_features.yaml"
# —————————————————————————————————————————————————————————————————————————


def load_data() -> pd.DataFrame:
    """
    Load all ticker feature CSVs into one concatenated DataFrame.

    Each CSV in DATA_DIR must have a datetime index and feature columns.
    Adds a 'ticker' column to identify the source symbol.

    Raises:
        FileNotFoundError: if no CSV files are found.

    Returns:
        DataFrame: concatenated data from all tickers.
    """
    parts: List[pd.DataFrame] = []
    for f in DATA_DIR.glob("*.csv"):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df["ticker"] = f.stem
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    return pd.concat(parts, axis=0)


def prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Clean and split the raw DataFrame into X (features) and y (target).

    - Drops any rows containing NaNs.
    - Removes forward-return columns (to prevent leakage).
    - Removes raw price columns: open/high/low/close/adj close.
    - Leaves only label and feature columns.

    Args:
        df: DataFrame with columns including LABEL_COL, raw prices, and features.

    Returns:
        X: DataFrame of selected feature columns.
        y: Series of the target label.
        feats: List of feature column names used.
    """
    # 1) Drop rows with missing values
    df_clean = df.dropna().copy()

    # 2) Candidate feature names (exclude label & ticker)
    feats = [c for c in df_clean.columns if c not in [LABEL_COL, "ticker"]]

    # 3) Remove any forward-return columns
    feats = [c for c in feats if "_return" not in c]

    # 4) Remove raw price columns to avoid leakage
    raw_price_cols = {"open", "high", "low", "close", "adj close"}
    feats = [c for c in feats if c not in raw_price_cols]

    # 5) Slice out X and y
    X = df_clean[feats]
    y = df_clean[LABEL_COL]

    return X, y, feats


def main() -> None:
    """
    Main routine to train and evaluate the model.

    Workflow:
      1. Load all data
      2. Prepare X, y and list of all features
      3. Load train_features.yaml to pick enabled features
      4. Filter X by those features
      5. Print sanity checks
      6. Split into train/test by TRAIN_END date
      7. Fit DummyClassifier & LogisticRegression baselines
      8. Fit XGBClassifier with class-imbalance handling
      9. Evaluate and print metrics
     10. Serialize model + feature list
    """
    # — Step 1: Load & prepare data
    df_all = load_data()
    X_all, y_all, all_feats = prepare(df_all)

    # — Step 2: Load training feature flags
    if not TRAIN_CFG.exists():
        raise FileNotFoundError(f"Training config not found: {TRAIN_CFG}")
    train_cfg = yaml.safe_load(TRAIN_CFG.read_text(encoding="utf-8")) or {}
    flags = train_cfg.get("features", {})

    # — Step 3: Determine which features are enabled
    enabled_feats = {name for name, flag in flags.items() if flag == 1}
    feats = [f for f in all_feats if f in enabled_feats]
    if not feats:
        raise ValueError("No features selected; check train_features.yaml")

    # — Step 4: Slice X to only use selected features
    X = X_all[feats]
    y = y_all

    # — Sanity checks before training
    print("=== SANITY CHECKS ===")
    print("X shape:", X.shape)
    print("Label distribution:\n", y.value_counts(normalize=True))
    print("Any NaNs in X? ", X.isna().any().any())
    print("Number of features:", len(feats))
    print("Feature names:", ", ".join(feats))
    print("Date range:", X.index.min(), "to", X.index.max())
    print("======================\n")

    # — Step 5: Train/test split by date
    dates = pd.to_datetime(X.index)
    cutoff = pd.to_datetime(TRAIN_END)
    train_mask = dates <= cutoff
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[~train_mask], y[~train_mask]

    # — Step 6: Baseline DummyClassifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_proba = dummy.predict_proba(X_test)[:, 1]
    print("DummyClassifier AUC:", roc_auc_score(y_test, dummy_proba))

    # — Step 7: Baseline LogisticRegression
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    print("LogisticRegression AUC:", roc_auc_score(y_test, lr_proba))

    # — Step 8: Compute scale_pos_weight for XGBoost
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos

    # — Step 9: Train XGBoost classifier
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # — Step 10: Evaluate on test set
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"XGBClassifier Test ROC AUC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    # — Step 11: Save the trained model and feature list
    joblib.dump({"model": model, "features": feats}, MODEL_OUT)
    print(f"\nModel and feature list saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
