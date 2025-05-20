# src/train_model.py

#!/usr/bin/env python3
"""
train_model.py

Train & evaluate an XGBoost classifier to predict 5-day positive returns.
Uses a separate `config/train_features.yaml` to select which features to include.
Excludes forward-looking return and raw price features to avoid leakage.
Includes sanity checks and baseline models for comparison.
"""

import yaml
import pandas as pd
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# —— CONFIGURATION —— #
PROJECT_ROOT = Path.cwd()
DATA_DIR     = PROJECT_ROOT / "data" / "features_labeled"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_OUT    = MODEL_DIR / "xgb_classifier_selected_features.pkl"
TRAIN_END    = "2022-12-31"  # last date for training (YYYY-MM-DD)
LABEL_COL    = "label_5d"
TRAIN_CFG    = PROJECT_ROOT / "config" / "train_features.yaml"

def load_data():
    """
    Load all ticker CSVs from DATA_DIR into a single DataFrame.
    """
    parts = []
    for f in DATA_DIR.glob("*.csv"):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df["ticker"] = f.stem
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    return pd.concat(parts, axis=0)

def prepare(df):
    """
    Drop rows with any NaNs and split into features X and target y.
    Automatically drops forward-return columns and raw price columns.
    """
    df_clean = df.dropna().copy()
    # all candidate features
    feats = [c for c in df_clean.columns if c not in [LABEL_COL, "ticker"]]
    # drop leakage of forward returns
    feats = [c for c in feats if "_return" not in c]
    # drop raw price columns
    raw_price_cols = {"open", "high", "low", "close", "adj close"}
    feats = [c for c in feats if c not in raw_price_cols]
    # split
    X = df_clean[feats]
    y = df_clean[LABEL_COL]
    return X, y, feats

def main():
    # 1. Load & prepare data
    df_all = load_data()
    X, y, all_feats = prepare(df_all)

    # 2. Read training feature flags
    if not TRAIN_CFG.exists():
        raise FileNotFoundError(f"{TRAIN_CFG} not found")
    train_cfg = yaml.safe_load(TRAIN_CFG.read_text(encoding="utf-8"))
    enabled = {
        name for name, flag in train_cfg.get("features", {}).items() if flag == 1
    }
    # filter features
    feats = [f for f in all_feats if f in enabled]
    if not feats:
        raise ValueError("No features selected for training. Check train_features.yaml")
    X = X[feats]

    # —— SANITY CHECKS —— #
    print("=== SANITY CHECKS ===")
    print("X shape:", X.shape)
    print("y distribution:\n", y.value_counts(normalize=True))
    print("Any NaNs in X? ", X.isna().values.any())
    print("Number of features:", len(feats))
    print("Feature names:", ", ".join(feats))
    print("Date range:", X.index.min(), "to", X.index.max())
    print("======================\n")

    # 3. Train/test split by date using boolean mask
    dates = pd.to_datetime(X.index)
    cutoff = pd.to_datetime(TRAIN_END)
    mask = dates <= cutoff
    X_train, y_train = X[mask], y[mask]
    X_test,  y_test  = X[~mask], y[~mask]

    # —— BASELINE MODELS —— #
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_proba = dummy.predict_proba(X_test)[:, 1]
    print("DummyClassifier AUC:", roc_auc_score(y_test, dummy_proba))

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    print("LogisticRegression AUC:", roc_auc_score(y_test, lr_proba))

    # 4. Handle class imbalance for XGBoost
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    # 5. Instantiate and train XGBoost classifier
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # 6. Evaluate on the test set
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"XGBClassifier Test ROC AUC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    # 7. Save model and feature list
    joblib.dump({"model": model, "features": feats}, MODEL_OUT)
    print(f"\nModel and feature list saved to: {MODEL_OUT}")

if __name__ == "__main__":
    main()
