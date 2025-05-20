#!/usr/bin/env python3
import argparse
import glob
import pandas as pd
import joblib
from pathlib import Path
import json

from utils.splits import walk_forward_splits

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the engineered feature columns to use for training and prediction
FEATURE_COLS = [
    '5d_return',
    '10d_return',
    'atr',
    'bb_width',
    'ema_cross',
    'obv',
    'rsi'
]

def load_and_clean(features_dir):
    """
    Load all feature CSVs and drop rows with any missing engineered features or label.
    """
    all_files = glob.glob(f"{features_dir}/*.csv")
    dfs = []
    tickers = []
    for f in all_files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        ticker = Path(f).stem
        df['ticker'] = ticker
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    # Drop rows where features or label are NaN
    df = df.dropna(subset=FEATURE_COLS + ['label_5d'])
    return df

def train_and_evaluate(df, splits, model_path, report_path):
    """
    Train & evaluate an XGBoost classifier using walk-forward splits,
    then retrain on the full dataset and save the final model.
    """
    metrics = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        # Select by date index
        train_df = df[df.index.isin(train_idx)]
        test_df  = df[df.index.isin(test_idx)]

        X_train = train_df[FEATURE_COLS]
        y_train = train_df['label_5d']
        X_test  = test_df[FEATURE_COLS]
        y_test  = test_df['label_5d']

        model = XGBClassifier(eval_metric='logloss')
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics.append({
            'fold': fold,
            'accuracy':  accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall':    recall_score(y_test, preds, zero_division=0),
            'f1':        f1_score(y_test, preds, zero_division=0)
        })

    # Save fold metrics
    report_df = pd.DataFrame(metrics)
    report_df.to_csv(report_path, index=False)
    print(f"Saved training metrics to {report_path}")

    # Retrain on full dataset
    final_model = XGBClassifier(eval_metric='logloss')
    final_model.fit(df[FEATURE_COLS], df['label_5d'])
    joblib.dump(final_model, model_path)
    print(f"Saved final model to {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost on engineered features")
    parser.add_argument("--features-dir", required=True,
                        help="Directory of labeled feature CSVs")
    parser.add_argument("--splits", required=True,
                        help="Path to splits.json")
    parser.add_argument("--model-out", required=True,
                        help="Output path for trained model (.pkl)")
    parser.add_argument("--report-out", required=True,
                        help="Output path for train metrics CSV")
    args = parser.parse_args()

    # 1. Load & clean data
    df = load_and_clean(args.features_dir)

    # 2. Load splits.json
    with open(args.splits) as f:
        split_json = json.load(f)
    splits = [
        (pd.to_datetime(v['train']), pd.to_datetime(v['test']))
        for v in split_json.values()
    ]

    # 3. Train and evaluate
    train_and_evaluate(df, splits, args.model_out, args.report_out)

if __name__ == "__main__":
    main()
