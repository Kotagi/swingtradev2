# ML Model Training Overview

## Model Choice: XGBoost Classifier

**What it is:**  
XGBoost (`XGBClassifier`) is a gradient-boosted tree ensemble algorithm wrapped in an scikit-learn–compatible interface. It builds multiple decision trees sequentially, where each new tree corrects errors of the previous ensemble.

### Why XGBoost?
- **High performance:** Efficient C++ implementation optimized for speed and memory.  
- **Handles mixed data:** Works well with numeric indicators of varying scales (RSI, ATR, OBV, etc.).  
- **Built-in regularization:** Reduces overfitting via tree- and leaf-level penalties.  
- **Missing values:** Automatically learns best direction for missing data.  
- **Feature importance:** Provides feature ranking to identify your strongest indicators.

## Handling Class Imbalance
Short-term returns tend to have less than 50% positive days. We’ll use `scale_pos_weight` in `XGBClassifier` to weight the positive class:

```python
from xgboost import XGBClassifier
# Compute ratio: number of negative samples / number of positive samples
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
```

## Pipeline Steps

1. **Load & Prepare Data**  
   - Concatenate all tickers from `data/features_labeled/*.csv`.  
   - Drop rows with any NaNs to ensure full feature vectors.  
   - Define features (`X`) as all numeric columns except `label_5d` and metadata.  
   - Define target (`y`) as the binary `label_5d`.

2. **Train/Test Split**  
   - Use a time-based split: train on data up to `TRAIN_END` (e.g., `'2022-12-31'`), test after.  
   - Optionally use `TimeSeriesSplit` for cross-validation.

3. **Train Model**  
   - Instantiate `XGBClassifier` with tuned hyperparameters and `scale_pos_weight`.  
   - Fit on `(X_train, y_train)`.

4. **Evaluate Performance**  
   - Compute ROC AUC and classification report on the test set.  
   - Sweep a probability threshold to find the best backtest P&L.

5. **Save Model**  
   - Serialize the trained model and feature list with `joblib.dump`.

6. **Backtest Predictions**  
   - Reload model; compute `model.predict_proba(X)[:,1]`.  
   - Generate a boolean `pred_signal = proba > threshold`.  
   - Feed `pred_signal` into `backtest_signals()` to get real-world P&L metrics.

## Code: `src/train_model.py`

```python
#!/usr/bin/env python3
"""
train_model.py

Train & evaluate an XGBoost classifier to predict 5-day positive returns.
"""

import pandas as pd
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

# —— CONFIGURATION —— #
DATA_DIR    = Path.cwd() / "data" / "features_labeled"
MODEL_DIR   = Path.cwd() / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_OUT   = MODEL_DIR / "xgb_classifier.pkl"
TRAIN_END   = "2022-12-31"  # last date for training
LABEL_COL   = "label_5d"

def load_data():
    # Load all tickers into a single DataFrame
    parts = []
    for f in DATA_DIR.glob("*.csv"):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df["ticker"] = f.stem
        parts.append(df)
    return pd.concat(parts)

def prepare(Xy):
    # Drop NaNs
    df = Xy.dropna().copy()
    # Separate features and target
    feats = [c for c in df.columns if c not in [LABEL_COL, "ticker"]]
    X = df[feats]
    y = df[LABEL_COL]
    return X, y, feats

def main():
    df_all = load_data()
    X, y, feats = prepare(df_all)

    # Train/test split
    X_train, y_train = X.loc[:TRAIN_END], y.loc[:TRAIN_END]
    X_test,  y_test  = X.loc[TRAIN_END:], y.loc[TRAIN_END:]

    # Compute imbalance ratio
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    # Instantiate XGBoost classifier
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate
    y_proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC AUC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    # Save model and feature names
    joblib.dump({"model": model, "features": feats}, MODEL_OUT)
    print(f"Model and feature list saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
```

## How to Run

1. **Install dependencies**:
   ```bash
   pip install pandas scikit-learn xgboost joblib
   ```
2. **Run the script**:
   ```bash
   python src/train_model.py
   ```
3. **Review output**:
   - ROC AUC and classification report.
   - Model saved at `models/xgb_classifier.pkl`.

## Next Steps
- **Threshold tuning:** Sweep `y_proba > threshold` to maximize backtest P&L.
- **Cross-validation:** Use `TimeSeriesSplit` or expanding window CV.
- **Hyperparameter search:** GridSearchCV or Bayesian optimization on `n_estimators`, `max_depth`, `learning_rate`, etc.
- **Backtest integration:** Load the saved model, generate signals, and feed into `backtest_signals` for real-world performance.

