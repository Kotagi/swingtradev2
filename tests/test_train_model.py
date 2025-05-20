import pandas as pd
import json
import subprocess
import sys
from pathlib import Path

def test_train_model_smoke(tmp_path):
    # 1. Synthetic features directory
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    # Ensure both classes are in the first 20 days (train fold)
    labels = [1]*15 + [0]*15
    df = pd.DataFrame({
        '5d_return':  [0.1]*30,
        '10d_return': [0.2]*30,
        'atr':        [1.0]*30,
        'bb_width':   [0.1]*30,
        'ema_cross':  [0.0]*30,
        'obv':        [100.0]*30,
        'rsi':        [50.0]*30,
        'label_5d':   labels
    }, index=dates)
    df.to_csv(feat_dir / "AAPL.csv")

    # 2. Single-fold splits.json (train first 20 days, test next 5)
    splits = {
        "fold_0": {
            "train": [d.strftime("%Y-%m-%d") for d in dates[:20]],
            "test":  [d.strftime("%Y-%m-%d") for d in dates[20:25]]
        }
    }
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits))

    # 3. Define output paths
    model_out  = tmp_path / "model.pkl"
    report_out = tmp_path / "report.csv"

    # 4. Run the training script
    script = Path(__file__).parents[1] / "scripts" / "train_model.py"
    cmd = [
        sys.executable, str(script),
        "--features-dir", str(feat_dir),
        "--splits",       str(splits_path),
        "--model-out",    str(model_out),
        "--report-out",   str(report_out)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    # 5. Check that files were created
    assert model_out.exists(),  "Model file not created"
    assert report_out.exists(), "Report file not created"

    # 6. Sanity-check model predictions
    import joblib
    model = joblib.load(str(model_out))
    sample = df.drop(columns=['label_5d']).iloc[:5]
    preds = model.predict(sample)
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})
