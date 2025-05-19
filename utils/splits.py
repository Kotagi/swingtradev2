import pandas as pd
import json
from typing import List, Tuple

def walk_forward_splits(dates: pd.DatetimeIndex,
                        train_days: int,
                        test_days: int,
                        step_days: int
                       ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Return a list of (train_idx, test_idx) pairs,
    where both are subsets of `dates`, advancing by `step_days` each fold.
    """
    splits = []
    start = 0
    N = len(dates)
    while True:
        train_start = start
        train_end = train_start + train_days
        test_end = train_end + test_days
        if test_end > N:
            break
        train_idx = dates[train_start:train_end]
        test_idx  = dates[train_end:test_end]
        splits.append((train_idx, test_idx))
        start += step_days
    return splits

def save_splits(splits, path: str):
    """Serialize splits to JSON of {fold: {train: [...], test: [...]}}."""
    out = {}
    for i, (tr, te) in enumerate(splits):
        out[f"fold_{i}"] = {
            "train": [d.strftime("%Y-%m-%d") for d in tr],
            "test":  [d.strftime("%Y-%m-%d") for d in te],
        }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
