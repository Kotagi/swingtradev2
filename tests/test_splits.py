import pandas as pd
import pytest
from utils.splits import walk_forward_splits

def test_walk_forward_basic():
    # 100 daily dates
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # 20 train, 10 test, step 10 -> expected folds = floor((100 - 20 - 10)/10) + 1 = 8
    splits = walk_forward_splits(dates, train_days=20, test_days=10, step_days=10)
    assert len(splits) == 8

    # Check first fold indices
    tr0, te0 = splits[0]
    assert tr0[0] == dates[0]
    assert tr0[-1] == dates[19]
    assert te0[0] == dates[20]
    assert te0[-1] == dates[29]

    # Ensure no overlap in any fold
    for tr, te in splits:
        assert set(tr).isdisjoint(set(te))

def test_walk_forward_edge():
    # 25 daily dates
    dates = pd.date_range("2020-01-01", periods=25, freq="D")
    # 10 train, 5 test, step 5 -> expected 3 folds:
    # fold0: train[0:10], test[10:15]
    # fold1: train[5:15], test[15:20]
    # fold2: train[10:20], test[20:25]
    splits = walk_forward_splits(dates, train_days=10, test_days=5, step_days=5)
    assert len(splits) == 3

    # Check last fold boundaries
    tr2, te2 = splits[-1]
    assert tr2[0] == dates[10]
    assert tr2[-1] == dates[19]
    assert te2[0] == dates[20]
    assert te2[-1] == dates[24]

def test_walk_forward_no_folds():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    # Not enough data for a single fold
    splits = walk_forward_splits(dates, train_days=8, test_days=5, step_days=1)
    assert splits == []
