import pandas as pd
import numpy as np
import pytest

from utils.labeling import label_future_return

def test_label_future_return_basic():
    # close increases linearly, so 5-day return always > 0
    data = {'close': list(range(1, 11))}
    df = pd.DataFrame(data)
    out = label_future_return(df, horizon=5, threshold=0.0, label_name='label_5d')
    # First 5 rows should be 1, last 5 rows cannot be labeled (shift produces NaN >0 → False→0)
    expected = pd.Series([1]*5 + [0]*5, name='label_5d', index=df.index)
    pd.testing.assert_series_equal(out['label_5d'], expected)

def test_label_future_return_threshold():
    # custom threshold
    data = {'close': [100, 102, 104, 106, 108, 110]}
    df = pd.DataFrame(data)
    # 2-day horizon, threshold .03 (3%)
    out = label_future_return(df, horizon=2, threshold=0.03, label_name='lbl2')
    # returns: index0=(104/100-1)=.04>thr→1; idx1=(106/102-1)=.0392>thr→1; idx2=(108/104-1)=.0385→1; idx3=(110/106-1)=.0377→1; last2 idx4/5→0
    expected = pd.Series([1,1,1,1,0,0], name='lbl2', index=df.index)
    pd.testing.assert_series_equal(out['lbl2'], expected)

def test_label_raises_if_no_close():
    df = pd.DataFrame({'price': [1,2,3]})
    with pytest.raises(KeyError):
        label_future_return(df, close_col='close_missing')
