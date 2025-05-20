import pandas as pd

def label_future_return(df: pd.DataFrame, 
                        close_col: str = 'close',
                        horizon: int = 5,
                        threshold: float = 0.0,
                        label_name: str = 'label_5d') -> pd.DataFrame:
    """
    Adds a binary label: 1 if forward (horizon)-day return > threshold, else 0.
    - horizon: number of days to look ahead
    - threshold: e.g. 0.0 for any positive return
    - label_name: name for the output column
    """
    if close_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{close_col}' column")
    close = df[close_col]
    future_ret = close.shift(-horizon) / close - 1
    df[label_name] = (future_ret > threshold).astype(int)
    return df
