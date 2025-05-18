import glob
import pandas as pd

def audit_file(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    nat = df.index.isnull().sum()
    dup = df.index.duplicated().sum()
    mono = df.index.is_monotonic_increasing
    dtypes = {col: str(dt) for col, dt in df.dtypes.items()}
    print(f"{path:<30} | NaT={nat:2d} | Dups={dup:2d} | Monotonic={mono} | dtypes={dtypes}")

if __name__ == "__main__":
    for f in glob.glob("data/clean/*.csv"):
        audit_file(f)
