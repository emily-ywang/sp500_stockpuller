"""
inspect_parquet.py
Prints a summary of every parquet file found under the project directory

Usage:
  python inspect_parquet.py            # all parquet files
  python inspect_parquet.py test       # only files under test/
  python inspect_parquet.py processed/prices_adj_close.parquet  # single file
"""

import sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent


def summarize(path):
    df = pd.read_parquet(path)

    print(f"\n{path.relative_to(BASE)}")
    print(f"Shape:{df.shape[0]:,} rows x {df.shape[1]} cols")

    if pd.api.types.is_datetime64_any_dtype(df.index):
        print(f"Dates:{df.index.min().date()} to {df.index.max().date()}")

    print(f"Columns:{list(df.columns[:10])}" + (" ..." if len(df.columns) > 10 else ""))

    missing = df.isna().sum().sum()
    if missing:
        print(f"Missing: {missing:,} values")
    print(f"\n{df.head(5).to_string()}")


if __name__ == "__main__":
    target = BASE / sys.argv[1] if len(sys.argv) > 1 else BASE
    paths = [target] if target.is_file() else sorted(target.rglob("*.parquet"))

    for p in paths:
        summarize(p)
