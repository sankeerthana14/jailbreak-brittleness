import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\Users\SATI0004\Documents\jailbreak-brittleness\datasets")

for path in sorted(DATA_DIR.glob("*.parquet")):
    df = pd.read_parquet(path)

    print("=" * 80)
    print(path.name)
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    print(df.head(2))
    print()