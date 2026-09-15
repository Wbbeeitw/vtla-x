import glob
import os

import pandas as pd

files = sorted(glob.glob("/data_vtlax/datasets/wine_rack_tactile/data/chunk-000/*.parquet"))
print("parquet files:", len(files))
tot = 0
for f in files:
    df = pd.read_parquet(f, columns=["episode_index"])
    vc = df["episode_index"].value_counts().sort_index()
    tot += len(df)
    print(os.path.basename(f), "rows:", len(df), "eps:", dict(vc))
print("total rows:", tot)
