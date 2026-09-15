import glob

import pandas as pd

files = sorted(glob.glob("/data_vtlax/datasets/wine_rack_tactile/data/chunk-000/*.parquet"))
print("parquet files:", len(files))
total = 0
eps_counts = {}
for f in files:
    df = pd.read_parquet(f, columns=["episode_index"])
    for v, c in df["episode_index"].value_counts().items():
        eps_counts[int(v)] = eps_counts.get(int(v), 0) + int(c)
    total += len(df)
print("total rows:", total)
print("distinct episode_index values:", len(eps_counts))
for k in sorted(eps_counts):
    print(f"  ep {k}: {eps_counts[k]} frames")
