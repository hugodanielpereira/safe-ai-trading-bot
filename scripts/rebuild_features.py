# scripts/rebuild_features.py
import sys, glob, os, pandas as pd
from app.signals import make_features

sym, interval, out = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(glob.glob(f"data/{sym}/{interval}/*.csv"))
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
# normaliza tempo se existir
for c in ("close_time","open_time","time","timestamp"):
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        break
fe = make_features(df.rename(columns={"Close":"close","High":"high","Low":"low","Volume":"volume"}))
os.makedirs(os.path.dirname(out), exist_ok=True)
fe.to_csv(out, index=False)
print(f"saved {len(fe)} rows -> {out}")