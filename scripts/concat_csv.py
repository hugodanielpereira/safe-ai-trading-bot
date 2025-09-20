# scripts/concat_csv.py
from __future__ import annotations
import argparse, sys
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="ficheiro de saída")
    ap.add_argument("inputs", nargs="+", help="CSV(s) de entrada (klines)")
    args = ap.parse_args()

    if not args.inputs:
        print("Nenhum ficheiro de entrada", file=sys.stderr)
        sys.exit(1)

    frames = []
    for path in args.inputs:
        df = pd.read_csv(path, parse_dates=["open_time","close_time"])
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} rows to {args.out}")

if __name__ == "__main__":
    main()