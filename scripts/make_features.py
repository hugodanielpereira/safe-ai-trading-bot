# scripts/make_features.py
from __future__ import annotations
import argparse, os
import pandas as pd
import numpy as np

# --- features leves (sem libs externas) ---
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rd = pd.Series(dn, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100/(1+rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]; low = df["low"]; close_prev = df["close"].shift(1)
    tr = pd.concat([(high-low), (high-close_prev).abs(), (low-close_prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ret1"]  = df["close"].pct_change()
    out["ret5"]  = df["close"].pct_change(5)
    out["ret15"] = df["close"].pct_change(15)
    out["vol"]   = df.get("volume", pd.Series(0, index=df.index))
    out["rsi14"] = rsi(df["close"], 14)
    out["atr14"] = atr(df, 14) / (df["close"] + 1e-12)
    out["sma9"]  = df["close"].rolling(9).mean()  / (df["close"] + 1e-12)
    out["sma21"] = df["close"].rolling(21).mean() / (df["close"] + 1e-12)
    # extras
    m20 = df["close"].rolling(20).mean()
    s20 = df["close"].rolling(20).std()
    out["bb_width"]  = (2*s20) / (m20 + 1e-12)
    out["zscore_20"] = (df["close"] - m20) / (s20 + 1e-12)
    out["ret1_abs"]  = out["ret1"].abs()
    out = out.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0.0)
    return out

def make_labels_quantiles(df: pd.DataFrame, horizon: int = 10, q: float = 0.6) -> pd.Series:
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    lab = pd.Series(0, index=df.index)  # 0=HOLD
    up = fwd.quantile(q); dn = fwd.quantile(1-q)
    lab[fwd >= up] = 1  # BUY
    lab[fwd <= dn] = 2  # SELL
    return lab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klines", default="data/klines.csv")
    ap.add_argument("--outfile", default="data/features.csv")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--use_quantiles", action="store_true")
    ap.add_argument("--q", type=float, default=0.6)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    df = pd.read_csv(args.klines, parse_dates=["open_time","close_time"])
    df = df.sort_values("open_time").reset_index(drop=True)

    feats = make_features(df)
    y = make_labels_quantiles(df, args.horizon, args.q) if args.use_quantiles else \
        (lambda d,h: (d["close"].pct_change(h).shift(-h)).pipe(
            lambda f: pd.Series(np.where(f>=0.001,1, np.where(f<=-0.001,2,0)), index=d.index))
        )(df, args.horizon)

    out = feats.copy()
    out["y"] = y
    out = out.iloc[21:-args.horizon] if args.horizon>0 else out.iloc[21:]
    out.dropna(inplace=True)
    out.to_csv(args.outfile, index=False)
    print(f"Saved features to {args.outfile} rows={len(out)} cols={len(out.columns)}")

if __name__ == "__main__":
    main()