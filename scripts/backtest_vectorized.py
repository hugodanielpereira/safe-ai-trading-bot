# scripts/backtest_vectorized.py
from __future__ import annotations
import argparse
import pandas as pd

FEE = 0.0004  # 4 bps por lado, ajusta conforme exchange


def backtest(signals: pd.Series, prices: pd.Series) -> pd.DataFrame:
    # positions: +1 long, -1 short, 0 flat
    pos = signals.map({"BUY": 1, "SELL": -1, "HOLD": 0}).astype(float).shift(1).fillna(0.0)
    ret = prices.pct_change().fillna(0)
    strat = pos * ret
    # fees whenever position changes
    turns = (pos != pos.shift(1)).astype(int)
    strat_after_fee = strat - turns * FEE
    equity = (1 + strat_after_fee).cumprod()
    return pd.DataFrame({"ret": strat_after_fee, "equity": equity, "pos": pos})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--prices_csv", default="data/klines.csv")
    args = ap.parse_args()

    feats = pd.read_csv(args.features)
    prices = pd.read_csv(args.prices_csv, parse_dates=["open_time","close_time"])  # has close
    close = prices["close"].iloc[-len(feats):].reset_index(drop=True)

    # naive signal from labels (for sanity): 0/1/2 -> HOLD/BUY/SELL
    y = feats["y"].astype(int)
    sig = y.replace({0: "HOLD", 1: "BUY", 2: "SELL"})
    bt = backtest(sig, close)
    print(bt.tail(10))
    print("Final equity:", float(bt["equity"].iloc[-1]))

if __name__ == "__main__":
    main()
