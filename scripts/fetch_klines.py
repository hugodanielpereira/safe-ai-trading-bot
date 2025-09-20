# scripts/fetch_klines.py
from __future__ import annotations
import os, argparse, time, datetime as dt
import pandas as pd
import requests
from typing import Optional, List

MAINNET_BASE = "https://api.binance.com"
COLS = ["open_time","open","high","low","close","volume",
        "close_time","qav","trades","taker_base","taker_quote","ignore"]

def iso_to_ms(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    if "T" not in s:
        s += "T00:00:00"
    return int(dt.datetime.fromisoformat(s).timestamp() * 1000)

def fetch_page(base: str, symbol: str, interval: str,
               start_ms: Optional[int], limit: int = 1000):
    url = f"{base}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def to_df(rows: List) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time","high","low","close","volume","close_time"]]

def fetch_range(base: str, symbol: str, interval: str,
                start_ms: int, end_ms: Optional[int] = None,
                sleep_ms: int = 120, show_progress=True) -> pd.DataFrame:
    rows = []
    cursor = start_ms
    count = 0
    while True:
        batch = fetch_page(base, symbol, interval, cursor)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][6] + 1  # próximo start = último close_time + 1 ms
        count += len(batch)
        if show_progress:
            print(f"\rFetched {count} rows...", end="")
        if end_ms and cursor >= end_ms:
            break
        time.sleep(sleep_ms / 1000.0)
    print()
    if not rows:
        return pd.DataFrame(columns=["open_time","high","low","close","volume","close_time"])
    df = to_df(rows)
    df = df.sort_values("open_time").drop_duplicates(subset="open_time").reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD ou ISO")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD ou ISO")
    ap.add_argument("--outfile", default="data/klines.csv")
    ap.add_argument("--source", choices=["mainnet","testnet"], default="mainnet")
    args = ap.parse_args()

    base = MAINNET_BASE if args.source == "mainnet" else "https://testnet.binance.vision"
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    start_ms = iso_to_ms(args.start)
    end_ms   = iso_to_ms(args.end)

    df = fetch_range(base, args.symbol, args.interval, start_ms, end_ms)
    df.to_csv(args.outfile, index=False)
    print(f"Saved {len(df)} rows to {args.outfile}")

if __name__ == "__main__":
    main()