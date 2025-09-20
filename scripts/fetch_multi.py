# scripts/fetch_multi.py
from __future__ import annotations
import os, time, argparse, yaml, sys, math
from typing import Optional, List, Any
import pandas as pd
import requests

MAINNET_BASE = "https://api.binance.com"
TESTNET_BASE = "https://testnet.binance.vision"
COLS = ["open_time","open","high","low","close","volume",
        "close_time","qav","trades","taker_base","taker_quote","ignore"]

def to_df(rows: List[Any]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    if df.empty:
        return pd.DataFrame(columns=["open_time","high","low","close","volume","close_time"])
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time","high","low","close","volume","close_time"]]

def ms(dt_str: str) -> int:
    if "T" not in dt_str:
        dt_str += "T00:00:00"
    return int(pd.Timestamp(dt_str, tz="UTC").timestamp() * 1000)

def fetch_page(base: str, symbol: str, interval: str, start_ms: int, limit: int = 1000, timeout: int = 30):
    url = f"{base}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": start_ms}
    headers = {"User-Agent": "ai-trading-bot/1.0"}  # ajuda em alguns proxies
    r = requests.get(url, params=params, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.json()

def fetch_range(base: str, symbol: str, interval: str, start: str, end: str,
                sleep_ms=120, verbose=True, checkpoint_path: Optional[str]=None) -> pd.DataFrame:
    start_ms = ms(start); end_ms = ms(end)
    rows: List[Any] = []
    cursor = start_ms
    last_close = None
    page = 0
    backoff = sleep_ms / 1000.0

    while True:
        page += 1
        # retries básicos
        for attempt in range(6):
            try:
                batch = fetch_page(base, symbol, interval, cursor)
                break
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else None
                if code in (429, 418, 500, 502, 503, 504):
                    wait = min(60.0, backoff * (2 ** attempt))
                    if verbose:
                        print(f"[{symbol}] HTTP {code} página {page} -> retry em {wait:.1f}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise
            except requests.RequestException as e:
                wait = min(60.0, backoff * (2 ** attempt))
                if verbose:
                    print(f"[{symbol}] erro rede '{e}' página {page} -> retry em {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
        else:
            # esgotou retries
            break

        if not batch:
            if verbose:
                print(f"[{symbol}] página {page}: vazio, termina.", file=sys.stderr)
            break

        # corta por end
        batch = [row for row in batch if row[6] <= end_ms]
        if not batch:
            if verbose:
                print(f"[{symbol}] página {page}: tudo após end, termina.", file=sys.stderr)
            break

        # proteção anti-loop
        if last_close is not None and batch[-1][6] == last_close:
            if verbose:
                print(f"[{symbol}] página {page}: repetiu último close_time, termina.", file=sys.stderr)
            break

        rows.extend(batch)
        last_close = batch[-1][6]
        cursor = last_close + 1

        if verbose:
            last_dt = pd.to_datetime(last_close, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")
            total = len(rows)
            # impressão compacta de progresso
            print(f"\r[{symbol}] +{len(batch):4d}  total={total:7d}  até {last_dt} UTC  p{page}", end="", file=sys.stderr)
        # checkpoint opcional para ver crescer no disco
        if checkpoint_path and (page % 20 == 0):
            to_df(rows).to_csv(checkpoint_path, index=False)

        if last_close >= end_ms:
            if verbose:
                print(f"\n[{symbol}] atingiu end, termina.", file=sys.stderr)
            break

        time.sleep(sleep_ms / 1000.0)

    if verbose:
        print(file=sys.stderr)  # newline final
    df = to_df(rows)
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/symbols.yaml")
    ap.add_argument("--only", nargs="*", help="limita aos símbolos listados (sobrepõe config)")
    ap.add_argument("--years", nargs="*", type=int, help="limita aos anos listados (sobrepõe config)")
    ap.add_argument("--sleep_ms", type=int, default=120)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    symbols: list[str] = args.only or cfg["symbols"]
    interval: str = cfg.get("interval", "1m")
    years: list[int] = args.years or cfg["years"]
    base = MAINNET_BASE if cfg.get("source", "mainnet") == "mainnet" else TESTNET_BASE

    os.makedirs("data", exist_ok=True)

    for sym in symbols:
        os.makedirs(f"data/{sym}", exist_ok=True)
        for y in years:
            out = f"data/{sym}/klines_{interval}_{y}.csv"
            if os.path.exists(out):
                print(f">> Skip {sym} {y} (existe {out})")
                continue
            print(f">> Fetch {sym} {y} ({interval})")
            ckpt = f"data/{sym}/_tmp_{interval}_{y}.csv"
            df = fetch_range(base, sym, interval, f"{y}-01-01", f"{y}-12-31",
                             sleep_ms=args.sleep_ms, verbose=True, checkpoint_path=ckpt if args.verbose else None)
            df.to_csv(out, index=False)
            if os.path.exists(ckpt):
                try: os.remove(ckpt)
                except Exception: pass
            print(f"   saved {len(df)} rows -> {out}")

        # concat por símbolo
        files = [f"data/{sym}/klines_{interval}_{y}.csv" for y in years if os.path.exists(f"data/{sym}/klines_{interval}_{y}.csv")]
        if not files:
            print(f"! Sem ficheiros para {sym}, skip concat")
            continue
        frames = [pd.read_csv(p, parse_dates=["open_time","close_time"]) for p in files]
        all_df = pd.concat(frames, ignore_index=True).sort_values("open_time").drop_duplicates(subset=["open_time"])
        all_df.to_csv(f"data/{sym}/klines.csv", index=False)
        print(f">> {sym} concat -> data/{sym}/klines.csv rows={len(all_df)}")

if __name__ == "__main__":
    main()