# tools/save_prices.py
import os, csv
from datetime import datetime

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_ohlcv_rows(symbol: str, interval: str, rows: list[dict]):
    """
    rows: [{open_time, open, high, low, close, volume, close_time, ...}, ...]
    Garante estrutura: data/SYMBOL/INTERVAL/YYYY.csv (um ficheiro por ano).
    """
    if not rows: return 0
    written = 0
    buckets = {}  # yyyy -> list[dict]
    for r in rows:
        # tenta pegar o timestamp (close_time > open_time se vier dos klines da Binance)
        ts = r.get("close_time") or r.get("open_time") or r.get("time")
        dt = datetime.utcfromtimestamp(int(ts) / 1000) if ts and int(ts) > 10**10 else datetime.utcfromtimestamp(int(ts or 0))
        y = dt.year
        buckets.setdefault(y, []).append(r)

    base = os.path.join("data", symbol.upper(), interval)
    ensure_dir(base)

    for year, chunk in buckets.items():
        fp = os.path.join(base, f"{year}.csv")
        write_header = not os.path.exists(fp)
        with open(fp, "a", newline="") as f:
            w = None
            for r in chunk:
                if w is None:
                    cols = list(r.keys())
                    w = csv.DictWriter(f, fieldnames=cols)
                    if write_header:
                        w.writeheader()
                w.writerow(r)
                written += 1
    return written