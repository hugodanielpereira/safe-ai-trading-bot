# app/main.py
from __future__ import annotations

import csv
import json
import os
import re
import glob
import hashlib
import time
import uuid
import threading
import concurrent.futures
import subprocess
import shlex
import tempfile
import joblib
from typing import Optional, Dict  # <- removei Any
from collections import defaultdict

from datetime import datetime, timezone  # <- junto datetime e timezone

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status, Query, Body  # <- removei BackgroundTasks
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .backtest import run_backtest, BTConfig
from .config import settings
from .security import verify_token
from .worker_binance import worker

# === tentar apanhar exceções específicas da lib da Binance ===
try:
    from binance.exceptions import BinanceAPIException as ClientError
except Exception:
    try:
        from binance.error import ClientError  # fallback
    except Exception:
        ClientError = Exception

app = FastAPI(title="AI Bot Starter", version=settings.bot_version)

# --- servir SPA (web/) ---
if not os.path.exists("web"):
    os.makedirs("web", exist_ok=True)
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

@app.get("/")
async def root():
    # redireciona sempre para o SPA
    return RedirectResponse(url="/web/")

# =========================
#        MODELOS
# =========================
class FetchReq(BaseModel):
    symbol: str
    interval: str
    start: str  # "YYYY-MM-DD" ou ISO
    end: str    # idem
    migrate_existing: bool = False  # opcional: migrar CSVs soltos no diretório do symbol

class MigrateReq(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    symbol: Optional[str] = None

class StartResponse(BaseModel):
    status: str
    version: str

class StopResponse(BaseModel):
    status: str

class StatusResponse(BaseModel):
    running: bool
    version: str

class SetSymbolRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    symbol: str
    model_path: Optional[str] = None

class SetSymbolResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    symbol: str
    model_path: Optional[str]
    model_loaded: bool
    info: dict
    was_running: bool
    running: bool

class BacktestReq(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    symbol: str
    interval: str = "1m"
    csv_path: str
    csv_paths: list[str] | None = None  # 👈 NOVO: permite vários ficheiros/pastas/globs
    strategy: str = "ai"
    model_path: str | None = None
    buy_th: float = 0.55
    sell_th: float = 0.55
    fee_bps: float = 1.0
    slippage_bps: float = 0.0
    start: str | None = None
    end:   str | None = None
    # filtros
    date_from: str | None = None
    date_to: str | None = None
    max_rows: int | None = None

class BacktestResp(BaseModel):
    strategy: str
    n: int
    total_return: float
    max_drawdown: float
    sharpe: float
    equity: list[dict]

# --- MODELOS DE TREINO ---
class TrainReq(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    symbol: str
    interval: str                     # "1m","5m","1h","1d", etc
    # fonte de dados (igual ao backtest loader):
    csv_path: str | None = None
    csv_paths: list[str] | None = None
    date_from: str | None = None      # janela de TREINO
    date_to: str | None = None
    max_rows: int | None = None

    # opções do algoritmo (exemplo):
    algo: str = "gbm"                 # placeholder
    params: dict | None = None

    # saída:
    out_path: str | None = None       # se não vier, geramos um nome

class TrainResp(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    ok: bool
    model_path: str
    metrics: dict | None = None

# =========================
#      UTILITÁRIOS MODELO
# =========================
# === Constantes/Helpers ===
KNOWN_INTERVALS = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"}

def _data_root():
    d = os.path.join("data")
    os.makedirs(d, exist_ok=True)
    return d

def _interval_dir(symbol: str, interval: str) -> str:
    d = os.path.join(_data_root(), symbol.upper(), interval)
    os.makedirs(d, exist_ok=True)
    return d

def _year_path(symbol: str, interval: str, year: int) -> str:
    return os.path.join(_interval_dir(symbol, interval), f"{year}.csv")

def _norm_klines_to_df(rows: list) -> pd.DataFrame:
    """
    rows estilo Binance kline:
    [ open_time, open, high, low, close, volume, close_time, qav, trades, tbb, tbq, i ]
    """
    if not rows:
        return pd.DataFrame()
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","trades","taker_buy_base","taker_buy_quote","ignore"
    ]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    # tipos
    for c in ("open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("open_time","close_time"):
        if c in df.columns:
            # Binance devolve ms epoch
            df[c] = pd.to_datetime(df[c], unit="ms", utc=True)
    if "trades" in df.columns:
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)
    # ordena por close_time se existir
    tcol = "close_time" if "close_time" in df.columns else ("open_time" if "open_time" in df.columns else None)
    if tcol:
        df = df.sort_values(tcol).reset_index(drop=True)
    return df

def _write_partitioned_by_year(df: pd.DataFrame, symbol: str, interval: str) -> dict:
    """
    Parte o DF por ano (pela close_time ou open_time) e grava/append em data/<SYM>/<INT>/<YYYY>.csv,
    com dedupe por close_time.
    """
    if df is None or df.empty:
        return {"written": 0, "files": []}
    tcol = "close_time" if "close_time" in df.columns else "open_time"
    if tcol not in df.columns:
        # nada de timestamps → não grava
        return {"written": 0, "files": []}

    out = {"written": 0, "files": []}
    df = df.copy()
    # coluna auxiliar 'year'
    years = df[tcol].dt.tz_convert("UTC").dt.year if getattr(df[tcol].dtype, "tz", None) else df[tcol].dt.year
    df["_year"] = years

    # garante conjunto de colunas e a sua ordem
    cols_order = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","trades","taker_buy_base","taker_buy_quote"
    ]
    have = [c for c in cols_order if c in df.columns]
    df = df[have + (["_year"] if "_year" in df.columns else [])]

    for y, chunk in df.groupby("_year"):
        path = _year_path(symbol, interval, int(y))
        # se existir, concat e dedupe
        if os.path.exists(path):
            try:
                old = pd.read_csv(path)
                # normaliza tempos do velho também
                for c in ("open_time","close_time"):
                    if c in old.columns:
                        old[c] = pd.to_datetime(old[c], utc=True, errors="coerce")
                # alinha colunas
                all_cols = list(dict.fromkeys(list(old.columns) + list(chunk.columns)))
                old = old.reindex(columns=all_cols)
                chunk = chunk.reindex(columns=all_cols)
                merged = pd.concat([old, chunk], ignore_index=True)

                # dedupe por close_time preferencialmente, senão open_time
                keys = "close_time" if "close_time" in merged.columns else ("open_time" if "open_time" in merged.columns else None)
                if keys:
                    merged = merged.dropna(subset=[keys]).drop_duplicates(subset=[keys], keep="last")
                # ordena temporalmente
                if keys:
                    merged = merged.sort_values(keys)
                merged.to_csv(path, index=False)
                n_new = max(0, len(merged) - len(old))
                out["written"] += n_new
                out["files"].append(path)
            except Exception:
                # fallback: escreve só o novo (caso formato antigo seja incompatível)
                chunk.drop(columns=["_year"], errors="ignore").to_csv(path, index=False)
                out["written"] += len(chunk)
                out["files"].append(path)
        else:
            chunk = chunk.drop(columns=["_year"], errors="ignore")
            chunk.to_csv(path, index=False)
            out["written"] += len(chunk)
            out["files"].append(path)
    return out

def _next_month(d: datetime) -> datetime:
    y, m = d.year, d.month
    m2 = 1 if m == 12 else m+1
    y2 = y+1 if m == 12 else y
    return datetime(y2, m2, 1, tzinfo=timezone.utc)

def _month_range_utc(start_dt: datetime, end_dt: datetime):
    # gera [ (month_start, month_end_exclusive), ... ]
    cur = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
    while cur < end_dt:
        nxt = _next_month(cur)
        yield (cur, min(nxt, end_dt))
        cur = nxt

# === Fetcher principal (Binance) ===
def _fetch_binance_klines(symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Tenta usar client.get_historical_klines; se não existir, usa paginado via get_klines.
    Devolve DF normalizado.
    """
    b = worker.bridge
    client = b.client
    rows_all = []

    # 1) tenta API “histórica” completa por janela mensal (menos erros/limites)
    has_hist = hasattr(client, "get_historical_klines")
    for m_start, m_end in _month_range_utc(start_dt, end_dt):
        start_str = m_start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = m_end.strftime("%Y-%m-%d %H:%M:%S")
        try:
            if has_hist:
                rows = client.get_historical_klines(symbol, interval, start_str, end_str)
            else:
                # fallback paginado manual
                start_ms = int(m_start.timestamp()*1000)
                end_ms = int(m_end.timestamp()*1000)
                rows = []
                last = start_ms
                while last < end_ms:
                    batch = client.get_klines(symbol=symbol, interval=interval, startTime=last, endTime=end_ms, limit=1000)
                    if not batch:
                        break
                    rows.extend(batch)
                    last_close = batch[-1][6]  # close_time
                    # avança 1ms para evitar duplicar último
                    last = int(last_close) + 1
            if rows:
                rows_all.extend(rows)
        except Exception as e:
            # não aborta a série inteira por um mês falhado
            print(f"[fetch] aviso: {symbol} {interval} {start_str}..{end_str} falhou: {e}")

    return _norm_klines_to_df(rows_all)

@app.post("/data_fetch")
async def data_fetch(req: FetchReq, _: None = Depends(verify_token)):
    """
    Vai buscar klines históricos e grava particionado em data/<SYMBOL>/<INTERVAL>/<YYYY>.csv
    """
    symbol = req.symbol.upper()
    interval = req.interval
    if interval not in KNOWN_INTERVALS:
        raise HTTPException(status_code=400, detail=f"Intervalo inválido. Usa um de {sorted(KNOWN_INTERVALS)}")

    try:
        start_dt = pd.to_datetime(req.start, utc=True).to_pydatetime()
        end_dt = pd.to_datetime(req.end, utc=True).to_pydatetime()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Datas inválidas: {e}")

    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end deve ser posterior a start")

    # fetch
    df = _fetch_binance_klines(symbol, interval, start_dt, end_dt)
    written = _write_partitioned_by_year(df, symbol, interval)

    # (opcional) migra legacy “solto” para a estrutura por ano
    migrated = None
    if req.migrate_existing:
        migrated = _migrate_flat_symbol(symbol)

    return {
        "symbol": symbol,
        "interval": interval,
        "rows_fetched": int(len(df)),
        "written": written,
        "migrated": migrated,
    }

@app.post("/data_migrate_flat")
async def data_migrate_flat(req: MigrateReq, _: None = Depends(verify_token)):
    if req.symbol:
        return _migrate_flat_symbol(req.symbol.upper())
    results = {}
    root = _data_root()
    for sym in os.listdir(root):
        if not os.path.isdir(os.path.join(root, sym)):
            continue
        results[sym] = _migrate_flat_symbol(sym)
    return results

def _detect_interval_from_filename(name: str) -> Optional[str]:
    # tenta apanhar “_1m”, “/1m/”, “-1h-”, etc
    m = re.search(r'(?i)(?:^|[^\w])((1|3|5|15|30)m|(1|2|4|6|8|12)h|1d|3d|1w|1M)(?:[^\w]|$)', name)
    return m.group(1) if m else None

def _migrate_flat_symbol(symbol: str) -> dict:
    """
    Procura CSVs “soltos” em data/<SYMBOL> (e subpastas ambíguas) e regrava para interval/year.
    Usa heurística para descobrir o intervalo (nome do ficheiro).
    """
    base = os.path.join(_data_root(), symbol)
    if not os.path.exists(base):
        return {"migrated": 0, "files": []}

    moved = 0
    files_done = []
    # varre tudo por baixo de data/<symbol>, mas ignora pastas que já sejam intervalos KNOWN_INTERVALS
    for root, dirs, files in os.walk(base):
        # se já estamos dentro de uma pasta “interval”, ignora (já está certo)
        if os.path.basename(root) in KNOWN_INTERVALS:
            continue
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            full = os.path.join(root, fn)
            interval = _detect_interval_from_filename(fn) or _detect_interval_from_filename(root) or "1m"
            try:
                df = pd.read_csv(full)
                # normaliza tempos
                for c in ("close_time","open_time"):
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
                # se não tiver tempos, tenta criar a partir de “time”/“timestamp”
                if "close_time" not in df.columns:
                    for c in ("time","timestamp"):
                        if c in df.columns:
                            dt = pd.to_datetime(df[c], errors="coerce", utc=True)
                            df["close_time"] = dt
                            break
                # grava particionado
                w = _write_partitioned_by_year(df, symbol, interval)
                moved += w.get("written", 0)
                files_done.append(full)
            except Exception as e:
                print(f"[migrate] erro a processar {full}: {e}")

    return {"migrated": moved, "files": files_done}


def _load_prices(req) -> pd.DataFrame:
    """
    Carrega preços a partir de:
      - req.csv_paths: lista de ficheiros/diretórios/globs
      - req.csv_path:  ficheiro/diretório/glob

    Se req.interval estiver vazio, tenta inferir um dos KNOWN_INTERVALS a partir
    do caminho (pasta ou nome dos ficheiros).

    Regras:
      - Diretório com subpasta do intervalo (ex.: 1m/): usa só essa.
      - Diretório “flat”: filtra *.csv cujo nome contenha o intervalo (ex.: *_1m_*.csv).
      - Glob: expande normalmente (aplica filtro por nome se fizer sentido).
      - Aplica filtros de data (UTC) e max_rows.
    """
    def detect_interval_from_path(path_or_list: str) -> str | None:
        low = path_or_list.lower()
        # ordenar por comprimento desc para apanhar '12h' antes de '1h', etc
        for iv in sorted(KNOWN_INTERVALS, key=len, reverse=True):
            if re.search(rf"(?:^|[._\-/]){re.escape(iv.lower())}(?:[._\-/]|$)", low):
                return iv
        return None

    # 1) construir lista bruta de caminhos
    raw_inputs: list[str] = []
    if getattr(req, "csv_paths", None):
        raw_inputs.extend([p for p in (req.csv_paths or []) if p])
    elif getattr(req, "csv_path", None):
        raw_inputs.append(req.csv_path)
    else:
        raise HTTPException(status_code=400, detail="Indica csv_path ou csv_paths")
    if not raw_inputs:
        raise HTTPException(status_code=400, detail="Nenhum caminho de CSV fornecido")

    # 2) apurar/infere intervalo
    interval = (getattr(req, "interval", "") or "").strip()
    if not interval:
        interval_detected = detect_interval_from_path(" ".join(raw_inputs))
        interval = interval_detected or ""
    IV_RE_SAFE = re.compile(rf"(?:^|[._\-]){re.escape(interval)}(?:[._\-]|$)", re.IGNORECASE) if interval else None

    # 3) expandir para lista de ficheiros
    files: list[str] = []

    def add_dir(dir_path: str):
        """Adiciona ficheiros de um diretório (prefere subpasta do intervalo se existir)."""
        # (a) subpasta do intervalo
        if interval:
            subdir = os.path.join(dir_path, interval)
            if os.path.isdir(subdir):
                files.extend(sorted(glob.glob(os.path.join(subdir, "*.csv"))))
                return
            # tentativa case-insensitive
            try:
                for name in os.listdir(dir_path):
                    full = os.path.join(dir_path, name)
                    if os.path.isdir(full) and name.lower() == interval.lower():
                        files.extend(sorted(glob.glob(os.path.join(full, "*.csv"))))
                        return
            except Exception:
                pass

        # (b) diretório “flat”: apanhar todos *.csv e filtrar por nome se soubermos o intervalo
        all_csvs = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
        if interval and IV_RE_SAFE:
            filtered = [f for f in all_csvs if IV_RE_SAFE.search(os.path.basename(f))]
            files.extend(filtered if filtered else all_csvs)
        else:
            files.extend(all_csvs)

    for inp in raw_inputs:
        p = str(inp).strip()
        if not p:
            continue
        # glob?
        if any(ch in p for ch in ["*", "?", "["]):
            matched = sorted(glob.glob(p))
            if not matched:
                continue
            # se não houver intervalo explícito no próprio glob, filtra por nome
            if interval and IV_RE_SAFE and interval.lower() not in p.lower():
                matched = [f for f in matched if f.lower().endswith(".csv") and IV_RE_SAFE.search(os.path.basename(f))]
            files.extend([f for f in matched if f.lower().endswith(".csv")])
        elif os.path.isdir(p):
            add_dir(p)
        elif os.path.isfile(p):
            files.append(p)
        else:
            # tentativa de glob como fallback
            matched = sorted(glob.glob(p))
            if matched:
                files.extend([f for f in matched if f.lower().endswith(".csv")])

    # deduplicar/ordenar
    files = sorted(dict.fromkeys(files).keys())
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum CSV encontrado depois de expandir os caminhos")

    # 4) ler e concatenar
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df["__src"] = fp
            if interval:
                df["__interval_hint"] = interval
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Falha ao ler {fp}: {e}") from e
        frames.append(df)

    if not frames:
        raise HTTPException(status_code=400, detail="Falha: não foi possível ler nenhum CSV")

    prices = pd.concat(frames, ignore_index=True)

    # 5) ordenar por coluna temporal
    time_cols = ("close_time", "open_time", "time", "timestamp")
    tcol = next((c for c in time_cols if c in prices.columns), None)
    if tcol:
        prices = prices.sort_values(tcol, kind="mergesort")

    # 6) filtros de data (UTC)
    if getattr(req, "date_from", None) or getattr(req, "date_to", None):
        if not tcol:
            raise HTTPException(status_code=400, detail="Não há coluna temporal para filtrar")
        t = pd.to_datetime(prices[tcol], errors="coerce", utc=True)

        dt_from = None
        dt_to_excl = None
        if getattr(req, "date_from", None):
            dt_from = pd.to_datetime(req.date_from, errors="coerce")
            if dt_from.tzinfo is None:
                dt_from = dt_from.tz_localize("UTC")
        if getattr(req, "date_to", None):
            dt_to = pd.to_datetime(req.date_to, errors="coerce")
            if dt_to.tzinfo is None:
                dt_to = dt_to.tz_localize("UTC")
            dt_to_excl = dt_to + pd.Timedelta(days=1)

        mask = pd.Series(True, index=prices.index)
        if dt_from is not None:
            mask &= (t >= dt_from)
        if dt_to_excl is not None:
            mask &= (t < dt_to_excl)
        prices = prices.loc[mask]

    # 7) max_rows
    if getattr(req, "max_rows", None):
        try:
            prices = prices.tail(int(req.max_rows))
        except Exception:
            pass

    return prices

# ----- Job store em memória -----
BACKTEST_JOBS: Dict[str, dict] = {}
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)  # ajusta se quiseres
JOBS_LOCK = threading.Lock()

def _job_new(payload: dict) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "queued",      # queued | running | done | error | cancelled
        "phase": "created",      # created | loading_csv | running | saving | finished
        "created_at": time.time(),
        "started_at": None,
        "ended_at": None,
        "elapsed": 0.0,
        "percent": None,         # 0-100 ou None
        "message": "",
        "request": payload,      # BacktestReq como dict
        "result_path": None,     # caminho do .json guardado
        "error": None,
        "rows_total": None,
        "rows_done": 0,
        "cancel_flag": False,
    }
    with JOBS_LOCK:
        BACKTEST_JOBS[job_id] = job
    return job

def _job_update(job_id: str, **kw):
    with JOBS_LOCK:
        j = BACKTEST_JOBS.get(job_id)
        if not j: return
        j.update(kw)
        if j.get("started_at"):
            j["elapsed"] = time.time() - j["started_at"]

# =========================
#      CONTROLOS BOT
# =========================
@app.post("/start", response_model=StartResponse)
async def start(_: None = Depends(verify_token)):
    await worker.start()
    return StartResponse(status="started", version=settings.bot_version)

@app.post("/stop", response_model=StopResponse)
async def stop(_: None = Depends(verify_token)):
    await worker.stop()
    return StopResponse(status="stopped")

@app.get("/status", response_model=StatusResponse)
async def status():
    return StatusResponse(running=worker.running, version=settings.bot_version)

@app.get("/logs")
async def logs():
    return worker.logs()

# =========================
#         BINANCE
# =========================
@app.get("/balance")
async def balance(_: None = Depends(verify_token)):
    b = worker.bridge
    if b.use_futures:
        return b.client.futures_account_balance()
    return b.client.get_asset_balance(asset="USDT")

@app.get("/trades")
async def trades(
    _: None = Depends(verify_token),
    symbol: Optional[str] = None,
    limit: int = Query(50, ge=1, le=1000),
):
    b = worker.bridge
    sym = symbol or b.symbol
    try:
        return b.client.get_my_trades(symbol=sym, limit=limit)
    except ClientError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/orders")
async def orders(_: None = Depends(verify_token), limit: int = Query(10, ge=1, le=1000)):
    b = worker.bridge
    try:
        data = b.client.get_all_orders(symbol=b.symbol, limit=limit)
        return data[-limit:]
    except ClientError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ticker")
async def ticker(symbol: Optional[str] = None):
    b = worker.bridge
    sym = symbol or b.symbol
    try:
        return b.client.get_symbol_ticker(symbol=sym)
    except ClientError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/positions")
async def positions(_: None = Depends(verify_token)):
    b = worker.bridge
    if not b.use_futures:
        return {"detail": "Spot mode: no positions endpoint. Switch to futures to use this."}
    try:
        acct = b.client.futures_account()
        return acct.get("positions", [])
    except ClientError as e:
        raise HTTPException(status_code=400, detail=str(e))

def _assert_testnet_and_live_or_403(b):
    api_url = getattr(b.client, "API_URL", "")
    fu_url = getattr(b.client, "FUTURES_URL", "")
    on_testnet = ("binance.vision" in api_url) or ("binancefuture" in fu_url)
    if not on_testnet:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Disabled outside testnet")
    if not b.live:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="TRADE_LIVE=false")

@app.post("/force_buy")
async def force_buy(_: None = Depends(verify_token)):
    b = worker.bridge
    _assert_testnet_and_live_or_403(b)
    order = b.market_order("BUY")
    return {"ok": True, "orderId": order.get("orderId"), "side": "BUY", "symbol": b.symbol}

@app.post("/force_sell")
async def force_sell(_: None = Depends(verify_token)):
    b = worker.bridge
    _assert_testnet_and_live_or_403(b)
    order = b.market_order("SELL")
    return {"ok": True, "orderId": order.get("orderId"), "side": "SELL", "symbol": b.symbol}

@app.get("/config")
async def config():
    b = worker.bridge
    return {
        "testnet": ("binance.vision" in getattr(b.client, "API_URL", "")) or ("binancefuture" in getattr(b.client, "FUTURES_URL", "")),
        "futures": b.use_futures,
        "live": b.live,
        "symbol": b.symbol,
        "qty": b.qty,
        "cooldown_seconds": settings.cooldown_seconds,
        "min_proba_gap": settings.min_proba_gap,
        "version": settings.bot_version,
    }

# =========================
#       MODELOS / MÉTRICAS
# =========================
def _read_metrics_summary(path: str = "models/metrics_summary.csv"):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["rows"] = int(r.get("rows", 0) or 0)
            try:
                r["cv_accuracy"] = float(r.get("cv_accuracy", 0) or 0)
            except Exception:
                r["cv_accuracy"] = 0.0
            rows.append(r)
    rows.sort(key=lambda x: x.get("cv_accuracy", 0), reverse=True)
    return rows

def _read_symbol_detail(symbol: str):
    meta_path = f"models/metrics_{symbol}.json"
    fi_path = f"models/feature_importances_{symbol}.csv"
    detail = {"symbol": symbol}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                detail.update(json.load(f))
        except Exception as e:
            detail["error_meta"] = str(e)
    if os.path.exists(fi_path):
        try:
            top = []
            with open(fi_path, "r", newline="") as f:
                reader = csv.reader(f)
                # tenta ler "feature,importance"
                for row in reader:
                    if len(row) >= 2:
                        try:
                            top.append({"feature": row[0], "importance": float(row[1])})
                        except Exception:
                            pass
                    if len(top) >= 25:
                        break
            detail["top_features"] = top
        except Exception as e:
            detail["error_features"] = str(e)
    return detail

@app.get("/metrics")
async def metrics(symbol: Optional[str] = None, _: None = Depends(verify_token)):
    if symbol:
        return _read_symbol_detail(symbol)
    return {"summary": _read_metrics_summary()}

# === trocar símbolo + modelo em runtime ===
@app.post("/set_symbol", response_model=SetSymbolResponse)
async def set_symbol(req: SetSymbolRequest, _: None = Depends(verify_token)):
    was_running = worker.running
    if was_running:
        await worker.stop()

    worker.bridge.symbol = req.symbol

    # carregar modelo
    from .signals import set_model_from_joblib, current_model_info
    mp = req.model_path or f"models/gbm_{req.symbol}.pkl"
    model_loaded = False
    info = {}
    try:
        info = set_model_from_joblib(mp)
        model_loaded = bool(info.get("model_loaded"))
        info["current"] = current_model_info()
    except Exception as e:
        info = {"error": str(e)}

    if was_running:
        await worker.start()

    return SetSymbolResponse(
        symbol=req.symbol,
        model_path=mp,
        model_loaded=model_loaded,
        info=info,
        was_running=was_running,
        running=worker.running,
    )

# =========================
#    PORTFÓLIO + GRÁFICO
# =========================
def _symbol_assets(client, symbol: str) -> tuple[str, str]:
    try:
        info = client.get_symbol_info(symbol)
        if info and "baseAsset" in info and "quoteAsset" in info:
            return info["baseAsset"], info["quoteAsset"]
    except Exception:
        pass
    for q in ("USDT","BUSD","USDC","BTC","ETH","BNB","EUR","TRY"):
        if symbol.endswith(q):
            return symbol[:-len(q)], q
    return symbol[:3], symbol[3:]

def _compute_spot_position_from_trades(trades: list[dict]) -> tuple[float, float]:
    qty = 0.0
    cost = 0.0
    for t in trades:
        side_is_buy = bool(t.get("isBuyer"))
        price = float(t.get("price", 0))
        q = float(t.get("qty", 0))
        gross = price * q
        if side_is_buy:
            qty += q
            cost += gross
        else:
            qty -= q
            cost -= gross
    return qty, cost

@app.get("/portfolio")
async def portfolio(_: None = Depends(verify_token), symbol: str | None = None, limit: int = 1000):
    b = worker.bridge
    sym = symbol or b.symbol
    base, quote = _symbol_assets(b.client, sym)

    tr = b.client.get_my_trades(symbol=sym, limit=min(limit, 1000))
    tr = sorted(tr, key=lambda x: x.get("time", 0))

    pos_qty, pos_cost = _compute_spot_position_from_trades(tr)
    avg_price = (pos_cost / pos_qty) if pos_qty > 0 else 0.0

    tick = b.client.get_symbol_ticker(symbol=sym)
    px = float(tick.get("price", 0.0))

    unreal = (px - avg_price) * pos_qty if pos_qty > 0 else 0.0
    unreal_pct = ((px / avg_price) - 1.0) if avg_price > 0 else 0.0

    try:
        bal_base = b.client.get_asset_balance(asset=base)
        bal_quote = b.client.get_asset_balance(asset=quote)
    except Exception:
        bal_base = {"free":"0","locked":"0"}
        bal_quote = {"free":"0","locked":"0"}

    return {
        "symbol": sym,
        "base": base,
        "quote": quote,
        "price": px,
        "position_qty": pos_qty,
        "avg_price": avg_price,
        "unrealized_pnl": unreal,
        "unrealized_pnl_pct": unreal_pct,
        "balances": { base: bal_base, quote: bal_quote },
        "trades_recent": len(tr),
        "trades": tr[-200:],
    }

@app.get("/klines")
async def klines(symbol: str | None = None, interval: str = "1m", limit: int = 200):
    b = worker.bridge
    sym = symbol or b.symbol
    # respeitar o símbolo pedido
    df = b.klines_df(symbol=sym, interval=interval, limit=min(max(limit, 10), 1500))
    out = []
    use_time = "close_time" if "close_time" in df.columns else None
    tail = df.tail(limit).reset_index(drop=True)
    for _, row in tail.iterrows():
        ts = int(pd.Timestamp(row["close_time"]).value // 10**6) if use_time else None
        out.append({"t": ts, "close": float(row["close"])})
    return {"symbol": sym, "interval": interval, "data": out}

@app.post("/backtest", response_model=BacktestResp)
async def backtest(req: BacktestReq, _: None = Depends(verify_token)):
    # usa o mesmo loader do async: suporta diretório/glob e filtros de data (UTC)
    try:
        prices = _load_prices(req)
    except HTTPException:
        # re-raise tal como veio do loader
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha a preparar preços: {e}") from e

    try:
        cfg = BTConfig(
            strategy=req.strategy.lower(),
            model_path=req.model_path,
            buy_th=req.buy_th,
            sell_th=req.sell_th,
            fee_bps=req.fee_bps,
            slippage_bps=req.slippage_bps,
        )
        res = run_backtest(prices, cfg)  # retorna dict compatível com BacktestResp
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest falhou: {e}") from e

    return BacktestResp(**res)

# ===== Backtest: guardar / listar / carregar =====
def _bt_dir():
    d = os.path.join("web", "backtests")
    os.makedirs(d, exist_ok=True)
    return d

def _slug_for(req: dict) -> str:
    raw = json.dumps(req, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]

SYMB_RE = re.compile(r'([A-Z]{3,12}(?:USDT|BUSD|USDC|BTC|ETH|BNB))', re.IGNORECASE)

def _infer_symbol(csv_path: str | None, model_path: str | None, default: str, csv_paths: list[str] | None = None) -> str:
    candidates = []
    if csv_paths:
        candidates.extend(csv_paths)
    candidates.extend([csv_path or '', model_path or ''])

    for p in candidates:
        mdir = re.search(r'data/([A-Za-z0-9]{3,12})/', p or '')
        if mdir:
            return mdir.group(1).upper()
        m = SYMB_RE.search(p or '')
        if m:
            return m.group(1).upper()
    return (default or '').upper()

@app.post("/backtest_run")
async def backtest_run(req: BacktestReq, save: bool = True, _: None = Depends(verify_token)):
    # 1) carregar/preparar preços (suporta glob, diretório, vários CSVs e filtros)
    prices = _load_prices(req)

    # 2) correr estratégia
    cfg = BTConfig(
        strategy=req.strategy.lower(),
        model_path=req.model_path,
        buy_th=req.buy_th,
        sell_th=req.sell_th,
        fee_bps=req.fee_bps,
        slippage_bps=req.slippage_bps,
    )
    res = run_backtest(prices, cfg)

    # 3) compor payload (com símbolo deduzido com fallback no req.symbol)
    sym = _infer_symbol(req.csv_path, req.model_path, req.symbol, req.csv_paths)
    eq = res.get("equity", []) or []
    payload = {
        "summary": {
            "symbol": sym,
            "interval": req.interval,
            "strategy": res.get("strategy"),
            "n": res.get("n", 0),
            "total_return": res.get("total_return", 0.0),
            "max_drawdown": res.get("max_drawdown", 0.0),
            "sharpe": res.get("sharpe", 0.0),
        },
        "equity_preview": {"head": eq[:3], "tail": eq[-3:]},
        "equity_sample": [eq[i] for i in range(0, len(eq), max(1, len(eq)//1200))] + ([eq[-1]] if eq else []),
        "request": req.model_dump(),
    }

    if save:
        rec = dict(payload)
        slug = _slug_for(rec["request"])
        rec["id"] = slug
        outp = os.path.join(_bt_dir(), f"{slug}.json")
        with open(outp, "w") as f:
            json.dump(rec, f)
        payload["id"] = slug
        payload["saved_path"] = outp

    return payload

@app.post("/backtest_run_async")
async def backtest_run_async(req: BacktestReq, _: None = Depends(verify_token)):
    """
    Cria um job assíncrono de backtest:
    - Usa _load_prices(req) (respeita date_from/date_to/max_rows, diretórios e globs)
    - Corre run_backtest(...)
    - Guarda o resultado em web/backtests/<id>.json
    - Expõe progresso via /backtest_status?job_id=...
    """
    job = _job_new(req.model_dump())
    job_id = job["id"]

    def _worker():
        try:
            _job_update(job_id, status="running", phase="loading_csv", started_at=time.time(), percent=1, message="A ler CSV…")

            # usa o loader único (com glob, diretórios e filtros de data)
            prices = _load_prices(req)

            # estimativa simples para percent
            rows_total = int(prices.shape[0]) if hasattr(prices, "shape") else None
            _job_update(job_id, rows_total=rows_total, message="CSV carregado", percent=10, phase="running")

            from .backtest import run_backtest, BTConfig
            cfg = BTConfig(
                strategy=req.strategy.lower(),
                model_path=req.model_path,
                buy_th=req.buy_th,
                sell_th=req.sell_th,
                fee_bps=req.fee_bps,
                slippage_bps=req.slippage_bps,
            )

            # corre — enquanto corre, mantemos percent ~10..95
            _job_update(job_id, message="A executar estratégia…", percent=15)
            res = run_backtest(prices, cfg)

            _job_update(job_id, phase="saving", message="A guardar resultado…", percent=98)

            sym = _infer_symbol(req.csv_path, req.model_path, req.symbol, req.csv_paths)
            eq = res.get("equity", []) or []
            payload = {
                "summary": {
                    "symbol": sym,
                    "interval": req.interval,
                    "strategy": res.get("strategy"),
                    "n": res.get("n", 0),
                    "total_return": res.get("total_return", 0.0),
                    "max_drawdown": res.get("max_drawdown", 0.0),
                    "sharpe": res.get("sharpe", 0.0),
                },
                "equity_preview": {"head": eq[:3], "tail": eq[-3:]},
                "equity_sample": [eq[i] for i in range(0, len(eq), max(1, len(eq)//1200))] + ([eq[-1]] if eq else []),
                "request": req.model_dump(),
            }

            slug = _slug_for(payload["request"])
            payload["id"] = slug
            outp = os.path.join(_bt_dir(), f"{slug}.json")
            with open(outp, "w") as f:
                json.dump(payload, f)

            _job_update(job_id, phase="finished", status="done", percent=100,
                        ended_at=time.time(), result_path=outp, message="Concluído")
        except Exception as e:
            _job_update(job_id, status="error", phase="finished",
                        ended_at=time.time(), error=str(e), message="Erro")

    # lança no pool de threads
    EXECUTOR.submit(_worker)
    return {"job_id": job_id}

@app.get("/backtest_list")
async def backtest_list(_: None = Depends(verify_token)):
    items = []
    d = _bt_dir()
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"): continue
        try:
            with open(os.path.join(d, fn), "r") as f:
                j = json.load(f)
            items.append({
                "id": j.get("id") or fn[:-5],
                "summary": j.get("summary", {}),
                "saved_path": os.path.join(d, fn),
            })
        except Exception:
            pass
    # ordena por ordem alfabética de id (ou poderias por data de ficheiro)
    return {"items": items}

@app.get("/backtest_get")
async def backtest_get(id: Optional[str] = None, path: Optional[str] = None, _: None = Depends(verify_token)):
    if not id and not path:
        raise HTTPException(status_code=400, detail="Fornece id ou path")
    fp = path
    if id and not path:
        fp = os.path.join(_bt_dir(), f"{id}.json")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail=f"Não encontrado: {fp}")
    with open(fp, "r") as f:
        return json.load(f)
    
@app.get("/backtest_get/{id}")
async def backtest_get_path(id: str, _: None = Depends(verify_token)):
    fp = os.path.join(_bt_dir(), f"{id}.json")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail=f"Não encontrado: {fp}")
    with open(fp, "r") as f:
        return json.load(f)
    
@app.get("/backtest_status")
async def backtest_status(job_id: str, _: None = Depends(verify_token)):
    with JOBS_LOCK:
        j = BACKTEST_JOBS.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="Job não encontrado")

        # 1) atualizar 'elapsed' em tempo real enquanto corre
        if j.get("started_at") and j.get("status") == "running":
            j["elapsed"] = time.time() - j["started_at"]

            # 2) progresso “suave” enquanto a estratégia corre (sem telemetria real)
            #    Mantém entre 10% e <97% e cresce com o tempo.
            p = j.get("percent", 10) or 10
            if 10 <= p < 97 and j.get("phase") == "running":
                # Heurística: duração-alvo em segundos em função do nº de linhas (cap entre 30s e 10min)
                rows_total = float(j.get("rows_total") or 0.0)
                cap_s = min(600.0, max(30.0, rows_total / 5000.0))  # ajusta se quiseres
                prog = 10.0 + min(87.0, (j["elapsed"] / cap_s) * 87.0)
                j["percent"] = max(p, prog)

        return j

@app.post("/backtest_cancel")
async def backtest_cancel(job_id: str = Body(..., embed=True), _: None = Depends(verify_token)):
    with JOBS_LOCK:
        j = BACKTEST_JOBS.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="Job não encontrado")
        # Nota: run_backtest não suporta cancelamento hard; marcamos flag p/ futuro
        j["cancel_flag"] = True
        j["status"] = "cancelled"
        j["phase"] = "finished"
        j["ended_at"] = time.time()
    return {"ok": True}

@app.get("/backtest_jobs")
async def backtest_jobs(_: None = Depends(verify_token)):
    with JOBS_LOCK:
        # devolve snapshot leve (sem request completo)
        return {"items": [
            {
                "id": j["id"],
                "status": j["status"],
                "phase": j["phase"],
                "elapsed": j["elapsed"],
                "percent": j["percent"],
                "message": j["message"],
                "result_path": j["result_path"],
                "error": j["error"],
            }
            for j in BACKTEST_JOBS.values()
        ]}

# =========================
#       TREINO MODELO
# ----- job store p/ treino -----
TRAIN_JOBS: Dict[str, dict] = {}

def _train_job_new(payload: dict) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "queued",      # queued | running | done | error | cancelled
        "phase": "created",      # created | loading | training | saving | finished
        "created_at": time.time(),
        "started_at": None,
        "ended_at": None,
        "elapsed": 0.0,
        "percent": None,
        "message": "",
        "request": payload,
        "result": None,          # dict com {model_path, metrics}
        "error": None,
        "cancel_flag": False,
    }
    with JOBS_LOCK:
        TRAIN_JOBS[job_id] = job
    return job

def _train_job_update(job_id: str, **kw):
    with JOBS_LOCK:
        j = TRAIN_JOBS.get(job_id)
        if not j: return
        j.update(kw)
        if j.get("started_at"):
            j["elapsed"] = time.time() - j["started_at"]

def _default_model_name(req: TrainReq) -> str:
    f = f"models/gbm_{req.symbol.upper()}_{req.interval}_{(req.date_from or 'x')}--{(req.date_to or 'y')}.pkl"
    os.makedirs(os.path.dirname(f), exist_ok=True)
    return f

INACTIVITY_HEARTBEAT_S = 2         # ping UI no máx. 1x/2s
CLI_TIMEOUT_SECS = 60 * 15         # 15 min para fallback CLI

def _safe_dump_joblib(model_obj, model_path: str) -> str:
    """
    Grava o modelo de forma atómica:
      - cria diretório
      - escreve para um ficheiro temporário com compressão moderada
      - faz os.replace() para o destino (atómico)
    """
    out_dir = os.path.dirname(model_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix="mdl_", suffix=".pkl", dir=out_dir)
    os.close(fd)
    try:
        # compress=3 ~ rápido; protocol=4 p/ compat.
        joblib.dump(model_obj, tmp_path, compress=3, protocol=4)
        os.replace(tmp_path, model_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass
    return model_path

@app.post("/train_run_async")
async def train_run_async(req: TrainReq, _: None = Depends(verify_token)):
    """
    Cria um job de treino:
      - carrega dados via _load_prices(req)
      - tenta treinar via Python (.train.train_model)
      - fallback: CLI (make retrain-multi ...)
      - guarda modelo (dump atómico) e devolve métricas
      - progresso/mensagens por fase
    """
    job = _train_job_new(req.model_dump())
    job_id = job["id"]

    def _worker():
        try:
            # ============ LOADING ============
            _train_job_update(job_id, status="running", phase="loading",
                              started_at=time.time(), percent=5, message="A carregar dados…")
            prices = _load_prices(req)

            _train_job_update(job_id, phase="training", percent=20, message="A treinar via Python…")

            model_path = req.out_path or _default_model_name(req)
            metrics = {}
            used_python_trainer = False
            model_obj = None

            # heartbeat simples (pinga no máximo 1x/2s)
            last_ping = 0.0
            def heartbeat(msg=None, bump=None):
                nonlocal last_ping
                now = time.time()
                if now - last_ping >= INACTIVITY_HEARTBEAT_S:
                    kw = {"message": msg or "a treinar…"}
                    if bump is not None:
                        # sobe percent mas não passa 95 enquanto treina
                        p = TRAIN_JOBS.get(job_id, {}).get("percent") or 20
                        kw["percent"] = min(95, max(p, bump))
                    _train_job_update(job_id, **kw)
                    last_ping = now

            # ======= TENTA PYTHON TRAINER =======
            try:
                from .train import train_model, TrainConfig
                cfg = TrainConfig(
                    symbol=req.symbol,
                    interval=req.interval,
                    algo=req.algo,
                    params=req.params or {},
                )
                # Se o teu train_model aceitar callback, passa o heartbeat:
                #   model_obj, metrics = train_model(prices, cfg, out_path=None, callback=heartbeat)
                out = train_model(prices, cfg, out_path=None)  # sem out_path -> nós guardamos

                # suportar formatos diferentes que o teu trainer possa devolver:
                # 1) (model_obj, metrics)      2) (model_path, metrics)      3) dict
                if isinstance(out, tuple) and len(out) == 2:
                    a, b = out
                    # heurística: string -> path; caso contrário -> objeto
                    if isinstance(a, str):
                        model_path = a
                        metrics = b or {}
                        model_obj = None
                    else:
                        model_obj = a
                        metrics = b or {}
                elif isinstance(out, dict):
                    model_obj = out.get("model")
                    metrics = out.get("metrics") or {}
                    model_path = out.get("model_path") or model_path
                elif isinstance(out, str):
                    model_path = out
                    metrics = {}
                    model_obj = None
                else:
                    # desconhecido → tenta assumir que é o objeto
                    model_obj = out
                    metrics = {}

                used_python_trainer = True
                heartbeat("Treino Python concluído. A guardar modelo…", bump=96)

            except Exception as e_py:
                # ======= FALLBACK CLI =======
                _train_job_update(job_id, message=f"A treinar via CLI… (fallback) Motivo: {e_py.__class__.__name__}",
                                  percent=22)
                csv_arg = ''
                if req.csv_paths:
                    csv_arg = ','.join(req.csv_paths)
                elif req.csv_path:
                    csv_arg = req.csv_path

                # Ajusta o comando ao teu Makefile real
                cmd = f'make retrain-multi SYMBOL={req.symbol} INTERVAL={req.interval} CSV="{csv_arg}" FROM="{req.date_from or ""}" TO="{req.date_to or ""}" OUT="{model_path}"'
                proc = subprocess.Popen(shlex.split(cmd),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        text=True,
                                        bufsize=1)
                p = 22
                try:
                    for line in iter(proc.stdout.readline, ''):
                        line = line.strip()
                        p = min(95, p + 1)
                        _train_job_update(job_id, percent=p, message=(line[:180] or "CLI…"))
                        # cancelamento “soft”
                        with JOBS_LOCK:
                            if TRAIN_JOBS.get(job_id, {}).get("cancel_flag"):
                                proc.kill()
                                _train_job_update(job_id, status="cancelled", phase="finished",
                                                  ended_at=time.time(), message="Cancelado durante CLI")
                                return
                    code = proc.wait(timeout=CLI_TIMEOUT_SECS)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    _train_job_update(job_id, status="error", phase="finished", ended_at=time.time(),
                                      message=f"CLI timeout após {CLI_TIMEOUT_SECS}s")
                    return

                if code != 0:
                    raise RuntimeError(f"CLI terminou com código {code}")

                # no CLI assumimos que já gravou em model_path
                model_obj = None
                metrics = metrics or {}

            # ============ SAVING ============
            _train_job_update(job_id, phase="saving", percent=96, message="A guardar modelo…")

            # se ainda não houver ficheiro pronto e tivermos objeto, guardamos nós
            if model_obj is not None:
                try:
                    _safe_dump_joblib(model_obj, model_path)
                except Exception as e_dump:
                    _train_job_update(job_id, status="error", phase="finished", ended_at=time.time(),
                                      message=f"Erro a guardar modelo: {e_dump}")
                    return
            else:
                # se veio apenas caminho, certifica-te que existe
                d = os.path.dirname(model_path) or "."
                os.makedirs(d, exist_ok=True)
                # nada a fazer se o trainer/CLI já guardou

            # cancelamento tardio (antes de concluir)
            with JOBS_LOCK:
                if TRAIN_JOBS.get(job_id, {}).get("cancel_flag"):
                    _train_job_update(job_id, status="cancelled", phase="finished",
                                      ended_at=time.time(), percent=100, message="Cancelado no fim do treino")
                    return

            result = {"model_path": model_path, "metrics": metrics, "via": "python" if used_python_trainer else "cli"}
            _train_job_update(job_id, status="done", phase="finished", ended_at=time.time(),
                              percent=100, result=result, message=f"Treino concluído ({result['via']})")
        except Exception as e:
            _train_job_update(job_id, status="error", phase="finished", ended_at=time.time(),
                              message=f"Erro no treino: {e}")

    EXECUTOR.submit(_worker)
    return {"job_id": job_id}

@app.get("/train_status")
async def train_status(job_id: str, _: None = Depends(verify_token)):
    with JOBS_LOCK:
        j = TRAIN_JOBS.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="Job não encontrado")
        # mantém elapsed actualizado
        if j.get("started_at") and j.get("status") == "running":
            j["elapsed"] = time.time() - j["started_at"]
        return j

@app.post("/train_cancel")
async def train_cancel(job_id: str = Body(..., embed=True), _: None = Depends(verify_token)):
    with JOBS_LOCK:
        j = TRAIN_JOBS.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="Job não encontrado")
        j["cancel_flag"] = True
        j["status"] = "cancelled"
        j["phase"] = "finished"
        j["message"] = "Cancelado pelo utilizador"
        j["ended_at"] = time.time()
    return {"ok": True}

# =========================
#       DATA INVENTORY
# =========================
@app.get("/data_inventory")
async def data_inventory(_: None = Depends(verify_token)):
    """
    Varre data/<SYMBOL>/<INTERVAL>/*.csv e devolve:
    { items: [{symbol, interval, years:[YYYY,...], count:int, path:str}, ...] }
    """
    root = _data_root()
    items_map = defaultdict(lambda: {"years": set(), "count": 0, "path": ""})

    if not os.path.isdir(root):
        return {"items": []}

    for sym in os.listdir(root):
        sym_dir = os.path.join(root, sym)
        if not os.path.isdir(sym_dir):
            continue
        for iv in os.listdir(sym_dir):
            iv_dir = os.path.join(sym_dir, iv)
            if not os.path.isdir(iv_dir):
                continue
            key = (sym.upper(), iv)
            rec = items_map[key]
            rec["path"] = f"data/{sym}/{iv}/"

            # apanha CSVs no diretório do intervalo
            for fn in os.listdir(iv_dir):
                if not fn.lower().endswith(".csv"):
                    continue
                rec["count"] += 1
                # tenta extrair ano do nome do ficheiro (ex: 2022.csv)
                m = re.search(r'(\d{4})', fn)
                if m:
                    rec["years"].add(int(m.group(1)))

    items = []
    for (sym, iv), r in items_map.items():
        years = sorted(list(r["years"]))
        items.append({
            "symbol": sym,
            "interval": iv,
            "years": years,
            "count": r["count"],
            "path": r["path"],
        })
    # ordena por símbolo/intervalo
    items.sort(key=lambda x: (x["symbol"], x["interval"]))
    return {"items": items}