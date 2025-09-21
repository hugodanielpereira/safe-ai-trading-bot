# app/signals.py
from __future__ import annotations

from typing import Literal, Optional, Dict, Tuple, List
import os
import numpy as np
import pandas as pd
import joblib

from .config import settings

SignalSide = Literal["BUY", "SELL", "HOLD"]

# =============================================================================
# 1) Regras simples (fallback): SMA crossover
# =============================================================================
def sma_crossover_signal(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> SignalSide:
    if df is None or df.empty or "close" not in df.columns:
        return "HOLD"
    s_fast = df["close"].rolling(fast).mean()
    s_slow = df["close"].rolling(slow).mean()
    if len(s_fast) < slow + 2:
        return "HOLD"
    if s_fast.iloc[-2] <= s_slow.iloc[-2] and s_fast.iloc[-1] > s_slow.iloc[-1]:
        return "BUY"
    if s_fast.iloc[-2] >= s_slow.iloc[-2] and s_fast.iloc[-1] < s_slow.iloc[-1]:
        return "SELL"
    return "HOLD"


# =============================================================================
# 2) Feature engineering leve (sem TA-lib)
# =============================================================================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df.get("high", df["close"])
    low = df.get("low", df["close"])
    close = df["close"].shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - close).abs(),
            (low - close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera um pequeno conjunto de features numéricas robustas para 1m.
    """
    out = pd.DataFrame(index=df.index)
    out["ret1"] = df["close"].pct_change()
    out["ret5"] = df["close"].pct_change(5)
    out["ret15"] = df["close"].pct_change(15)
    out["vol"] = df.get("volume", pd.Series(0, index=df.index))
    out["rsi14"] = rsi(df["close"], 14)
    _df_for_atr = df.assign(
        high=df.get("high", df["close"]),
        low=df.get("low", df["close"]),
    )
    out["atr14"] = atr(_df_for_atr, 14) / df["close"]
    out["sma9"] = df["close"].rolling(9).mean() / df["close"]
    out["sma21"] = df["close"].rolling(21).mean() / df["close"]
    out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return out


# =============================================================================
# 3) Modelo ML (LightGBM / scikit) — cache única em memória
# =============================================================================
_MODEL = None
_COLUMNS: Optional[List[str]] = None


def _assign_bundle(bundle) -> None:
    """
    Aceita formatos:
      - dict: {"model": mdl, "columns": [...]}
      - tuple/list: (mdl, columns)
      - mdl direto
    e atualiza _MODEL/_COLUMNS.
    """
    global _MODEL, _COLUMNS
    if isinstance(bundle, dict):
        _MODEL = bundle.get("model", bundle)
        _COLUMNS = bundle.get("columns")
    elif isinstance(bundle, (list, tuple)) and len(bundle) == 2:
        _MODEL, _COLUMNS = bundle
    else:
        _MODEL = bundle
        _COLUMNS = None


def load_model(path: Optional[str] = None):
    """
    Carrega o modelo (uma vez) para a cache global. Se já estiver carregado, devolve-o.
    """
    global _MODEL, _COLUMNS
    if _MODEL is not None and (path is None or path == settings.model_path):
        return _MODEL

    path = path or settings.model_path
    if path and os.path.exists(path):
        bundle = joblib.load(path)
        _assign_bundle(bundle)
    return _MODEL


def set_model_from_joblib(path: str) -> Dict:
    """
    Força (re)carregar um modelo a partir de um .pkl e atualiza a cache.
    Usado pelo endpoint /set_symbol.
    """
    bundle = joblib.load(path)
    _assign_bundle(bundle)
    # refletir no settings em runtime (se permitido)
    try:
        settings.model_path = path  # se for Pydantic sem "frozen", isto funciona
    except Exception:
        pass

    return {
        "model_loaded": _MODEL is not None,
        "n_features": len(_COLUMNS) if _COLUMNS is not None else None,
        "path": path,
        "classes_": getattr(_MODEL, "classes_", None) if _MODEL is not None else None,
    }


def current_model_info() -> Dict:
    return {
        "loaded": _MODEL is not None,
        "n_features": len(_COLUMNS) if _COLUMNS is not None else None,
        "classes_": getattr(_MODEL, "classes_", None) if _MODEL is not None else None,
        "path": getattr(settings, "model_path", None),
    }


# =============================================================================
# 4) Utilitários de predição
# =============================================================================
def _prepare_latest_feature_row(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Constrói a última linha de features X (1xN) já alinhada com _COLUMNS, preenchendo
    eventuais colunas em falta com 0.0 para robustez.
    """
    feats = make_features(df)
    if feats.empty:
        return None

    X = feats.iloc[[-1]].copy()

    if _COLUMNS is not None:
        missing = [c for c in _COLUMNS if c not in X.columns]
        for m in missing:
            X[m] = 0.0
        X = X[_COLUMNS]
    return X


def _proba_map(mdl, X: pd.DataFrame) -> Dict[str, float]:
    """
    Retorna probabilidades num dict com chaves 'HOLD','BUY','SELL'.
    Convenção de classes do treino: 0=HOLD, 1=BUY, 2=SELL.
    Se o modelo não tiver predict_proba, faz um fallback simples.
    """
    # valores default
    out = {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}

    try:
        proba = mdl.predict_proba(X)[0]
        classes = list(getattr(mdl, "classes_", [0, 1, 2]))
        tmp = {int(c): float(p) for c, p in zip(classes, proba)}
        out["HOLD"] = tmp.get(0, 0.0)
        out["BUY"] = tmp.get(1, 0.0)
        out["SELL"] = tmp.get(2, 0.0)
        # normalização defensiva
        s = out["HOLD"] + out["BUY"] + out["SELL"]
        if s > 0:
            for k in out:
                out[k] = float(out[k] / s)
        return out
    except Exception:
        # fallback: usa apenas a classe prevista
        try:
            pred = int(mdl.predict(X)[0])
        except Exception:
            pred = 0
        if pred == 1:
            out["BUY"] = 1.0
        elif pred == 2:
            out["SELL"] = 1.0
        else:
            out["HOLD"] = 1.0
        return out


# =============================================================================
# 5) Sinais: simples e com probabilidades
# =============================================================================
def ai_signal(df: pd.DataFrame) -> SignalSide:
    """
    Sinal baseado em modelo ML + thresholds do settings.
    Se não houver modelo carregado, cai para SMA crossover.
    """
    mdl = load_model()
    if mdl is None:
        return sma_crossover_signal(df)

    X = _prepare_latest_feature_row(df)
    if X is None:
        return "HOLD"

    p = _proba_map(mdl, X)
    p_buy = p.get("BUY", 0.0)
    p_sell = p.get("SELL", 0.0)

    if p_buy >= settings.buy_threshold and p_buy > p_sell:
        return "BUY"
    if p_sell >= settings.sell_threshold and p_sell > p_buy:
        return "SELL"
    return "HOLD"


def ai_signal_with_proba(df: pd.DataFrame) -> Tuple[SignalSide, Dict[str, float]]:
    """
    Mesma lógica do ai_signal, mas devolve também o dict de probabilidades.
    Útil para logging/telemetria no worker.
    """
    mdl = load_model()
    if mdl is None:
        return sma_crossover_signal(df), {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}

    X = _prepare_latest_feature_row(df)
    if X is None:
        return "HOLD", {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}

    p = _proba_map(mdl, X)
    p_buy = p.get("BUY", 0.0)
    p_sell = p.get("SELL", 0.0)

    if p_buy >= settings.buy_threshold and p_buy > p_sell:
        return "BUY", p
    if p_sell >= settings.sell_threshold and p_sell > p_buy:
        return "SELL", p
    return "HOLD", p