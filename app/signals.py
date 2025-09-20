# app/signals.py
from __future__ import annotations
from typing import Literal
import os
import pandas as pd
import numpy as np
from .config import settings

SignalSide = Literal["BUY","SELL","HOLD"]

# ---- baseline SMA (fallback)
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

# ---- lightweight feature engineering (no external TA deps)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df.get("high", df["close"])  # accept minimal df
    low = df.get("low", df["close"])
    close = df["close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ret1"] = df["close"].pct_change()
    out["ret5"] = df["close"].pct_change(5)
    out["ret15"] = df["close"].pct_change(15)
    out["vol"] = df.get("volume", pd.Series(0, index=df.index))
    out["rsi14"] = rsi(df["close"], 14)
    out["atr14"] = atr(df.assign(high=df.get("high", df["close"]), low=df.get("low", df["close"])), 14) / df["close"]
    out["sma9"] = df["close"].rolling(9).mean() / df["close"]
    out["sma21"] = df["close"].rolling(21).mean() / df["close"]
    out = out.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0.0)
    return out

# ---- AI model loader (LightGBM via scikit)
_model = None
_model_cols: list[str] | None = None

def load_model(path: str = None):
    global _model, _model_cols
    if _model is not None:
        return _model
    path = path or settings.model_path
    if os.path.exists(path):
        import joblib
        bundle = joblib.load(path)
        # bundle can be (model, columns) or dict
        if isinstance(bundle, dict):
            _model = bundle["model"]
            _model_cols = bundle.get("columns")
        elif isinstance(bundle, (list, tuple)) and len(bundle) == 2:
            _model, _model_cols = bundle
        else:
            _model = bundle
            _model_cols = None
    return _model


def ai_signal(df: pd.DataFrame) -> SignalSide:
    mdl = load_model()
    if mdl is None:
        return sma_crossover_signal(df)
    feats = make_features(df)
    X = feats.iloc[[-1]]  # last row
    if _model_cols is not None:
        missing = [c for c in _model_cols if c not in X.columns]
        for m in missing:
            X[m] = 0.0
        X = X[_model_cols]
    # predict proba: assume class order [HOLD, BUY, SELL] or similar; we standardize
    # To avoid class-order ambiguity, expect model to expose classes_
    try:
        proba = mdl.predict_proba(X)[0]
        classes = list(getattr(mdl, "classes_", [0,1,2]))
        prob_map = {int(c): float(p) for c, p in zip(classes, proba)}
        # convention: 0=HOLD, 1=BUY, 2=SELL
        p_buy = prob_map.get(1, 0.0)
        p_sell = prob_map.get(2, 0.0)
    except Exception:
        # fallback to decision_function or prediction
        pred = mdl.predict(X)[0]
        if int(pred) == 1:
            p_buy, p_sell = 1.0, 0.0
        elif int(pred) == 2:
            p_buy, p_sell = 0.0, 1.0
        else:
            p_buy, p_sell = 0.0, 0.0
    if p_buy >= settings.buy_threshold and p_buy > p_sell:
        return "BUY"
    if p_sell >= settings.sell_threshold and p_sell > p_buy:
        return "SELL"
    return "HOLD"