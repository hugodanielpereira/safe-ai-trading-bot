# app/train.py
from __future__ import annotations

import os, json, math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from joblib import dump

from .signals import make_features  # usa exatamente as mesmas features do runtime

METRICS_CSV = "models/metrics_summary.csv"

@dataclass
class TrainConfig:
    symbol: str
    interval: str
    algo: str = "gbm"
    params: Optional[Dict[str, Any]] = None
    lookahead: int = 1              # quantos candles à frente para rotular
    neutral_band_bps: float = 5.0   # banda neutra em bps (ex.: 5 = 0.05%)
    # se quiseres, podes guardar aqui fee/slippage para calibrar a banda

def _ensure_dir(p: str):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)

def _first_time_col(df: pd.DataFrame) -> str | None:
    for c in ("close_time","open_time","time","timestamp","date"):
        if c in df.columns: return c
    return None

def _time_split_index(n: int, val_ratio: float = 0.2):
    n_val = max(1, int(n * val_ratio))
    n_tr  = max(1, n - n_val)
    idx = np.arange(n)
    return idx[:n_tr], idx[n_tr:]

def _build_model(algo: str, params: Dict[str, Any] | None) -> Pipeline:
    if (algo or "").lower() in ("gbm","gbrt","gradientboosting"):
        base = GradientBoostingClassifier(
            n_estimators=(params or {}).get("n_estimators", 300),
            learning_rate=(params or {}).get("learning_rate", 0.05),
            max_depth=(params or {}).get("max_depth", 3),
            subsample=(params or {}).get("subsample", 0.8),
            random_state=(params or {}).get("random_state", 42),
        )
    else:
        base = GradientBoostingClassifier(random_state=42)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler(with_mean=True, with_std=True)),
        ("model",   base),
    ])

def _multiclass_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, Any]:
    y_pred = proba.argmax(axis=1)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    # AUC multi-classe (pode falhar se faltar uma classe no split)
    try:
        auc = roc_auc_score(y_true, proba, multi_class="ovo", average="macro")
    except Exception:
        auc = None
    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "roc_auc_macro": float(auc) if auc is not None and not math.isnan(auc) else None,
    }

def _append_metrics_summary(row: Dict[str, Any]):
    _ensure_dir(METRICS_CSV)
    cols = ["symbol","interval","rows","cv_accuracy","model_path"]
    df_row = pd.DataFrame([[row.get(c) for c in cols]], columns=cols)
    try:
        if os.path.exists(METRICS_CSV):
            old = pd.read_csv(METRICS_CSV)
            out = pd.concat([old, df_row], ignore_index=True)
        else:
            out = df_row
        out.to_csv(METRICS_CSV, index=False)
    except Exception:
        # não quebra treino por causa do CSV
        pass

def _make_labels(df: pd.DataFrame, lookahead: int, neutral_band_bps: float) -> pd.Series:
    """
    0=HOLD, 1=BUY, 2=SELL
    Usa variação percentual a 'lookahead' candles (close_{+h} / close - 1).
    BUY  se r > +band; SELL se r < -band; HOLD no intervalo [-band, +band].
    """
    band = float(neutral_band_bps) / 10000.0
    fwd = df["close"].shift(-lookahead)
    r = (fwd / df["close"]) - 1.0
    y = pd.Series(0, index=df.index, dtype=int)
    y[r >  band] = 1
    y[r < -band] = 2
    return y

def train_model(prices: pd.DataFrame, cfg: TrainConfig, out_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Treina um modelo multi-classe compatível com signals._proba_map:
      classes_: [0 (HOLD), 1 (BUY), 2 (SELL)]
    Guarda bundle {"model": pipeline, "columns": feats}.
    """
    if prices is None or prices.empty:
        raise ValueError("Sem dados para treino")

    # ordenar por tempo (caso seja necessário)
    tcol = _first_time_col(prices)
    if tcol:
        prices = prices.sort_values(tcol)

    # mesmas features do runtime
    X_full = make_features(prices)
    y_full = _make_labels(prices, lookahead=cfg.lookahead, neutral_band_bps=cfg.neutral_band_bps)

    # alinhar índices e limpar NaNs (rolling/shift)
    valid = X_full.notna().all(axis=1) & y_full.notna()
    X = X_full.loc[valid].copy()
    y = y_full.loc[valid].astype(int).copy()

    n = len(X)
    if n < 300:
        raise ValueError(f"Poucos dados após features/labels: {n}")

    tr_idx, va_idx = _time_split_index(n, val_ratio=0.2)
    Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
    Xva, yva = X.iloc[va_idx], y.iloc[va_idx]

    pipe = _build_model(cfg.algo, cfg.params or {})
    pipe.fit(Xtr, ytr)

    # probabilidades (para métricas e para garantir compatibilidade)
    try:
        proba = pipe.predict_proba(Xva)
    except Exception:
        # fallback para modelos sem predict_proba (não deve acontecer com GBM do sklearn)
        pred = pipe.predict(Xva)
        # one-hot tosca
        proba = np.zeros((len(pred), 3), dtype=float)
        for i, c in enumerate(pred.astype(int)):
            proba[i, max(0, min(2, c))] = 1.0

    m = _multiclass_metrics(yva.to_numpy(), proba)

    # guardar bundle + meta
    _ensure_dir(out_path)
    bundle = {"model": pipe, "columns": list(X.columns)}
    dump(bundle, out_path)

    meta = {
        "symbol": cfg.symbol,
        "interval": cfg.interval,
        "algo": cfg.algo,
        "params": cfg.params or {},
        "features": list(X.columns),
        "rows_total": int(n),
        "lookahead": cfg.lookahead,
        "neutral_band_bps": cfg.neutral_band_bps,
        "val_metrics": m,
        "classes_": [0,1,2],
    }
    with open(out_path.replace(".pkl", ".json"), "w") as f:
        json.dump(meta, f, indent=2)

    # atualizar summary (aparece no /metrics e na tua UI)
    _append_metrics_summary({
        "symbol": cfg.symbol,
        "interval": cfg.interval,
        "rows": int(n),
        "cv_accuracy": float(m.get("accuracy", 0.0)),
        "model_path": out_path,
    })

    return out_path, {"rows": int(n), **m, "model_path": out_path}