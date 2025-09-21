# app/backtest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, List
import math

import joblib
import numpy as np
import pandas as pd

from .signals import make_features, sma_crossover_signal

Strategy = Literal["ai", "sma"]

@dataclass
class BTConfig:
    strategy: Strategy = "ai"
    model_path: Optional[str] = None
    buy_th: float = 0.55
    sell_th: float = 0.55
    fee_bps: float = 1.0        # comissão em basis points
    slippage_bps: float = 0.0   # slippage em bps

def _align_to_model_columns(feats: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """
    Garante que o DataFrame tem exatamente as colunas usadas no treino.
    - Colunas em falta -> 0.0
    - Colunas a mais -> descartadas
    - Ordem -> igual à do treino
    """
    cols = bundle.get("columns")
    if not cols:
        # bundle pode ser só o modelo sem metadados de colunas (menos ideal)
        return feats
    X = feats.copy()
    missing = [c for c in cols if c not in X.columns]
    for m in missing:
        X[m] = 0.0
    # reordenar / subselecionar
    X = X[cols]
    # defensivo contra inf/NaN
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return X

def _apply_costs(px: float, side: str, fee_bps: float, slip_bps: float) -> float:
    """
    Aplica custos de execução a um preço:
    - BUY: preço piora (sobe) com slippage; fee aumenta custo
    - SELL: preço piora (desce) com slippage; fee também impacta
    Implementação simples: aplica variação percentual simétrica.
    """
    fee = fee_bps / 10_000.0
    slip = slip_bps / 10_000.0
    if side == "BUY":
        return px * (1 + slip + fee)
    else:
        return px * (1 - slip - fee)

def _equity_curve(prices: pd.Series, signals: pd.Series,
                  fee_bps: float, slip_bps: float) -> Dict[str, Any]:
    """
    Backtest “vectorizado” simples, posição 0/1/-1 (BUY/HOLD/SELL) com custo por trade.
    `signals` deve estar em {1, 0, -1} e estar *shiftado* (trade na barra seguinte).
    """
    closes = prices.values.astype(float)
    pos = signals.shift(1).fillna(0).values.astype(float)   # entrar na próxima barra
    # retornos “brutos”
    ret = np.zeros_like(closes, dtype=float)
    ret[1:] = (closes[1:] - closes[:-1]) / np.where(closes[:-1]==0, 1e-12, closes[:-1])

    # custo por mudança de posição
    pos_change = np.abs(np.diff(pos, prepend=0.0))
    # custo efetivo por “switch” de posição:
    # aproximamos custo por ida e volta: aplica fee+slip quando muda sinal
    cost_per_switch = (fee_bps + slip_bps) / 10_000.0
    net = pos * ret - pos_change * cost_per_switch

    equity = (1.0 + net).cumprod()
    # métricas
    total_return = float(equity[-1] - 1.0)
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity / np.where(cummax==0, 1e-12, cummax)) - 1.0
    max_dd = float(drawdown.min())
    # Sharpe diário aproximado: supõe ~1440 barras/dia para 1m; adapta se quiseres
    # Para não estourar se std ~0:
    std = float(np.std(net))
    sharpe = float((np.mean(net) / (std if std > 1e-12 else 1e-12)) * math.sqrt(1440))

    # série para o frontend
    series = [{"t": int(i), "equity": float(eq)} for i, eq in enumerate(equity)]
    return dict(n=int(len(closes)),
                total_return=total_return,
                max_drawdown=max_dd,
                sharpe=sharpe,
                equity=series)

def run_backtest(prices_df: pd.DataFrame, cfg: BTConfig) -> Dict[str, Any]:
    """
    prices_df: DataFrame com colunas pelo menos ["close"] (idealmente high, low, volume).
    cfg: parâmetros de estratégia/custos.
    """
    if "close" not in prices_df.columns:
        raise ValueError("CSV precisa da coluna 'close'")

    # features “inline” (iguais às usadas no live)
    feats = make_features(prices_df)

    if cfg.strategy == "sma":
        # gerar sinais -1/0/1 com o baseline SMA
        sig_series = []
        for i in range(len(prices_df)):
            # usa apenas histórico até ao i (evita lookahead)
            df_slice = prices_df.iloc[: i + 1]
            side = sma_crossover_signal(df_slice)
            if side == "BUY":
                sig_series.append(1)
            elif side == "SELL":
                sig_series.append(-1)
            else:
                sig_series.append(0)
        signals = pd.Series(sig_series, index=prices_df.index).astype(float)
        return dict(strategy="sma", **_equity_curve(prices_df["close"], signals, cfg.fee_bps, cfg.slippage_bps))

    # === AI ===
    if not cfg.model_path or not isinstance(cfg.model_path, str):
        raise FileNotFoundError("model_path não fornecido para strategy='ai'")

    bundle = joblib.load(cfg.model_path)
    mdl = bundle.get("model", bundle)
    X = _align_to_model_columns(feats, bundle)

    # prever proba para cada linha (vectorizado)
    # Nota: NÃO desativamos a verificação de shape — alinhamos colunas corretamente
    proba = mdl.predict_proba(X)

    # classes esperadas: 0=HOLD, 1=BUY, 2=SELL
    classes = list(getattr(mdl, "classes_", [0, 1, 2]))
    idx_H = classes.index(0) if 0 in classes else None
    idx_B = classes.index(1) if 1 in classes else None
    idx_S = classes.index(2) if 2 in classes else None

    p_buy = proba[:, idx_B] if idx_B is not None else np.zeros(len(X))
    p_sell = proba[:, idx_S] if idx_S is not None else np.zeros(len(X))

    # regra de decisão
    decision = np.zeros(len(X), dtype=float)
    buy_mask = (p_buy >= cfg.buy_th) & (p_buy > p_sell)
    sell_mask = (p_sell >= cfg.sell_th) & (p_sell > p_buy)
    decision[buy_mask] = 1.0
    decision[sell_mask] = -1.0
    signals = pd.Series(decision, index=prices_df.index)

    return dict(strategy="ai", **_equity_curve(prices_df["close"], signals, cfg.fee_bps, cfg.slippage_bps))