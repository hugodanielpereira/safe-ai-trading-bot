# app/worker_binance.py
from __future__ import annotations
import asyncio
from typing import Optional
from rich.console import Console
from .config import settings
from .exchanges.binance import BinanceBridge
from .signals import ai_signal, sma_crossover_signal, make_features, load_model, _model_cols

console = Console()

class BotWorker:
    def __init__(self, interval: float):
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = asyncio.Event()
        self._log_buffer: list[dict] = []
        self.bridge = BinanceBridge()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def logs(self) -> list[dict]:
        return list(self._log_buffer)[-250:]

    async def _loop(self):
        console.log("Worker loop started (Binance)")
        try:
            while self._running.is_set():
                try:
                    df = self.bridge.klines_df(interval="1m", limit=50)
                    side = ai_signal(df) if settings.use_ai else sma_crossover_signal(df)  # BUY/SELL/HOLD
                    # --- (NOVO) probabilidades no log quando USE_AI=true ---
                    event = {
                        "symbol": self.bridge.symbol,
                        "side": side,
                        "close": float(df["close"].iloc[-1]),
                    }

                    if settings.use_ai:
                        try:
                            mdl = load_model()
                            if mdl is not None:
                                feats = make_features(df).iloc[[-1]]
                                # alinhar colunas com o treino, se existir a lista salva
                                if _model_cols is not None:
                                    # adiciona colunas em falta com 0.0 e reordena
                                    missing = [c for c in _model_cols if c not in feats.columns]
                                    for m in missing:
                                        feats[m] = 0.0
                                    feats = feats[_model_cols]
                                # tentar predict_proba; se não houver, faz fallback
                                try:
                                    proba = mdl.predict_proba(feats)[0]  # array tipo [p0,p1,p2]
                                    classes = list(getattr(mdl, "classes_", [0, 1, 2]))
                                    prob_map = {int(c): float(p) for c, p in zip(classes, proba)}
                                    # convenção do treino: 0=HOLD, 1=BUY, 2=SELL
                                    event["proba"] = {
                                        "HOLD": prob_map.get(0, 0.0),
                                        "BUY": prob_map.get(1, 0.0),
                                        "SELL": prob_map.get(2, 0.0),
                                    }
                                except Exception:
                                    pred = mdl.predict(feats)[0]
                                    event["proba"] = {
                                        "HOLD": 1.0 if int(pred) == 0 else 0.0,
                                        "BUY":  1.0 if int(pred) == 1 else 0.0,
                                        "SELL": 1.0 if int(pred) == 2 else 0.0,
                                    }
                        except Exception as e:
                            # se falhar, não bloqueia o loop — só regista o erro
                            event["proba_error"] = str(e)
                        # --- fim probabilidades ---
                    if side in ("BUY","SELL") and self.bridge.live:
                        order = self.bridge.market_order(side)
                        event.update({"executed": True, "orderId": order.get("orderId")})
                    else:
                        event.update({"executed": False})
                    self._log_buffer.append({"ok": True, "event": event})
                    console.log({"ok": True, "event": event})
                except Exception as e:
                    self._log_buffer.append({"ok": False, "error": str(e)})
                    console.log({"ok": False, "error": str(e)})
                await asyncio.sleep(self.interval)
        finally:
            console.log("Worker loop stopped")

    async def start(self):
        if self.running:
            return
        self._running.set()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if not self.running:
            return
        self._running.clear()
        await self._task
        self._task = None

worker = BotWorker(settings.loop_interval_seconds)