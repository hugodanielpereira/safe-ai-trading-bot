from __future__ import annotations
import asyncio, time
from typing import Optional
from rich.console import Console

from .config import settings
from .exchanges.binance import BinanceBridge
from .signals import (
    ai_signal_with_proba,
    sma_crossover_signal,
    load_model,
    current_model_info,
)

console = Console()

class BotWorker:
    def __init__(self, interval: float):
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = asyncio.Event()
        self._log_buffer: list[dict] = []
        self.bridge = BinanceBridge()

        # --- novos travões / estado
        self.is_long = False            # posição atual (spot: comprado = True)
        self.last_exec_ts = 0.0         # timestamp última ordem
        self.cooldown_sec = getattr(settings, "cooldown_seconds", 60)  # podes pôr no .env
        self.min_proba_gap = getattr(settings, "min_proba_gap", 0.10)  # diferença mínima entre BUY e SELL

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def logs(self) -> list[dict]:
        return list(self._log_buffer)[-250:]

    def _can_execute(self) -> bool:
        return (time.time() - self.last_exec_ts) >= self.cooldown_sec

    def _update_position(self, side: str):
        # Spot simples: se comprou, fica long; se vendeu, fica flat.
        if side == "BUY":
            self.is_long = True
        elif side == "SELL":
            self.is_long = False

    async def _loop(self):
        console.log("Worker loop started (Binance)")
        try:
            load_model()
            console.log({"model_info": current_model_info()})
        except Exception as e:
            console.log({"model_load_error": str(e)})

        try:
            while self._running.is_set():
                try:
                    df = self.bridge.klines_df(interval="1m", limit=50)

                    if settings.use_ai:
                        side, proba = ai_signal_with_proba(df)
                    else:
                        side = sma_crossover_signal(df)
                        proba = {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}

                    # Margem mínima de confiança entre BUY e SELL
                    p_buy = float(proba.get("BUY", 0.0))
                    p_sell = float(proba.get("SELL", 0.0))
                    gap = abs(p_buy - p_sell)

                    # Gate de execução
                    do_exec = False
                    reason = "hold"
                    if side == "BUY" and (not self.is_long) and self._can_execute() and gap >= self.min_proba_gap:
                        do_exec = True
                        reason = "buy_ok"
                    elif side == "SELL" and self.is_long and self._can_execute() and gap >= self.min_proba_gap:
                        do_exec = True
                        reason = "sell_ok"

                    event = {
                        "symbol": self.bridge.symbol,
                        "side": side,
                        "close": float(df["close"].iloc[-1]),
                        "proba": proba,
                        "executed": False,
                        "reason": reason,
                        "is_long": self.is_long,
                    }

                    if do_exec and self.bridge.live:
                        try:
                            order = self.bridge.market_order(side)
                            event.update({"executed": True, "orderId": order.get("orderId")})
                            self.last_exec_ts = time.time()
                            self._update_position(side)
                            event["is_long"] = self.is_long
                        except Exception as e:
                            event["error"] = str(e)

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

# instancia única
worker = BotWorker(settings.loop_interval_seconds)