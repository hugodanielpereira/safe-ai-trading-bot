# app/worker_binance.py
from __future__ import annotations
import asyncio
from typing import Optional
from rich.console import Console
from .config import settings
from .exchanges.binance import BinanceBridge
from .signals import sma_crossover_signal

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
                    side = sma_crossover_signal(df)  # BUY/SELL/HOLD
                    event = {"symbol": self.bridge.symbol, "side": side, "close": float(df["close"].iloc[-1])}
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