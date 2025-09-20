# app/worker.py
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Callable
from rich.console import Console
from .config import settings
from .strategy import do_unit_of_work

console = Console()

class BotWorker:
    def __init__(self, interval: float):
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = asyncio.Event()
        self._log_buffer: list[dict] = []  # in-memory for demo; use DB in prod

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def logs(self) -> list[dict]:
        return list(self._log_buffer)[-250:]

    async def _loop(self):
        console.log("Worker loop started")
        try:
            while self._running.is_set():
                try:
                    result = await do_unit_of_work()
                    self._log_buffer.append({"ok": True, "event": result})
                    console.log({"ok": True, "event": result})
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