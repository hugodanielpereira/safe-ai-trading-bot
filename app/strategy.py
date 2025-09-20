
# app/strategy.py
# Placeholder, safe-by-default worker "task"
from __future__ import annotations
import httpx
from .config import settings

async def do_unit_of_work():
    """Fetch a public price and return a compact dict. Swap with your own logic later."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(settings.public_price_api)
        r.raise_for_status()
        data = r.json()
    # Minimum-normalized payload (example)
    result = {
        "source": "coindesk-bpi",
        "usd": data.get("bpi", {}).get("USD", {}).get("rate_float"),
        "time": data.get("time", {}).get("updatedISO"),
    }
    return result