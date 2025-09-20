from __future__ import annotations

from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .config import settings
from .security import verify_token
from .worker_binance import worker

# tentar apanhar exceções específicas da lib da Binance
try:
    from binance.exceptions import BinanceAPIException as ClientError
except Exception:
    try:
        from binance.error import ClientError  # fallback em versões antigas
    except Exception:
        ClientError = Exception

app = FastAPI(title="AI Bot Starter", version=settings.bot_version)

class StartResponse(BaseModel):
    status: str
    version: str

class StopResponse(BaseModel):
    status: str

class StatusResponse(BaseModel):
    running: bool
    version: str

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    # f-string + chaves JS escapadas ({{ }})
    return f"""
    <html>
      <head><title>AI Bot Starter</title></head>
      <body style='font-family:system-ui;margin:2rem;'>
        <h1>AI Bot Starter (v{settings.bot_version})</h1>
        <p>Two functions only: <b>Start</b> and <b>Stop</b>. Token protected.</p>
        <div style='display:flex;gap:1rem'>
          <form onsubmit="start();return false;">
            <input id='token' placeholder='X-API-Token' />
            <button type='submit'>Start</button>
          </form>
          <form onsubmit="stop();return false;">
            <input id='token2' placeholder='X-API-Token' />
            <button type='submit'>Stop</button>
          </form>
          <button onclick='status()'>Refresh status</button>
          <button onclick='logs()'>Load logs</button>
        </div>
        <pre id='out' style='margin-top:1rem;background:#111;color:#0f0;padding:1rem;'></pre>
        <script>
          async function start(){{ 
            const t=document.getElementById('token').value;
            const r=await fetch('/start',{{method:'POST',headers:{{'X-API-Token':t}}}});
            document.getElementById('out').textContent=await r.text();
          }}
          async function stop(){{ 
            const t=document.getElementById('token2').value;
            const r=await fetch('/stop',{{method:'POST',headers:{{'X-API-Token':t}}}});
            document.getElementById('out').textContent=await r.text();
          }}
          async function status(){{ 
            const r=await fetch('/status');
            document.getElementById('out').textContent=await r.text();
          }}
          async function logs(){{ 
            const r=await fetch('/logs');
            document.getElementById('out').textContent=await r.text();
          }}
        </script>
      </body>
    </html>
    """

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
    limit: int = Query(10, ge=1, le=1000),
):
    b = worker.bridge
    sym = symbol or b.symbol
    try:
        return b.client.get_my_trades(symbol=sym, limit=limit)
    except ClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/orders")
async def orders(_: None = Depends(verify_token), limit: int = Query(10, ge=1, le=1000)):
    b = worker.bridge
    try:
        data = b.client.get_all_orders(symbol=b.symbol, limit=limit)
        return data[-limit:]
    except ClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ticker")
async def ticker(symbol: Optional[str] = None):
    b = worker.bridge
    sym = symbol or b.symbol
    try:
        return b.client.get_symbol_ticker(symbol=sym)
    except ClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
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
    except Exception as e:
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
    }

@app.get("/logs")
async def logs():
    return worker.logs()