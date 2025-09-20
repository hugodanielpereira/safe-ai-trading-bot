from __future__ import annotations

import csv
import json
import os
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
    return """
    <html>
    <head>
      <title>AI Bot Starter</title>
      <meta charset="utf-8"/>
      <style>
        :root {{ --bg:#0b0b0b; --fg:#eaeaea; --muted:#9aa0a6; --accent:#00d4ff; }}
        body {{ font-family: system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial; margin:24px; background:var(--bg); color:var(--fg); }}
        h1 {{ margin: 0 0 8px 0; font-weight:700; }}
        .row {{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; }}
        input, button {{ padding:8px 10px; background:#141414; color:var(--fg); border:1px solid #222; border-radius:8px; }}
        button {{ cursor:pointer; }}
        .card {{ background:#0f0f0f; border:1px solid #1b1b1b; border-radius:12px; padding:14px; margin-top:12px; }}
        pre {{ background:#111; color:#0f0; padding:12px; border-radius:8px; overflow:auto; max-height:40vh; }}
        table {{ width:100%; border-collapse: collapse; margin-top:8px; }}
        th, td {{ padding:8px 10px; border-bottom:1px solid #1f1f1f; text-align:left; }}
        th {{ color:var(--muted); cursor:pointer; user-select:none; }}
        tr:hover {{ background:#151515; }}
        .muted {{ color:var(--muted); }}
        .pill {{ background:#151515; border:1px solid #222; padding:4px 8px; border-radius:999px; font-size:12px; color:var(--muted); }}
        #chart {{ width:100%; height:240px; }}
      </style>
    </head>
    <body>
      <h1>AI Bot Starter <span class="pill">v{settings.bot_version}</span></h1>
      <p class="muted">Arranque/Paragem + Tabela de comparação de modelos treinados (multi-símbolo).</p>

      <div class="card">
        <div class="row">
          <input id="token" placeholder="X-API-Token" value="" />
          <button onclick='start()'>Start</button>
          <button onclick='stop()'>Stop</button>
          <button onclick='status()'>Status</button>
          <button onclick='logs()'>Logs</button>
          <button onclick='loadMetrics()' style="margin-left:auto">Carregar métricas</button>
        </div>
        <pre id="out"></pre>
      </div>

      <div class="card">
        <h3 style="margin:0 0 6px 0;">Comparação de modelos treinados</h3>
        <div class="muted">Fonte: <code>models/metrics_summary.csv</code> via <code>/metrics</code></div>
        <div id="metricsWrap">
          <table id="metricsTable">
            <thead>
              <tr>
                <th onclick="sortBy('symbol')">symbol</th>
                <th onclick="sortBy('interval')">interval</th>
                <th onclick="sortBy('rows')">rows</th>
                <th onclick="sortBy('cv_accuracy')">cv_accuracy</th>
                <th>model_path</th>
              </tr>
            </thead>
            <tbody id="metricsBody"></tbody>
          </table>
        </div>
        <div id="emptyMsg" class="muted" style="display:none; margin-top:8px;">Sem métricas ainda. Corre <code>make train-multi</code> ou <code>make retrain-multi</code>.</div>
      </div>

      <div class="card" id="detailCard" style="display:none;">
        <h3 id="detailTitle" style="margin:0 0 6px 0;">Detalhe</h3>
        <div id="detailMeta" class="muted"></div>
        <canvas id="chart"></canvas>
        <div id="detailJson" style="margin-top:8px;"></div>
      </div>

      <script>
        const $ = (id)=>document.getElementById(id);
        function api(path, opts={{}}){
          const t = $('token').value.trim();
          const headers = Object.assign({ 'Content-Type':'application/json' }, t ? {{'X-API-Token': t}} : {});
          return fetch(path, Object.assign({ headers }, opts));
        }

        async function start(){ const r = await api('/start', {{method:'POST'}}); $('out').textContent = await r.text(); }
        async function stop(){ const r = await api('/stop', {{method:'POST'}}); $('out').textContent = await r.text(); }
        async function status(){ const r = await api('/status'); $('out').textContent = await r.text(); }
        async function logs(){ const r = await api('/logs'); $('out').textContent = await r.text(); }

        let metrics = [];
        let sortKey = 'cv_accuracy';
        let sortAsc = false;

        function renderTable(){
          const body = $('metricsBody');
          body.innerHTML = '';
          if(!metrics.length){ $('emptyMsg').style.display='block'; return; }
          $('emptyMsg').style.display='none';
          const rows = metrics.slice().sort((a,b)=>{
            const ak=a[sortKey], bk=b[sortKey];
            if(ak===bk) return 0;
            if(typeof ak==='number' && typeof bk==='number') return sortAsc? ak-bk : bk-ak;
            return sortAsc? (''+ak).localeCompare(''+bk) : (''+bk).localeCompare(''+ak);
          });
          for(const r of rows){
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td><a href="#" onclick="loadDetail('${r.symbol}');return false;">${r.symbol}</a></td>
              <td>${r.interval||''}</td>
              <td>${r.rows||0}</td>
              <td>${(r.cv_accuracy??0).toFixed(4)}</td>
              <td><code>${r.model_path||''}</code></td>
            `;
            body.appendChild(tr);
          }
        }

        function sortBy(key){
          if(sortKey===key) sortAsc=!sortAsc; else {{sortKey=key; sortAsc=true;}}
          renderTable();
        }

        async function loadMetrics(){
          $('detailCard').style.display='none';
          $('out').textContent = 'A carregar métricas...';
          try{
            const r = await api('/metrics');
            const data = await r.json();
            metrics = (data && data.summary) ? data.summary : [];
            $('out').textContent = JSON.stringify(metrics, null, 2);
            renderTable();
          }catch(e){
            $('out').textContent = 'Erro a carregar métricas: '+e;
          }
        }

        async function loadDetail(symbol){
          $('detailCard').style.display='block';
          $('detailTitle').textContent = `Detalhe: ${symbol}`;
          $('detailMeta').textContent = 'A carregar...';
          $('detailJson').textContent = '';
          try{
            const r = await api('/metrics?symbol='+encodeURIComponent(symbol));
            const d = await r.json();
            $('detailMeta').textContent = `cv_accuracy=${(d.cv_accuracy??0).toFixed(4)}`;
            // desenhar top features (se existirem)
            if(d.top_features && d.top_features.length){ drawBarChart(d.top_features.slice(0,15)); }
            else {{ clearChart(); }}
            $('detailJson').textContent = JSON.stringify(d, null, 2);
          }catch(e){
            $('detailMeta').textContent = 'Erro a carregar detalhe: '+e;
          }
        }

        let chartCtx=null;
        function clearChart(){ const c=$('chart'); const ctx=c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height); }
        function drawBarChart(items){
          const c=$('chart'); const ctx=c.getContext('2d');
          // resize canvas ao container
          const wrapWidth = c.parentElement.clientWidth||800;
          c.width = wrapWidth; c.height = 260;
          ctx.clearRect(0,0,c.width,c.height);
          const labels = items.map(x=>x.feature);
          const values = items.map(x=>x.importance||0);
          const max = Math.max(1, ...values);
          const pad=30, w=c.width-pad*2, h=c.height-pad*2;
          const barGap=6;
          const barW = Math.max(4, Math.floor((w - (values.length-1)*barGap)/values.length));
          // eixo
          ctx.strokeStyle='#333'; ctx.lineWidth=1;
          ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, pad+h); ctx.lineTo(pad+w, pad+h); ctx.stroke();
          // barras
          for(let i=0;i<values.length;i++){{
            const x = pad + i*(barW+barGap);
            const bh = Math.round((values[i]/max)*h);
            const y = pad + (h - bh);
            ctx.fillStyle = '#00d4ff';
            ctx.fillRect(x, y, barW, bh);
          }}
          // rótulos (top 10 apenas p/ não poluir)
          ctx.fillStyle='#9aa0a6'; ctx.font='11px system-ui';
          for(let i=0;i<Math.min(10,labels.length);i++){{
            const x = pad + i*(barW+barGap) + Math.max(0, barW-80)/2;
            ctx.save(); ctx.translate(x, pad+h+12); ctx.rotate(-Math.PI/6);
            ctx.fillText(labels[i], 0, 0); ctx.restore();
          }}
        }
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

def _read_metrics_summary(path: str = "models/metrics_summary.csv"):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # normaliza tipos
            r["rows"] = int(r.get("rows", 0) or 0)
            try:
                r["cv_accuracy"] = float(r.get("cv_accuracy", 0) or 0)
            except Exception:
                r["cv_accuracy"] = 0.0
            rows.append(r)
    # ordena por melhor accuracy desc
    rows.sort(key=lambda x: x.get("cv_accuracy", 0), reverse=True)
    return rows

def _read_symbol_detail(symbol: str):
    meta_path = f"models/metrics_{symbol}.json"
    fi_path = f"models/feature_importances_{symbol}.csv"
    detail = {"symbol": symbol}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                detail.update(json.load(f))
        except Exception as e:
            detail["error_meta"] = str(e)
    if os.path.exists(fi_path):
        try:
            top = []
            with open(fi_path, "r", newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    # ficheiro gerado pelo pandas: "feature,importance"
                    if i == 0 and row and row[0] == "0":
                        # fallback raro; ignora header estranho
                        pass
                    if len(row) >= 2:
                        try:
                            top.append({"feature": row[0], "importance": float(row[1])})
                        except Exception:
                            pass
                    if len(top) >= 25:
                        break
            detail["top_features"] = top
        except Exception as e:
            detail["error_features"] = str(e)
    return detail

@app.get("/metrics")
async def metrics(symbol: Optional[str] = None, _: None = Depends(verify_token)):
    """
    - Sem query: devolve comparação de todos os símbolos (summary).
    - Com ?symbol=BTCUSDT: devolve detalhes desse símbolo (metrics + top features).
    """
    if symbol:
        return _read_symbol_detail(symbol)
    return {"summary": _read_metrics_summary()}

@app.get("/logs")
async def logs():
    return worker.logs()