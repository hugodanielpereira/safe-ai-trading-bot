# safe-ai-trading-bot

// README (quick)
// ----------------
// 1) Create a new folder and copy these files (see below). 
// 2) `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
// 3) Create `.env` (see template in this file). 
// 4) Run: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
// 5) Use the simple dashboard at http://localhost:8000 to Start/Stop the bot and view logs.
// 6) Optional: `docker compose up --build`
//
// This skeleton mirrors the video’s flow but keeps everything safe and auditable. 
// The worker is a placeholder that fetches public market data (no funds, no keys) until you swap in a legit API.
// There are exactly TWO main actions: start() and stop().

/* ======================== NOTES: map to video steps ========================
- “Install 2 extensions / dev mode”: here we skip any risky browser plugins. Use this audited API server instead.
- “Paste script, check version at top”: BOT_VERSION controls compatibility banners; surfaced at / and /status.
- “Green check in console”: If /status returns {running:true} after /start, you’re good.
- “Connect extension / account shows up”: we use a token header. Treat that as your secure connection.
- “Two functions only”: /start and /stop.
- “Paste required amount”: DO NOT deposit funds into unknown scripts. Keep this bot on public/test data until you integrate a reputable API in code.
- “Cloud upload”: Docker image → run on server/VPS.
========================================================================== */