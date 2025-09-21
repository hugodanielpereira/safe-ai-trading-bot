# app/exchanges/binance.py

from __future__ import annotations
import os
import pandas as pd
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET

SPOT_TESTNET_API_URL = "https://testnet.binance.vision/api"
FUTURES_TESTNET_API_URL = "https://testnet.binancefuture.com/fapi"

class BinanceBridge:
    def __init__(self):
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.use_futures = os.getenv("BINANCE_FUTURES", "false").lower() == "true"
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

        self.symbol = os.getenv("TRADE_SYMBOL", "BTCUSDT")
        self.qty = float(os.getenv("TRADE_QTY", "0.001"))
        self.live = os.getenv("TRADE_LIVE", "false").lower() == "true"

        # cria o client
        self.client = Client(api_key, api_secret)

        # configura URL base conforme flags
        if testnet:
            if self.use_futures:
                # Futures testnet
                # python-binance usa .FUTURES_URL em muitas versões; definimos diretamente
                self.client.FUTURES_URL = FUTURES_TESTNET_API_URL
            else:
                # Spot testnet
                # Nem todas as versões têm .TESTNET_API_URL, por isso definimos a API_URL diretamente
                self.client.API_URL = SPOT_TESTNET_API_URL
        # se testnet = false, mantemos os defaults de mainnet

    def klines_df(self, interval: str = "1m", limit: int = 50, symbol: str | None = None) -> pd.DataFrame:
            """Devolve DataFrame de klines para `symbol` (ou self.symbol se None)."""
            sym = symbol or self.symbol

            # escolhe o endpoint correto (spot vs futures)
            if hasattr(self.client, "futures_klines") and getattr(self.client, "FUTURES_URL", ""):
                data = self.client.futures_klines(symbol=sym, interval=interval, limit=limit)
            else:
                data = self.client.get_klines(symbol=sym, interval=interval, limit=limit)

            cols = ["open_time","open","high","low","close","volume",
                    "close_time","qav","trades","taker_base","taker_quote","ignore"]
            df = pd.DataFrame(data, columns=cols)
            # tipos & colunas úteis
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            for c in ("open","high","low","close","volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["open_time","high","low","close","volume","close_time"]]

    def market_order(self, side: str):
        side_binance = SIDE_BUY if side == "BUY" else SIDE_SELL
        if self.use_futures:
            return self.client.futures_create_order(
                symbol=self.symbol, side=side_binance, type=ORDER_TYPE_MARKET, quantity=self.qty
            )
        else:
            return self.client.create_order(
                symbol=self.symbol, side=side_binance, type=ORDER_TYPE_MARKET, quantity=self.qty
            )