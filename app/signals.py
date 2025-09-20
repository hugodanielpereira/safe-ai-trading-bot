# app/signals.py
from __future__ import annotations
from typing import Literal
import pandas as pd

SignalSide = Literal["BUY","SELL","HOLD"]

def sma_crossover_signal(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> SignalSide:
    if df is None or df.empty or "close" not in df.columns:
        return "HOLD"
    s_fast = df["close"].rolling(fast).mean()
    s_slow = df["close"].rolling(slow).mean()
    if s_fast.iloc[-2] <= s_slow.iloc[-2] and s_fast.iloc[-1] > s_slow.iloc[-1]:
        return "BUY"
    if s_fast.iloc[-2] >= s_slow.iloc[-2] and s_fast.iloc[-1] < s_slow.iloc[-1]:
        return "SELL"
    return "HOLD"