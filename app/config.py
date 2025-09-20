# app/config.py
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()  # <-- ADICIONA ISTO

class Settings(BaseModel):
    bot_api_token: str = os.getenv("BOT_API_TOKEN", "supersecrettoken")
    bot_version: str = os.getenv("BOT_VERSION", "1.0.0")
    public_price_api: str = os.getenv("PUBLIC_PRICE_API", "https://api.coindesk.com/v1/bpi/currentprice.json")
    loop_interval_seconds: float = float(os.getenv("LOOP_INTERVAL_SECONDS", "5"))

settings = Settings()