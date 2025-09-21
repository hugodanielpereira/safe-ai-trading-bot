# app/config.py
from pydantic import BaseModel, ConfigDict
import os
from dotenv import load_dotenv

load_dotenv()  # <-- ADICIONA ISTO

# ai switches + thresholds
#   USE_AI: when true, worker uses model-based signals
#   MODEL_PATH: path to trained LightGBM model
#   BUY_THRESHOLD / SELL_THRESHOLD: probabilities
#   FEATURE_HORIZON: lookahead horizon used at training (informative)

class Settings(BaseModel):
    bot_api_token: str = os.getenv("BOT_API_TOKEN", "supersecrettoken")
    bot_version: str = os.getenv("BOT_VERSION", "1.0.0")
    public_price_api: str = os.getenv("PUBLIC_PRICE_API", "https://api.coindesk.com/v1/bpi/currentprice.json")
    loop_interval_seconds: float = float(os.getenv("LOOP_INTERVAL_SECONDS", "5"))

    # 👇 isto silencia o warning e permite campos como model_path
    model_config = ConfigDict(protected_namespaces=())

    # ai flags
    use_ai: bool = os.getenv("USE_AI", "false").lower() == "true"
    model_path: str = os.getenv("MODEL_PATH", "models/gbm.pkl")
    buy_threshold: float = float(os.getenv("BUY_THRESHOLD", "0.55"))
    sell_threshold: float = float(os.getenv("SELL_THRESHOLD", "0.55"))
    feature_horizon: int = int(os.getenv("FEATURE_HORIZON", "10"))
    cooldown_seconds: int = int(os.getenv("COOLDOWN_SECONDS", "60"))
    min_proba_gap: float = float(os.getenv("MIN_PROBA_GAP", "0.10"))

settings = Settings()