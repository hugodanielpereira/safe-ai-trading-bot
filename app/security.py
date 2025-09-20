# app/security.py
from fastapi import Header, HTTPException, status
from .config import settings
import logging

logger = logging.getLogger(__name__)

async def verify_token(x_api_token: str | None = Header(default=None)):
    if x_api_token is None or x_api_token != settings.bot_api_token:
        logger.warning("Auth failed: received token %r", x_api_token)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
        )